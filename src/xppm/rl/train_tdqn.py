"""Offline TDQN training with Double-DQN, action masking, and comprehensive logging."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

from xppm.rl.base import RLTrainer
from xppm.rl.models.masking import apply_action_mask
from xppm.rl.models.transformer import SimpleTransformerEncoder
from xppm.utils.io import load_json, load_npz
from xppm.utils.logging import ensure_dir, get_logger
from xppm.utils.seed import set_seed

logger = get_logger(__name__)


@dataclass
class TDQNConfig:
    """TDQN training configuration."""

    # Dataset
    npz_path: str | Path
    splits_path: str | Path
    vocab_path: str | Path

    # Model architecture
    max_len: int = 50
    vocab_size: int = 1000
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    dropout: float = 0.1
    n_actions: int = 2

    # Training
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    gamma: float = 0.99
    max_steps: int = 200000
    eval_every: int = 5000
    save_every: int = 10000

    # Stabilization
    double_dqn: bool = True
    target_update_every: int = 2000
    grad_clip_norm: float = 10.0

    # Loss
    loss_type: str = "huber"  # huber | mse

    # LR scheduler
    lr_scheduler_enabled: bool = True
    lr_scheduler_type: str = "cosine"  # cosine | step | none
    warmup_steps: int = 2000

    # Device
    device: str = "cuda"

    # Reproducibility
    seed: int = 42
    deterministic: bool = False


class TransformerQNetwork(nn.Module):
    """Q-network with Transformer encoder for sequence states."""

    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        n_actions: int,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model
        self.n_actions = n_actions

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Transformer encoder
        self.encoder = SimpleTransformerEncoder(
            input_dim=d_model, hidden_dim=d_model, n_layers=n_layers, dropout=dropout
        )

        # State representation: use last token or mean pooling
        # For now, use last token (CLS-style)
        self.state_proj = nn.Linear(d_model, d_model)

        # Q-network head
        self.q_head = nn.Linear(d_model, n_actions)

    def forward(self, states: torch.Tensor, state_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        Args:
            states: (batch, max_len) token IDs
            state_mask: (batch, max_len) 1 for real tokens, 0 for padding

        Returns:
            Q-values: (batch, n_actions)
        """
        # Embed tokens (clamp indices to valid range)
        # Clamp to [0, vocab_size-1] to avoid out-of-bounds errors
        states_clamped = torch.clamp(states, min=0, max=self.vocab_size - 1)
        x = self.embedding(states_clamped)  # (batch, max_len, d_model)

        # Encode with transformer
        encoded = self.encoder(x)  # (batch, max_len, d_model)

        # Pool: use last non-padded token
        if state_mask is not None:
            # Get last real token index per sequence
            lengths = state_mask.sum(dim=1).long() - 1  # -1 for 0-indexing
            lengths = torch.clamp(lengths, min=0, max=self.max_len - 1)
            batch_indices = torch.arange(encoded.size(0), device=encoded.device)
            state_repr = encoded[batch_indices, lengths]  # (batch, d_model)
        else:
            # Fallback: use last token
            state_repr = encoded[:, -1]  # (batch, d_model)

        # Project to state space
        state_repr = self.state_proj(state_repr)
        state_repr = torch.relu(state_repr)

        # Q-values
        q_values = self.q_head(state_repr)  # (batch, n_actions)

        return q_values


def load_dataset_with_splits(
    npz_path: str | Path,
    splits_path: str | Path,
    split_name: str = "train",
) -> dict[str, np.ndarray | torch.Tensor]:
    """Load dataset and filter by split (case-based, no leakage).

    Args:
        npz_path: Path to D_offline.npz
        splits_path: Path to splits.json
        split_name: Which split to load ("train", "val", "test")

    Returns:
        Dictionary with filtered transitions
    """
    # Load splits
    splits = load_json(splits_path)
    split_cases = set(splits["cases"][split_name])

    # Load dataset
    data = load_npz(npz_path)

    # Filter by case_id (case_ptr)
    case_ptr = data["case_ptr"]
    mask = np.isin(case_ptr, list(split_cases))

    logger.info(
        "Loaded %s split: %d cases, %d transitions",
        split_name,
        len(split_cases),
        mask.sum(),
    )

    # Extract filtered data
    filtered = {
        "s": data["s"][mask],
        "s_mask": data["s_mask"][mask],
        "a": data["a"][mask],
        "r": data["r"][mask],
        "s_next": data["s_next"][mask],
        "s_next_mask": data["s_next_mask"][mask],
        "done": data["done"][mask],
        "valid_actions": data["valid_actions"][mask],
    }

    # Optional fields
    if "case_ptr" in data:
        filtered["case_ptr"] = data["case_ptr"][mask]
    if "t_ptr" in data:
        filtered["t_ptr"] = data["t_ptr"][mask]

    return filtered


def create_data_loader(
    data: dict[str, np.ndarray | torch.Tensor],
    batch_size: int,
    shuffle: bool = True,
    device: str = "cuda",
) -> DataLoader:
    """Create PyTorch DataLoader from numpy arrays."""

    # Convert to tensors (handle both numpy arrays and tensors)
    def to_tensor(x: np.ndarray | torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(dtype=dtype)
        return torch.from_numpy(x).to(dtype=dtype)

    dataset = TensorDataset(
        to_tensor(data["s"], torch.long),
        to_tensor(data["s_mask"], torch.float),
        to_tensor(data["a"], torch.long),
        to_tensor(data["r"], torch.float),
        to_tensor(data["s_next"], torch.long),
        to_tensor(data["s_next_mask"], torch.float),
        to_tensor(data["done"], torch.float),
        to_tensor(data["valid_actions"], torch.float),
    )

    # Optimize DataLoader for GPU
    use_gpu = device == "cuda" and torch.cuda.is_available()
    num_workers = 4 if use_gpu else 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=use_gpu,  # Faster data transfer to GPU
        num_workers=num_workers,  # Parallel data loading for GPU
        persistent_workers=use_gpu and num_workers > 0,  # Keep workers alive
    )


def compute_q_stats(q_values: torch.Tensor) -> dict[str, float]:
    """Compute Q-value statistics."""
    q_flat = q_values.detach().cpu().flatten()
    return {
        "mean": float(q_flat.mean()),
        "std": float(q_flat.std()),
        "min": float(q_flat.min()),
        "max": float(q_flat.max()),
        "abs_mean": float(q_flat.abs().mean()),
    }


def compute_grad_stats(model: nn.Module) -> dict[str, float]:
    """Compute gradient statistics."""
    total_norm = 0.0
    n_params = 0

    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            n_params += 1

    total_norm = total_norm ** (1.0 / 2)

    return {
        "global_norm": float(total_norm),
        "n_params": n_params,
    }


def train_tdqn(
    config: TDQNConfig,
    checkpoint_dir: Path | None = None,
    config_hash: str | None = None,
    dataset_hash: str | None = None,
    vocab_hash: str | None = None,
    tracker: Any = None,  # Tracker from logging module
) -> dict[str, Any]:
    """Train TDQN offline with Double-DQN, action masking, and comprehensive logging.

    Args:
        config: TDQN configuration

    Returns:
        Training metrics and metadata
    """
    # Set seed
    set_seed(config.seed, deterministic=config.deterministic)

    # Device: prioritize GPU if available and requested
    if config.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("✅ Using GPU: %s", torch.cuda.get_device_name(0))
        logger.info("   CUDA version: %s", torch.version.cuda)
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info("   GPU memory: %.1f GB", gpu_memory_gb)
    elif config.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        logger.warning("⚠️  CUDA requested but not available, falling back to CPU")
    else:
        device = torch.device("cpu")
        logger.info("Using device: CPU")

    # Load vocabulary to get vocab_size
    vocab = load_json(config.vocab_path)
    vocab_size = len(vocab.get("token2id", {}))
    if vocab_size == 0:
        # Fallback: estimate from dataset
        data_sample = load_npz(config.npz_path)
        vocab_size = int(data_sample["s"].max() + 1)
        logger.warning("Could not get vocab_size from vocab, estimated: %d", vocab_size)

    # Check for vocabulary mismatch with dataset
    logger.info("Checking vocabulary compatibility with dataset...")
    data_sample = load_npz(config.npz_path)
    s_max = int(data_sample["s"].max())
    s_next_max = int(data_sample["s_next"].max())
    dataset_max_id = max(s_max, s_next_max)

    if dataset_max_id >= vocab_size:
        # Calculate OOV rate
        s_oov = (data_sample["s"] >= vocab_size).sum()
        s_next_oov = (data_sample["s_next"] >= vocab_size).sum()
        total_tokens_s = data_sample["s"].size
        total_tokens_s_next = data_sample["s_next"].size
        oov_rate_s = s_oov / total_tokens_s if total_tokens_s > 0 else 0.0
        oov_rate_s_next = s_next_oov / total_tokens_s_next if total_tokens_s_next > 0 else 0.0

        logger.warning("=" * 60)
        logger.warning("⚠️  VOCABULARY MISMATCH DETECTED")
        logger.warning("=" * 60)
        logger.warning("Vocab size: %d (IDs 0-%d)", vocab_size, vocab_size - 1)
        logger.warning("Dataset max ID: %d (in s) / %d (in s_next)", s_max, s_next_max)
        logger.warning(
            "OOV rate in s: %.6f (%.4f%%) - %d tokens out of %d",
            oov_rate_s,
            oov_rate_s * 100,
            s_oov,
            total_tokens_s,
        )
        logger.warning(
            "OOV rate in s_next: %.6f (%.4f%%) - %d tokens out of %d",
            oov_rate_s_next,
            oov_rate_s_next * 100,
            s_next_oov,
            total_tokens_s_next,
        )
        logger.warning("")
        logger.warning("⚠️  The dataset was created with a different vocabulary!")
        logger.warning("   The clamp in forward() will map OOV tokens to vocab_size-1, which")
        logger.warning("   may degrade training quality.")
        logger.warning("")
        if oov_rate_s > 0.01 or oov_rate_s_next > 0.01:
            logger.error("❌ OOV rate > 1%% - Training quality will be significantly degraded!")
            logger.error("   RECOMMENDED: Regenerate prefixes.npz and D_offline.npz with the")
            logger.error("   current vocabulary to ensure consistency.")
        elif oov_rate_s > 0.001 or oov_rate_s_next > 0.001:
            logger.warning("⚠️  OOV rate > 0.1%% - Consider regenerating the dataset.")
        else:
            logger.info("ℹ️  OOV rate < 0.1%% - Acceptable for now, but monitor training.")
        logger.warning("=" * 60)
        logger.warning("")

    # Update config with actual vocab_size
    config.vocab_size = vocab_size

    # Load datasets
    logger.info("Loading training dataset...")
    train_data = load_dataset_with_splits(config.npz_path, config.splits_path, "train")
    train_loader = create_data_loader(
        train_data, config.batch_size, shuffle=True, device=str(device)
    )

    logger.info("Loading validation dataset...")
    val_data = load_dataset_with_splits(config.npz_path, config.splits_path, "val")
    val_loader = create_data_loader(val_data, config.batch_size, shuffle=False, device=str(device))

    # Log split statistics for leakage verification
    splits = load_json(config.splits_path)
    train_cases = set(splits["cases"]["train"])
    val_cases = set(splits["cases"]["val"])
    test_cases = set(splits["cases"]["test"])

    n_train_cases = len(train_cases)
    n_val_cases = len(val_cases)
    n_test_cases = len(test_cases)
    n_train_transitions = len(train_data["a"])
    n_val_transitions = len(val_data["a"])

    # Verify no leakage (case overlap)
    case_overlap_train_val = len(train_cases & val_cases)
    case_overlap_train_test = len(train_cases & test_cases)
    case_overlap_val_test = len(val_cases & test_cases)

    logger.info("=" * 60)
    logger.info("SPLIT STATISTICS (Leakage Check)")
    logger.info("=" * 60)
    logger.info("Cases: train=%d, val=%d, test=%d", n_train_cases, n_val_cases, n_test_cases)
    logger.info("Transitions: train=%d, val=%d", n_train_transitions, n_val_transitions)
    logger.info("Case overlap (train∩val): %d", case_overlap_train_val)
    logger.info("Case overlap (train∩test): %d", case_overlap_train_test)
    logger.info("Case overlap (val∩test): %d", case_overlap_val_test)

    if case_overlap_train_val > 0 or case_overlap_train_test > 0 or case_overlap_val_test > 0:
        logger.error("❌ LEAKAGE DETECTED: Cases overlap between splits!")
        raise ValueError("Dataset split has leakage - cases appear in multiple splits")

    logger.info("✅ No leakage: all splits are disjoint")
    logger.info("=" * 60)

    # Log to tracker if available
    if tracker and hasattr(tracker, "log_metrics") and tracker.enabled:
        tracker.log_metrics(
            {
                "n_train_cases": n_train_cases,
                "n_val_cases": n_val_cases,
                "n_test_cases": n_test_cases,
                "n_train_transitions": n_train_transitions,
                "n_val_transitions": n_val_transitions,
                "case_overlap": 0,  # Verified to be 0
            },
            step=0,
        )

    # Initialize models
    q_net = TransformerQNetwork(
        vocab_size=config.vocab_size,
        max_len=config.max_len,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        dropout=config.dropout,
        n_actions=config.n_actions,
    ).to(device)

    target_q_net = TransformerQNetwork(
        vocab_size=config.vocab_size,
        max_len=config.max_len,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        dropout=config.dropout,
        n_actions=config.n_actions,
    ).to(device)

    # Initialize target network with online network weights
    target_q_net.load_state_dict(q_net.state_dict())
    target_q_net.eval()

    # Optimizer
    optimizer = optim.AdamW(
        q_net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    # LR scheduler
    scheduler: optim.lr_scheduler.LRScheduler | None = None
    if config.lr_scheduler_enabled:
        if config.lr_scheduler_type == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_steps)
        elif config.lr_scheduler_type == "step":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.1)

    # Loss function
    loss_fn: nn.Module
    if config.loss_type == "huber":
        loss_fn = nn.HuberLoss(reduction="mean", delta=1.0)
    else:
        loss_fn = nn.MSELoss()

    # Training loop
    global_step = 0
    history: list[dict[str, Any]] = []

    logger.info("Starting training for %d steps", config.max_steps)

    while global_step < config.max_steps:
        for batch in train_loader:
            if global_step >= config.max_steps:
                break

            s, s_mask, a, r, s_next, s_next_mask, done, valid_actions = [
                b.to(device) for b in batch
            ]

            # Forward pass: Q(s, a)
            q_values = q_net(s, s_mask)  # (batch, n_actions)
            q_a = q_values.gather(1, a.unsqueeze(1)).squeeze(1)  # (batch,)

            # Check mask sanity
            valid_mask = valid_actions.gather(1, a.unsqueeze(1)).squeeze(1)  # (batch,)
            n_invalid = (valid_mask < 0.5).sum().item()
            if n_invalid > 0:
                logger.error(
                    "Found %d invalid actions in batch (mask sanity check failed)!",
                    n_invalid,
                )
                raise ValueError("Invalid actions in batch - dataset corruption?")

            # Double-DQN target
            with torch.no_grad():
                # Online network selects action (with mask)
                q_next_online = q_net(s_next, s_next_mask)  # (batch, n_actions)
                q_next_online_masked = apply_action_mask(q_next_online, valid_actions)
                a_star = q_next_online_masked.argmax(dim=1)  # (batch,)

                # Target network evaluates action
                q_next_target = target_q_net(s_next, s_next_mask)  # (batch, n_actions)
                q_next_target_masked = apply_action_mask(q_next_target, valid_actions)
                q_next = q_next_target_masked.gather(1, a_star.unsqueeze(1)).squeeze(1)  # (batch,)

                # Target
                target = r + config.gamma * (1.0 - done) * q_next  # (batch,)

            # Loss
            loss = loss_fn(q_a, target)

            # Check for NaN/inf
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error("Loss is NaN or Inf! Aborting training.")
                raise ValueError("Loss exploded - check learning rate and reward scale")

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Grad clipping
            grad_norm = clip_grad_norm_(q_net.parameters(), config.grad_clip_norm)
            was_clipped = grad_norm > config.grad_clip_norm

            optimizer.step()

            # Update target network
            if global_step % config.target_update_every == 0:
                target_q_net.load_state_dict(q_net.state_dict())
                logger.info(
                    "🔄 Updated target network (step %d, every %d steps)",
                    global_step,
                    config.target_update_every,
                )

            # LR scheduler step
            if scheduler is not None:
                scheduler.step()

            # Logging
            if global_step % 100 == 0:
                # Compute stats
                q_stats = compute_q_stats(q_values)
                grad_stats = compute_grad_stats(q_net)
                target_stats = compute_q_stats(target.unsqueeze(1))

                metrics = {
                    "loss/td": float(loss.item()),
                    "q/mean": q_stats["mean"],
                    "q/std": q_stats["std"],
                    "q/min": q_stats["min"],
                    "q/max": q_stats["max"],
                    "q/abs_mean": q_stats["abs_mean"],
                    "grad/global_norm": grad_stats["global_norm"],
                    "grad/pct_clipped": float(100.0 if was_clipped else 0.0),
                    "target/mean": target_stats["mean"],
                    "target/std": target_stats["std"],
                    "mask/pct_invalid_in_batch": float(n_invalid / len(a) * 100),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                }

                history.append({**metrics, "step": global_step})

                logger.info(
                    "Step %d: loss=%.4f, q_mean=%.2f, q_max=%.2f, grad_norm=%.2f",
                    global_step,
                    metrics["loss/td"],
                    metrics["q/mean"],
                    metrics["q/max"],
                    metrics["grad/global_norm"],
                )

                # Log to tracker
                if tracker and hasattr(tracker, "log_metrics") and tracker.enabled:
                    tracker.log_metrics(metrics, step=global_step)

            # Save checkpoint periodically
            if (
                checkpoint_dir is not None
                and global_step % config.save_every == 0
                and global_step > 0
            ):
                save_checkpoint(
                    q_net,
                    target_q_net,
                    optimizer,
                    global_step,
                    epoch=0,
                    checkpoint_dir=checkpoint_dir,
                    config=config,
                    config_hash=config_hash,
                    dataset_hash=dataset_hash,
                    vocab_hash=vocab_hash,
                )

            # Validation
            if global_step % config.eval_every == 0 and global_step > 0:
                # Skip validation if val set is empty
                if len(val_data["a"]) == 0:
                    logger.warning("Skipping validation: val set is empty (0 transitions)")
                else:
                    q_net.eval()
                    val_losses = []
                    val_q_stats_list = []

                    with torch.no_grad():
                        for val_batch in val_loader:
                            vs, vs_mask, va, vr, vs_next, vs_next_mask, vdone, vvalid = [
                                b.to(device) for b in val_batch
                            ]

                            vq_values = q_net(vs, vs_mask)
                            vq_a = vq_values.gather(1, va.unsqueeze(1)).squeeze(1)

                            vq_next_online = q_net(vs_next, vs_next_mask)
                            vq_next_online_masked = apply_action_mask(vq_next_online, vvalid)
                            va_star = vq_next_online_masked.argmax(dim=1)

                            vq_next_target = target_q_net(vs_next, vs_next_mask)
                            vq_next_target_masked = apply_action_mask(vq_next_target, vvalid)
                            vq_next = vq_next_target_masked.gather(1, va_star.unsqueeze(1)).squeeze(
                                1
                            )

                            vtarget = vr + config.gamma * (1.0 - vdone) * vq_next
                            vloss = loss_fn(vq_a, vtarget)

                            val_losses.append(vloss.item())
                            val_q_stats_list.append(compute_q_stats(vq_values))

                    if val_losses:
                        val_loss = np.mean(val_losses)
                        val_q_mean = np.mean([s["mean"] for s in val_q_stats_list])

                        logger.info(
                            "Validation (step %d): loss=%.4f, q_mean=%.2f",
                            global_step,
                            val_loss,
                            val_q_mean,
                        )

                        val_metrics = {
                            "step": global_step,
                            "val/loss": val_loss,
                            "val/q_mean": val_q_mean,
                        }
                        history.append(val_metrics)

                        # Log validation metrics to tracker
                        if tracker and hasattr(tracker, "log_metrics") and tracker.enabled:
                            tracker.log_metrics(
                                {"val/loss": val_loss, "val/q_mean": val_q_mean},
                                step=global_step,
                            )

                    q_net.train()

            global_step += 1

    logger.info("Training completed after %d steps", global_step)

    return {
        "history": history,
        "final_step": global_step,
        "q_net": q_net,
        "target_q_net": target_q_net,
        "optimizer": optimizer,
    }


def save_checkpoint(
    q_net: nn.Module,
    target_q_net: nn.Module,
    optimizer: optim.Optimizer,
    global_step: int,
    epoch: int,
    checkpoint_dir: Path,
    config: TDQNConfig,
    config_hash: str | None = None,
    dataset_hash: str | None = None,
    vocab_hash: str | None = None,
) -> dict[str, Path]:
    """Save checkpoint with proper structure.

    Returns:
        Dictionary with paths to saved files
    """
    ensure_dir(checkpoint_dir)

    # Save model checkpoints
    q_ckpt_path = checkpoint_dir / "Q_theta.ckpt"
    target_ckpt_path = checkpoint_dir / "target_Q.ckpt"

    torch.save(
        {
            "algorithm": "tdqn",
            "model_state_dict": q_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step,
            "epoch": epoch,
            "config_hash": config_hash,
            "dataset_hash": dataset_hash,
            "vocab_hash": vocab_hash,
            "n_actions": config.n_actions,
            "max_len": config.max_len,
            "vocab_size": config.vocab_size,
            "d_model": config.d_model,
        },
        q_ckpt_path,
    )

    torch.save(
        {
            "model_state_dict": target_q_net.state_dict(),
            "global_step": global_step,
        },
        target_ckpt_path,
    )

    logger.info("Saved checkpoints to %s", checkpoint_dir)

    return {
        "q_theta": q_ckpt_path,
        "target_q": target_ckpt_path,
    }


class TDQNTrainer(RLTrainer):
    """Concrete ``RLTrainer`` implementation wrapping the TDQN training loop.

    This thin wrapper makes TDQN pluggable via the same interface as future
    algorithms (CQL, BCQ, SAC, …) without changing any internal training logic.
    """

    def train(self, config: Any, **kwargs: Any) -> dict[str, Any]:
        """Delegate to :func:`train_tdqn`."""
        return train_tdqn(config, **kwargs)

    def save_checkpoint(self, path: Path, **kwargs: Any) -> dict[str, Path]:
        """Delegate to :func:`save_checkpoint`."""
        return save_checkpoint(**kwargs)
