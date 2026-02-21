from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from xppm.rl.factory import AgentFactory
from xppm.rl.train_tdqn import TransformerQNetwork
from xppm.utils.io import load_json, load_npz


@dataclass
class BehaviorPolicy:
    """Behavior policy π_b(a|s) estimated over all transitions."""

    probs: np.ndarray
    n_actions: int
    metrics: dict[str, float]


class _BehaviorDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
):
    """Small Dataset wrapper to feed transitions to the behavior model."""

    def __init__(
        self,
        s: np.ndarray,
        s_mask: np.ndarray,
        valid_actions: np.ndarray,
        a: np.ndarray,
        indices: np.ndarray,
    ) -> None:
        self.s = s
        self.s_mask = s_mask
        self.valid_actions = valid_actions
        self.a = a
        self.indices = indices

    def __len__(self) -> int:  # type: ignore[override]
        return int(self.indices.shape[0])

    def __getitem__(  # type: ignore[override]
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        i = int(self.indices[idx])
        return (
            torch.from_numpy(self.s[i]).long(),
            torch.from_numpy(self.s_mask[i]).float(),
            torch.from_numpy(self.valid_actions[i]).float(),
            torch.tensor(int(self.a[i]), dtype=torch.long),
            torch.tensor(i, dtype=torch.long),
        )


class _BehaviorHead(nn.Module):
    """Simple policy head on top of frozen TDQN encoder."""

    def __init__(self, d_model: int, n_actions: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(d_model, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def fit_behavior_policy_tdqn_encoder(
    npz_path: str | Path,
    splits_path: str | Path,
    ckpt_path: str | Path,
    vocab_path: str | Path,
    config: dict[str, Any],
    device: torch.device | None = None,
) -> BehaviorPolicy:
    """Fit behavior policy π_b(a|s) using frozen TDQN encoder + softmax head.

    This implements the recommended Option A from 2-2-setup:
    - Reuse the TDQN encoder to obtain state embeddings z.
    - Train a small classifier head on top (softmax over actions).
    - Train on TRAIN split, evaluate on VAL split.

    Returns a BehaviorPolicy with π_b(a|s_t) for all transitions t.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_npz(npz_path)
    splits = load_json(splits_path)

    n_transitions = int(data["s"].shape[0])
    n_actions = int(data["valid_actions"].shape[1])

    # Build masks from case_id assignments in splits.json
    case_ids = data["case_ptr"]
    train_cases = set(int(c) for c in splits["cases"]["train"])
    val_cases = set(int(c) for c in splits["cases"]["val"])

    train_mask = np.isin(case_ids, list(train_cases))
    val_mask = np.isin(case_ids, list(val_cases))

    # Load frozen encoder from checkpoint via factory (builds + loads weights)
    q_net: TransformerQNetwork = AgentFactory.load(  # type: ignore[assignment]
        ckpt_path, npz_path, vocab_path, config, device
    )
    for p in q_net.parameters():
        p.requires_grad = False

    d_model = q_net.d_model
    head = _BehaviorHead(d_model=d_model, n_actions=n_actions, dropout=0.1).to(device)

    # Training hyperparameters (can be tuned later)
    behavior_cfg = config.get("behavior_model", {})
    batch_size = int(behavior_cfg.get("batch_size", 1024))
    epochs = int(behavior_cfg.get("epochs", 1))
    lr = float(behavior_cfg.get("learning_rate", 1e-3))
    label_smoothing = float(behavior_cfg.get("label_smoothing", 0.1))

    # Datasets / loaders
    train_indices = np.nonzero(train_mask)[0]
    val_indices = np.nonzero(val_mask)[0]

    train_ds = _BehaviorDataset(
        data["s"],
        data["s_mask"],
        data["valid_actions"],
        data["a"],
        train_indices,
    )
    val_ds = _BehaviorDataset(
        data["s"],
        data["s_mask"],
        data["valid_actions"],
        data["a"],
        val_indices,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)

    def _encode_states(
        s_batch: torch.Tensor,
        s_mask_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Encode states using frozen TDQN encoder to get state representation z."""
        with torch.no_grad():
            states_clamped = torch.clamp(s_batch, min=0, max=q_net.vocab_size - 1)
            x = q_net.embedding(states_clamped)
            encoded = q_net.encoder(x)

            # Use same pooling as TransformerQNetwork
            lengths = s_mask_batch.sum(dim=1).long() - 1
            lengths = torch.clamp(lengths, min=0, max=q_net.max_len - 1)
            batch_indices = torch.arange(encoded.size(0), device=encoded.device)
            state_repr = encoded[batch_indices, lengths]
            state_repr = q_net.state_proj(state_repr)
            state_repr = torch.relu(state_repr)
        return state_repr

    def _evaluate(loader: DataLoader) -> tuple[float, float, float]:
        head.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        total_entropy = 0.0
        with torch.no_grad():
            for s_b, s_mask_b, v_b, a_b, _ in loader:
                s_b = s_b.to(device)
                s_mask_b = s_mask_b.to(device)
                v_b = v_b.to(device)
                a_b = a_b.to(device)

                z = _encode_states(s_b, s_mask_b)
                logits = head(z)
                # Mask invalid actions with finite large negative value (not -inf)
                logits = logits.masked_fill(v_b == 0, -1e9)

                # Compute NLL stably using log_softmax (avoid using `loss` directly
                # to keep metrics consistent and reduce risk of inf/nan)
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                # Extract log_prob for logged action
                batch_indices = torch.arange(a_b.size(0), device=a_b.device)
                nll_stable = -log_probs[batch_indices, a_b]
                # Replace any inf/nan with a large finite value
                nll_stable = torch.clamp(nll_stable, min=0.0, max=1e6)

                probs = torch.softmax(logits, dim=-1)
                entropy = -(probs * torch.log(torch.clamp(probs, min=1e-8))).sum(dim=-1).mean()

                preds = probs.argmax(dim=-1)
                correct = (preds == a_b).sum().item()

                n = int(a_b.shape[0])
                # Use stable NLL for metrics
                total_loss += float(nll_stable.mean().item()) * n
                total_correct += int(correct)
                total_samples += n
                total_entropy += float(entropy.item()) * n

        if total_samples == 0:
            return 0.0, 0.0, 0.0
        avg_loss = total_loss / total_samples
        acc = total_correct / total_samples
        mean_entropy = total_entropy / total_samples
        return avg_loss, acc, mean_entropy

    # Training loop (single epoch by default)
    for _ in range(epochs):
        head.train()
        for s_b, s_mask_b, v_b, a_b, _ in train_loader:
            s_b = s_b.to(device)
            s_mask_b = s_mask_b.to(device)
            v_b = v_b.to(device)
            a_b = a_b.to(device)

            optimizer.zero_grad(set_to_none=True)
            z = _encode_states(s_b, s_mask_b)
            logits = head(z)
            # Use finite large negative value instead of -inf for stability
            logits = logits.masked_fill(v_b == 0, -1e9)
            loss = criterion(logits, a_b)
            loss.backward()
            optimizer.step()

    val_nll, val_acc, val_entropy = _evaluate(val_loader)

    # Compute π_b(a|s) for all transitions (train + val + test)
    all_indices = np.arange(n_transitions, dtype=int)
    full_ds = _BehaviorDataset(
        data["s"],
        data["s_mask"],
        data["valid_actions"],
        data["a"],
        all_indices,
    )
    full_loader = DataLoader(
        full_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    probs_all = np.zeros((n_transitions, n_actions), dtype=np.float32)
    head.eval()
    with torch.no_grad():
        for s_b, s_mask_b, v_b, _, idx_b in full_loader:
            s_b = s_b.to(device)
            s_mask_b = s_mask_b.to(device)
            v_b = v_b.to(device)

            z = _encode_states(s_b, s_mask_b)
            logits = head(z)
            # Use finite large negative value instead of -inf for stability
            logits = logits.masked_fill(v_b == 0, -1e9)
            probs = torch.softmax(logits, dim=-1)

            idx_np = idx_b.numpy()
            probs_all[idx_np] = probs.cpu().numpy()

    metrics = {
        "val_nll": float(val_nll),
        "val_acc": float(val_acc),
        "val_mean_entropy": float(val_entropy),
    }

    return BehaviorPolicy(probs=probs_all, n_actions=n_actions, metrics=metrics)
