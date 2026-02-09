from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn, optim

from xppm.rl.models.q_network import QNetwork
from xppm.rl.replay import ReplayBuffer
from xppm.utils.io import fingerprint_data, get_dvc_hashes, get_git_commit, load_npz
from xppm.utils.logging import (
    ensure_dir,
    finalize_run_metadata,
    get_logger,
    init_tracker,
    save_run_metadata,
    start_run_metadata,
)
from xppm.utils.seed import set_seed

logger = get_logger(__name__)


@dataclass
class TDQNConfig:
    state_dim: int
    n_actions: int
    hidden_dim: int = 128
    gamma: float = 0.99
    learning_rate: float = 3e-4
    batch_size: int = 128
    max_epochs: int = 1  # keep very small as a smoke default


def load_replay(dataset_path: str | Path, seed: int | None = None) -> ReplayBuffer:
    """Load replay buffer with optional seed for deterministic sampling."""
    data = load_npz(dataset_path)
    rng = None
    if seed is not None:
        rng = np.random.default_rng(seed)
    return ReplayBuffer(
        states=data["states"],
        actions=data["actions"],
        rewards=data["rewards"],
        next_states=data["next_states"],
        dones=data["dones"],
        rng=rng,
    )


def train_tdqn(
    config: TDQNConfig,
    dataset_path: str | Path,
    checkpoint_path: str | Path,
    seed: int = 42,
    deterministic: bool = False,
    config_hash: str | None = None,
    metadata_output: str | Path | None = None,
    tracking_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Minimal offline TDQN training loop (stub: single-network DQN-style).
    
    Args:
        config: TDQN configuration
        dataset_path: Path to offline dataset
        checkpoint_path: Where to save checkpoint
        seed: Random seed
        deterministic: Enable deterministic algorithms
        config_hash: Config hash for metadata
        metadata_output: Optional path to save run metadata
    """
    set_seed(seed, deterministic=deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize metadata tracking
    data_fp = fingerprint_data([dataset_path])
    metadata = start_run_metadata(
        stage="train",
        config_path="config",
        config_hash=config_hash or "unknown",
        seed=seed,
        deterministic=deterministic,
        data_fingerprint=data_fp,
    )
    
    # Initialize experiment tracker (MLflow/W&B)
    tracker = None
    if tracking_config:
        tracker = init_tracker(tracking_config)
        if tracker.enabled:
            git_info = get_git_commit()
            run_name = f"train_{metadata['run_id']}"
            
            # Get DVC hashes if available
            dvc_hashes = get_dvc_hashes({
                "clean_parquet": "data/interim/clean.parquet.dvc",
                "rlset": "data/processed/D_offline.npz.dvc",
                "splits": "data/processed/splits.json.dvc",
            })
            
            # Prepare tags and params
            commit_hash = git_info["commit"]
            commit_short = commit_hash[:8] if isinstance(commit_hash, str) else "unknown"
            tags = {
                "stage": "train",
                "git_commit": commit_short,
                "git_dirty": str(git_info["dirty"]),
            }
            tags.update({f"dvc_{k}": v for k, v in dvc_hashes.items()})
            
            params = {
                "seed": seed,
                "deterministic": deterministic,
                "config_hash": config_hash or "unknown",
                "data_fingerprint": data_fp,
                **{k: v for k, v in config.__dict__.items()},
            }
            
            tracker.init_run(run_name=run_name, stage="train", tags=tags, params=params)

    replay = load_replay(dataset_path, seed=seed)
    q_net = QNetwork(config.state_dim, config.n_actions, config.hidden_dim).to(device)
    target_q_net = QNetwork(config.state_dim, config.n_actions, config.hidden_dim).to(device)
    target_q_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()

    n_samples = len(replay.states)
    logger.info("Starting TDQN training on %d samples (stub loop).", n_samples)

    for epoch in range(config.max_epochs):
        batch = replay.sample(config.batch_size)
        states = torch.from_numpy(batch["states"]).float().to(device)
        actions = torch.from_numpy(batch["actions"]).long().to(device)
        rewards = torch.from_numpy(batch["rewards"]).float().to(device)
        next_states = torch.from_numpy(batch["next_states"]).float().to(device)
        dones = torch.from_numpy(batch["dones"]).float().to(device)

        q_values = q_net(states).gather(1, actions)
        with torch.no_grad():
            next_q = target_q_net(next_states).max(1, keepdim=True).values
            target = rewards + config.gamma * (1.0 - dones) * next_q

        loss = loss_fn(q_values, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_val = float(loss.item())
        logger.info("Epoch %d - loss: %.6f", epoch + 1, loss_val)
        
        # Log metrics to tracker
        if tracker and tracker.enabled:
            tracker.log_metrics({"loss": loss_val, "epoch": epoch + 1}, step=epoch)

    ensure_dir(Path(checkpoint_path).parent)
    torch.save(q_net.state_dict(), checkpoint_path)
    logger.info("Saved checkpoint to %s", checkpoint_path)
    
    # Finalize and save metadata
    metrics = {"final_loss": float(loss.item())}
    metadata = finalize_run_metadata(
        metadata,
        outputs=[checkpoint_path],
        metrics=metrics,
    )
    
    # Save metadata file
    if metadata_output:
        save_run_metadata(metadata, metadata_output)
    else:
        # Default location
        meta_path = Path(checkpoint_path).parent / f"{Path(checkpoint_path).stem}.meta.json"
        save_run_metadata(metadata, meta_path)
    
    # Log final metrics and artifacts to tracker
    if tracker and tracker.enabled and tracking_config:
        tracker.log_metrics(metrics)
        if tracking_config.get("log_artifacts", True):
            tracker.log_artifact(checkpoint_path, artifact_path="checkpoints")
            tracker.log_artifact(meta_path, artifact_path="metadata")
            # Log config if enabled
            if tracking_config.get("log_config", True):
                # Save resolved config (would need config dict passed in)
                pass
        tracker.finish()
    
    return metrics


