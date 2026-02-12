"""Train TDQN offline (Phase 2 - Step 4)."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from xppm.rl.train_tdqn import TDQNConfig, save_checkpoint, train_tdqn
from xppm.utils.config import Config
from xppm.utils.io import fingerprint_data, get_git_commit, load_json
from xppm.utils.logging import (
    ensure_dir,
    finalize_run_metadata,
    get_logger,
    init_tracker,
    save_run_metadata,
    start_run_metadata,
)

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train TDQN offline with Double-DQN and action masking (Phase 2 - Step 4)"
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing checkpoint directory if it exists",
    )
    args = parser.parse_args()

    # Load config
    config_obj = Config.from_yaml(args.config)
    cfg = config_obj.raw

    # Setup reproducibility
    seed = cfg.get("repro", {}).get("seed", 42)
    deterministic = cfg.get("repro", {}).get("deterministic", False)

    # Get paths
    paths_cfg = cfg.get("paths", {})
    training_cfg = cfg.get("training", {})
    tdqn_cfg = training_cfg.get("tdqn", {})
    transformer_cfg = training_cfg.get("transformer", {})
    mdp_cfg = cfg.get("mdp", {})

    npz_path = mdp_cfg.get("output", {}).get("path", "data/processed/D_offline.npz")
    splits_path = Path(paths_cfg.get("data_processed_dir", "data/processed")) / "splits.json"
    encoding_output = cfg.get("encoding", {}).get("output", {})
    vocab_activity_path = encoding_output.get("vocab_activity_path", "vocab_activity.json")
    # Handle both absolute and relative paths
    vocab_path_str = str(vocab_activity_path)
    if Path(vocab_path_str).is_absolute():
        vocab_path = Path(vocab_path_str)
    elif vocab_path_str.startswith("data/"):
        vocab_path = Path(vocab_path_str)
    else:
        vocab_path = Path(paths_cfg.get("data_interim_dir", "data/interim")) / vocab_path_str

    # Create checkpoint directory with run ID
    artifacts_dir = Path(paths_cfg.get("artifacts_dir", "artifacts"))
    models_dir = artifacts_dir / "models" / "tdqn"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = models_dir / run_id

    # Protection: don't overwrite real checkpoints accidentally
    if checkpoint_dir.exists():
        output_str = str(checkpoint_dir.resolve())
        is_safe_test_path = (
            "/tmp/pytest-" in output_str
            or "/tmp/tmp" in output_str
            or "/.tmp/" in output_str
            or output_str.endswith("/.tmp")
            or "/artifacts/tests/" in output_str
        )

        if not is_safe_test_path:
            if not args.overwrite:
                raise FileExistsError(
                    f"Checkpoint directory {checkpoint_dir} already exists. "
                    f"To overwrite, use --overwrite flag."
                )
            else:
                logger.warning("Overwriting existing checkpoint directory: %s", checkpoint_dir)
                shutil.rmtree(checkpoint_dir)

    ensure_dir(checkpoint_dir)

    # Build TDQNConfig from config.yaml
    vocab = load_json(vocab_path)
    vocab_size = len(vocab.get("token2id", {}))

    tdqn_config = TDQNConfig(
        npz_path=npz_path,
        splits_path=splits_path,
        vocab_path=vocab_path,
        max_len=transformer_cfg.get("max_len", 50),
        vocab_size=vocab_size,
        d_model=transformer_cfg.get("d_model", 128),
        n_heads=transformer_cfg.get("n_heads", 4),
        n_layers=transformer_cfg.get("n_layers", 3),
        dropout=transformer_cfg.get("dropout", 0.1),
        n_actions=len(mdp_cfg.get("actions", {}).get("id2name", [])),
        batch_size=training_cfg.get("batch_size", 256),
        learning_rate=tdqn_cfg.get("learning_rate", 3e-4),
        weight_decay=tdqn_cfg.get("weight_decay", 0.0),
        gamma=tdqn_cfg.get("gamma", 0.99),
        max_steps=training_cfg.get("max_steps", 200000),
        eval_every=training_cfg.get("eval_every", 5000),
        save_every=training_cfg.get("save_every", 10000),
        double_dqn=tdqn_cfg.get("double_dqn", True),
        target_update_every=tdqn_cfg.get("target_update_every", 2000),
        grad_clip_norm=tdqn_cfg.get("grad_clip_norm", 10.0),
        loss_type="huber",
        lr_scheduler_enabled=tdqn_cfg.get("lr_scheduler", {}).get("enabled", True),
        lr_scheduler_type=tdqn_cfg.get("lr_scheduler", {}).get("type", "cosine"),
        warmup_steps=tdqn_cfg.get("lr_scheduler", {}).get("warmup_steps", 2000),
        device=training_cfg.get("device", "cuda"),
        seed=seed,
        deterministic=deterministic,
    )

    # Initialize tracking
    tracking_cfg = cfg.get("tracking", {})
    tracker = None
    if tracking_cfg.get("enabled", False):
        tracker = init_tracker(tracking_cfg)
        if tracker and tracker.enabled:
            git_info = get_git_commit()
            run_name = f"train_tdqn_{run_id}"
            commit_hash = git_info["commit"]
            commit_short = commit_hash[:8] if isinstance(commit_hash, str) else "unknown"
            config_hash = config_obj.hash_config()

            tags = {
                "stage": "phase2_train_tdqn",
                "git_commit": commit_short,
                "git_dirty": str(git_info["dirty"]),
                "config_hash": config_hash[:8],
            }

            # Add data hashes
            try:
                npz_hash = fingerprint_data([npz_path])[:16]
                splits_hash = fingerprint_data([splits_path])[:16]
                vocab_hash = fingerprint_data([vocab_path])[:16]
                tags["npz_hash"] = npz_hash[:8]
                tags["splits_hash"] = splits_hash[:8]
                tags["vocab_hash"] = vocab_hash[:8]
            except Exception as e:
                logger.warning("Could not compute data hashes: %s", e)

            params = {
                "seed": seed,
                "deterministic": deterministic,
                "config_hash": config_hash,
                "batch_size": tdqn_config.batch_size,
                "learning_rate": tdqn_config.learning_rate,
                "gamma": tdqn_config.gamma,
                "max_steps": tdqn_config.max_steps,
                "double_dqn": tdqn_config.double_dqn,
                "target_update_every": tdqn_config.target_update_every,
                "grad_clip_norm": tdqn_config.grad_clip_norm,
            }

            tracker.init_run(run_name=run_name, stage="phase2_train_tdqn", tags=tags, params=params)

    # Start run metadata
    metadata = start_run_metadata(
        stage="train_tdqn",
        config_path=args.config,
        config_hash=config_obj.hash_config(),
        seed=seed,
        deterministic=deterministic,
        data_fingerprint=(
            fingerprint_data([npz_path, splits_path, vocab_path])
            if all(Path(p).exists() for p in [npz_path, splits_path, vocab_path])
            else None
        ),
    )

    try:
        # Prepare hashes
        config_hash = config_obj.hash_config()
        npz_hash = fingerprint_data([npz_path])[:16] if Path(npz_path).exists() else None
        vocab_hash = fingerprint_data([vocab_path])[:16] if vocab_path.exists() else None

        # Train
        logger.info("Starting TDQN training...")
        results = train_tdqn(
            tdqn_config,
            checkpoint_dir=checkpoint_dir,
            config_hash=config_hash,
            dataset_hash=npz_hash,
            vocab_hash=vocab_hash,
            tracker=tracker,
        )

        # Save final checkpoint
        checkpoint_paths = save_checkpoint(
            results["q_net"],
            results["target_q_net"],
            results["optimizer"],
            results["final_step"],
            epoch=0,  # We use steps, not epochs
            checkpoint_dir=checkpoint_dir,
            config=tdqn_config,
            config_hash=config_hash,
            dataset_hash=npz_hash,
            vocab_hash=vocab_hash,
        )

        # Copy artifacts to checkpoint directory
        shutil.copy(args.config, checkpoint_dir / "config.yaml")
        shutil.copy(splits_path, checkpoint_dir / "splits.json")
        shutil.copy(vocab_path, checkpoint_dir / "vocab_activity.json")

        # Save metrics/history
        history_path = checkpoint_dir / "history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(results["history"], f, indent=2)

        logger.info("Saved training history to %s", history_path)

        # Log metrics to tracker
        if tracker and tracker.enabled:
            # Log final metrics
            if results["history"]:
                final_metrics = results["history"][-1]
                tracker.log_metrics(final_metrics, step=results["final_step"])

            # Log artifacts
            if tracking_cfg.get("log_artifacts", True):
                tracker.log_artifact(str(checkpoint_paths["q_theta"]), artifact_path="checkpoints")
                tracker.log_artifact(str(checkpoint_paths["target_q"]), artifact_path="checkpoints")
                tracker.log_artifact(str(history_path), artifact_path="metadata")
                tracker.log_artifact(str(checkpoint_dir / "config.yaml"), artifact_path="metadata")
                tracker.log_artifact(str(checkpoint_dir / "splits.json"), artifact_path="metadata")
                tracker.log_artifact(
                    str(checkpoint_dir / "vocab_activity.json"), artifact_path="metadata"
                )

        # Finalize metadata
        metadata = finalize_run_metadata(
            metadata,
            outputs=[str(checkpoint_paths["q_theta"]), str(checkpoint_paths["target_q"])],
            metrics={
                "final_step": results["final_step"],
                "n_history_entries": len(results["history"]),
            },
        )
        metadata["status"] = "completed"

        # Save run metadata
        run_metadata_path = checkpoint_dir / "run_metadata.json"
        save_run_metadata(metadata, run_metadata_path)

        logger.info("✅ Training completed successfully")
        logger.info("   Checkpoint directory: %s", checkpoint_dir)
        logger.info("   Final step: %d", results["final_step"])

    except Exception as e:
        logger.error("❌ Training failed: %s", e, exc_info=True)
        metadata["status"] = "failed"
        metadata["error"] = str(e)
        run_metadata_path = checkpoint_dir / "run_metadata.json"
        save_run_metadata(metadata, run_metadata_path)
        if tracker and tracker.enabled:
            tracker.finish()
        raise

    finally:
        if tracker and tracker.enabled:
            tracker.finish()


if __name__ == "__main__":
    main()
