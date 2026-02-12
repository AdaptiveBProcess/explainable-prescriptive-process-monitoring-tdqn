import argparse
from pathlib import Path

from xppm.ope.behavior_model import BehaviorPolicy, fit_behavior_policy_tdqn_encoder
from xppm.ope.doubly_robust import doubly_robust_estimate
from xppm.ope.report import save_ope_report
from xppm.utils.config import Config
from xppm.utils.io import load_json
from xppm.utils.logging import get_logger

logger = get_logger(__name__)


def _resolve_checkpoint_path(cfg: dict, ckpt_arg: str | None) -> Path:
    if ckpt_arg:
        return Path(ckpt_arg)
    ckpt_cfg = cfg.get("experiment", {}).get(
        "checkpoint_path", "artifacts/models/tdqn/20260209_191903/Q_theta.ckpt"
    )
    return Path(ckpt_cfg)


def _resolve_vocab_path(cfg: dict, ckpt_path: Path) -> Path:
    # Prefer the vocab bundled with the checkpoint run (same folder)
    ckpt_dir = ckpt_path.parent
    vocab_in_ckpt = ckpt_dir / "vocab_activity.json"
    if vocab_in_ckpt.exists():
        return vocab_in_ckpt

    # Fallback to config encoding output
    encoding_cfg = cfg.get("encoding", {})
    vocab_cfg = encoding_cfg.get("output", {}).get("vocab_activity_path")
    if vocab_cfg is not None:
        return Path(vocab_cfg)

    # Last resort: default interim vocab
    return Path("data/interim/vocab_activity.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2 - Step 5: Off-Policy Evaluation (Doubly Robust)"
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to Q_theta checkpoint. If not provided, taken from config.",
    )
    parser.add_argument(
        "--rho-cap",
        type=float,
        default=20.0,
        help="Truncation cap for IS weights (rho).",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=200,
        help="Number of bootstrap samples for CI.",
    )
    parser.add_argument(
        "--pi-e-temperature",
        type=float,
        default=None,
        help="Temperature for π_e softmax policy. If not provided, taken from config.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to OPE report JSON. If not provided, taken from config.",
    )
    args = parser.parse_args()

    config_obj = Config.from_yaml(args.config)
    cfg = config_obj.raw

    # Resolve dataset (D_offline.npz) and splits.json from config
    paths_cfg = cfg.get("paths", {})
    data_processed_dir = Path(paths_cfg.get("data_processed_dir", "data/processed"))

    mdp_cfg = cfg.get("mdp", {})
    mdp_out = mdp_cfg.get("output", {})
    dataset_rel = mdp_out.get("path", str(data_processed_dir / "D_offline.npz"))
    dataset_path = Path(dataset_rel)

    split_cfg = cfg.get("validation_split", {})
    split_file = split_cfg.get("out_splits_json", "splits.json")
    splits_path = data_processed_dir / split_file
    ckpt_path = _resolve_checkpoint_path(cfg, args.ckpt)
    vocab_path = _resolve_vocab_path(cfg, ckpt_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"D_offline.npz not found at {dataset_path}")
    if not splits_path.exists():
        raise FileNotFoundError(f"splits.json not found at {splits_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary not found at {vocab_path}")

    logger.info("Starting OPE DR with:")
    logger.info("  Dataset:   %s", dataset_path)
    logger.info("  Splits:    %s", splits_path)
    logger.info("  Checkpoint:%s", ckpt_path)
    logger.info("  Vocab:     %s", vocab_path)

    # 1) Fit behavior policy π_b(a|s) using TDQN encoder (train/val only)
    behavior: BehaviorPolicy = fit_behavior_policy_tdqn_encoder(
        npz_path=dataset_path,
        splits_path=splits_path,
        ckpt_path=ckpt_path,
        vocab_path=vocab_path,
        config=cfg,
    )
    logger.info(
        "Behavior model fitted: val_nll=%.4f, val_acc=%.4f, val_entropy=%.4f",
        behavior.metrics.get("val_nll", 0.0),
        behavior.metrics.get("val_acc", 0.0),
        behavior.metrics.get("val_mean_entropy", 0.0),
    )

    # 2) Run DR + WIS on TEST split
    # Override pi_e_temperature if provided via CLI
    if args.pi_e_temperature is not None:
        if "ope" not in cfg:
            cfg["ope"] = {}
        cfg["ope"]["pi_e_temperature"] = float(args.pi_e_temperature)

    metrics = doubly_robust_estimate(
        ckpt_path=ckpt_path,
        dataset_path=dataset_path,
        splits_path=splits_path,
        vocab_path=vocab_path,
        config=cfg,
        behavior=behavior,
        rho_cap=float(args.rho_cap),
        n_bootstrap=int(args.bootstrap),
    )

    # 3) Enrich with metadata (hashes, config) if available
    metadata: dict[str, object] = {
        "run_id": cfg.get("experiment", {}).get("run_id"),
        "ckpt_path": str(ckpt_path),
        "dataset_path": str(dataset_path),
        "splits_path": str(splits_path),
        "vocab_path": str(vocab_path),
        "config_hash": config_obj.config_hash,
    }
    # Try to include dataset hash if present in build report
    try:
        mdp_report = load_json("artifacts/reports/mdp_build_report.json")
        metadata["dataset_hash"] = mdp_report.get("dataset_hash")
    except FileNotFoundError:
        pass

    full_report = {
        "metadata": metadata,
        "results": metrics.get("results", {}),
        "diagnostics": metrics.get("diagnostics", {}),
        "n_bootstrap": metrics.get("n_bootstrap", 0),
    }

    # 4) Save report
    output_path = args.output or cfg.get("experiment", {}).get(
        "ope_report_path", "artifacts/ope/ope_dr.json"
    )
    save_ope_report(full_report, output_path)
    logger.info("05_run_ope_dr completed. Report saved to %s", output_path)


if __name__ == "__main__":
    main()
