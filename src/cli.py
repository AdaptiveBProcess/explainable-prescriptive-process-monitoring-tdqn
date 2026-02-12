from __future__ import annotations

import argparse

from xppm.data.build_mdp import build_mdp_dataset
from xppm.data.encode_prefixes import encode_prefixes
from xppm.data.preprocess import preprocess_event_log
from xppm.ope.doubly_robust import doubly_robust_estimate
from xppm.ope.report import save_ope_report
from xppm.utils.config import Config
from xppm.utils.logging import get_logger
from xppm.utils.seed import set_seed

# Training is done via scripts/04_train_tdqn_offline.py, not via CLI

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(prog="xppm")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=str, default="configs/config.yaml")

    subparsers.add_parser("preprocess", parents=[common])
    subparsers.add_parser("encode", parents=[common])
    subparsers.add_parser("build_mdp", parents=[common])
    subparsers.add_parser("train", parents=[common])
    subparsers.add_parser("ope", parents=[common])

    args = parser.parse_args()
    config_obj = Config.from_yaml(args.config)
    cfg = config_obj.raw

    # Set seed and deterministic mode from config (reproducibility)
    # Support both new 'repro' section and legacy 'project' section
    repro_cfg = cfg.get("repro", {})
    if not repro_cfg:
        # Fallback to legacy location
        repro_cfg = cfg.get("project", {})
    seed = repro_cfg.get("seed", 42)
    deterministic = repro_cfg.get("deterministic", False)
    set_seed(seed, deterministic=deterministic)
    logger.info("Set seed=%d, deterministic=%s", seed, deterministic)

    if args.command == "preprocess":
        preprocess_event_log(cfg["data"]["event_log_path"], cfg["data"]["cleaned_log_path"])
        # Note: validate_and_split_dataset is called separately in step 01b
        logger.info("Preprocessing completed. Run step 01b to validate and split.")
    elif args.command == "encode":
        encode_prefixes(cfg["data"]["cleaned_log_path"], cfg["data"]["prefixes_path"])
    elif args.command == "build_mdp":
        # build_mdp_dataset requires clean_log_path, vocab_path, and config
        mdp_cfg = cfg.get("mdp", {})
        encoding_cfg = cfg.get("encoding", {})
        vocab_default = "data/interim/vocab_activity.json"
        vocab_path = encoding_cfg.get("output", {}).get("vocab_activity_path", vocab_default)
        build_mdp_dataset(
            prefixes_path=cfg["data"]["prefixes_path"],
            clean_log_path=cfg["data"]["cleaned_log_path"],
            vocab_path=vocab_path,
            output_path=mdp_cfg.get("output", {}).get("path", cfg["data"]["offline_dataset_path"]),
            config=mdp_cfg,
        )
    elif args.command == "train":
        # Use the training script directly instead of CLI
        logger.warning(
            "Training via CLI is deprecated. Use scripts/04_train_tdqn_offline.py instead."
        )
        logger.info("Example: python scripts/04_train_tdqn_offline.py --config %s", args.config)
    elif args.command == "ope":
        metrics = doubly_robust_estimate(
            cfg["experiment"].get("checkpoint_path", "artifacts/checkpoints/Q_theta.ckpt"),
            cfg["data"]["offline_dataset_path"],
        )
        save_ope_report(
            metrics,
            cfg["experiment"].get("ope_report_path", "artifacts/ope/ope_dr.json"),
        )
        logger.info("OPE metrics: %s", metrics)


if __name__ == "__main__":
    main()
