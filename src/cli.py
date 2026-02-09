from __future__ import annotations

import argparse

from pathlib import Path

from xppm.data.preprocess import preprocess_event_log
from xppm.data.encode_prefixes import encode_prefixes
from xppm.data.build_mdp import build_mdp_dataset
from xppm.data.validate_split import validate_and_split
from xppm.ope.doubly_robust import doubly_robust_estimate
from xppm.ope.report import save_ope_report
from xppm.rl.train_tdqn import TDQNConfig, train_tdqn
from xppm.utils.config import Config
from xppm.utils.logging import get_logger


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
    cfg = Config.from_yaml(args.config).raw

    if args.command == "preprocess":
        preprocess_event_log(cfg["data"]["event_log_path"], cfg["data"]["cleaned_log_path"])
        validate_and_split(cfg["data"]["cleaned_log_path"], cfg["data"]["splits_path"])
    elif args.command == "encode":
        encode_prefixes(cfg["data"]["cleaned_log_path"], cfg["data"]["prefixes_path"])
    elif args.command == "build_mdp":
        build_mdp_dataset(cfg["data"]["prefixes_path"], cfg["data"]["offline_dataset_path"], cfg["data"]["splits_path"])
    elif args.command == "train":
        # Minimal mapping from config to TDQNConfig; adjust as you refine config structure.
        tdqn_cfg = TDQNConfig(
            state_dim=int(cfg.get("model", {}).get("state_dim", 1)),
            n_actions=len(cfg.get("policy", {}).get("action_space", [])) or 1,
            hidden_dim=int(cfg.get("model", {}).get("hidden_dim", 128)),
            gamma=float(cfg.get("rl", {}).get("gamma", 0.99)),
            learning_rate=float(cfg.get("rl", {}).get("learning_rate", 3e-4)),
            batch_size=int(cfg.get("rl", {}).get("batch_size", 128)),
            max_epochs=int(cfg.get("rl", {}).get("max_epochs", 1)),
        )
        train_tdqn(
            tdqn_cfg,
            cfg["data"]["offline_dataset_path"],
            cfg["experiment"].get("checkpoint_path", "artifacts/checkpoints/Q_theta.ckpt"),
        )
    elif args.command == "ope":
        metrics = doubly_robust_estimate(
            cfg["experiment"].get("checkpoint_path", "artifacts/checkpoints/Q_theta.ckpt"),
            cfg["data"]["offline_dataset_path"],
        )
        save_ope_report(metrics, cfg["experiment"].get("ope_report_path", "artifacts/ope/ope_dr.json"))
        logger.info("OPE metrics: %s", metrics)


if __name__ == "__main__":
    main()


