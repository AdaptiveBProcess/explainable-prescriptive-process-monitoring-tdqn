"""Phase 3 - Step 6: Explain Policy (Risk, DeltaQ, Policy Summary)."""

import argparse

from xppm.utils.config import Config
from xppm.utils.logging import get_logger
from xppm.xai.explain_policy import explain_policy

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate XAI artifacts: risk, deltaQ explanations and policy summary",
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to Q_theta checkpoint (overrides config)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Split to use (default: from config or 'test')",
    )
    parser.add_argument(
        "--n-cases",
        type=int,
        default=None,
        help="Number of cases to explain (default: from config or 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for case selection (default: from config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for XAI artifacts (overrides config)",
    )
    args = parser.parse_args()

    config_obj = Config.from_yaml(args.config)
    cfg = config_obj.raw

    # CLI overrides
    if args.ckpt:
        cfg.setdefault("xai", {})["checkpoint_path"] = args.ckpt
    if args.split:
        cfg.setdefault("xai", {})["split"] = args.split
    if args.n_cases is not None:
        cfg.setdefault("xai", {})["n_cases"] = args.n_cases
    if args.seed is not None:
        cfg.setdefault("xai", {})["seed"] = args.seed
    if args.output_dir:
        cfg.setdefault("xai", {})["out_dir"] = args.output_dir

    outputs = explain_policy(cfg, config_hash=config_obj.config_hash)
    logger.info("06_explain_policy completed. Outputs: %s", list(outputs.keys()))


if __name__ == "__main__":
    main()
