"""Phase 3 - Step 7: Fidelity Tests (Q-drop, Action-flip, Rank-consistency).

NOTE (paper provenance): this script's fidelity.csv uses the original SIGNED
Q-drop convention and is NOT the source of any number in the ICPM paper. The
paper's fidelity/evaluability artifacts use the |displacement| convention of
Defs. 3-4 and come from scripts 23-26/28 and 32 (see docs/paper/MAPPING.md,
`make paper-tables`). Kept as the pipeline's operational smoke test.
"""

import argparse

from xppm.utils.config import Config
from xppm.utils.logging import get_logger
from xppm.xai.fidelity_tests import run_fidelity_tests

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run fidelity tests: Q-drop, action-flip, rank-consistency",
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name. Loads configs/datasets/{name}.yaml on top of --config.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to Q_theta checkpoint (overrides config)",
    )
    parser.add_argument(
        "--xai-dir",
        type=str,
        default=None,
        help="Directory with XAI outputs (overrides config, looks for final/ subdir)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (overrides config)",
    )
    parser.add_argument(
        "--n-items",
        type=int,
        default=None,
        help="Limit number of items for smoke test (default: all)",
    )
    parser.add_argument(
        "--p-remove",
        type=float,
        nargs="+",
        default=None,
        help="Removal percentages (default: [0.1, 0.2, 0.3, 0.5])",
    )
    parser.add_argument(
        "--n-random",
        type=int,
        default=None,
        help="Number of random repetitions (default: 20, use 5 for smoke test)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output (verbose logging of perturbations, Q-values, actions)",
    )
    args = parser.parse_args()

    config_obj = Config.for_dataset(args.config, args.dataset)
    cfg = config_obj.raw

    # CLI overrides
    if args.ckpt:
        cfg.setdefault("xai", {})["checkpoint_path"] = args.ckpt
    if args.xai_dir:
        # Update xai out_dir to point to provided directory
        cfg.setdefault("xai", {})["out_dir"] = args.xai_dir
    if args.output:
        cfg.setdefault("fidelity", {})["out_csv"] = args.output
    if args.n_items is not None:
        cfg.setdefault("fidelity", {})["n_items"] = args.n_items
    if args.p_remove is not None:
        cfg.setdefault("fidelity", {})["p_remove"] = args.p_remove
    if args.n_random is not None:
        cfg.setdefault("fidelity", {})["n_random"] = args.n_random
    if args.debug:
        cfg.setdefault("fidelity", {})["debug"] = True

    run_fidelity_tests(cfg, config_obj=config_obj)
    logger.info("07_fidelity_tests completed.")
    logger.warning(
        "fidelity.csv uses the legacy SIGNED Q-drop convention and does not "
        "back the paper's Table 2 (|displacement| convention, scripts 23-26/28/32); "
        "see docs/paper/MAPPING.md before comparing against the paper."
    )


if __name__ == "__main__":
    main()
