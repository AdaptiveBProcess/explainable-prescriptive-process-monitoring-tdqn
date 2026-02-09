import argparse

from xppm.utils.logging import get_logger
from xppm.utils.config import Config
from xppm.ope.doubly_robust import doubly_robust_estimate
from xppm.ope.report import save_ope_report


logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config).raw
    metrics = doubly_robust_estimate(
        cfg["experiment"].get("checkpoint_path", "artifacts/checkpoints/Q_theta.ckpt"),
        cfg["data"]["offline_dataset_path"],
    )
    save_ope_report(metrics, cfg["experiment"].get("ope_report_path", "artifacts/ope/ope_dr.json"))
    logger.info("05_run_ope_dr completed: %s", metrics)


if __name__ == "__main__":
    main()


