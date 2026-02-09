import argparse

from xppm.utils.logging import get_logger
from xppm.utils.config import Config
from xppm.xai.explain_policy import explain_policy


logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config).raw
    explain_policy(
        cfg["experiment"].get("checkpoint_path", "artifacts/checkpoints/Q_theta.ckpt"),
        cfg["data"]["offline_dataset_path"],
        cfg["experiment"].get("xai_output_path", "artifacts/xai/risk_explanations.json"),
    )
    logger.info("06_explain_policy completed.")


if __name__ == "__main__":
    main()


