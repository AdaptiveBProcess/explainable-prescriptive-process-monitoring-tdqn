import argparse

from xppm.utils.logging import get_logger
from xppm.utils.config import Config
from xppm.data.preprocess import preprocess_event_log
from xppm.data.validate_split import validate_and_split


logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config).raw
    preprocess_event_log(cfg["data"]["event_log_path"], cfg["data"]["cleaned_log_path"])
    validate_and_split(cfg["data"]["cleaned_log_path"], cfg["data"]["splits_path"])
    logger.info("01_preprocess_log completed.")


if __name__ == "__main__":
    main()


