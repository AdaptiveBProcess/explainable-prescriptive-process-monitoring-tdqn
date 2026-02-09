import argparse

from xppm.utils.logging import get_logger
from xppm.utils.config import Config
from xppm.data.encode_prefixes import encode_prefixes


logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config).raw
    encode_prefixes(cfg["data"]["cleaned_log_path"], cfg["data"]["prefixes_path"])
    logger.info("02_encode_prefixes completed.")


if __name__ == "__main__":
    main()


