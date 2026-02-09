import argparse

from xppm.utils.logging import get_logger
from xppm.utils.config import Config
from xppm.data.build_mdp import build_mdp_dataset


logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config).raw
    build_mdp_dataset(cfg["data"]["prefixes_path"], cfg["data"]["offline_dataset_path"], cfg["data"]["splits_path"])
    logger.info("03_build_mdp_dataset completed.")


if __name__ == "__main__":
    main()


