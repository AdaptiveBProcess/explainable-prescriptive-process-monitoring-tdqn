import argparse

from xppm.utils.logging import get_logger
from xppm.utils.config import Config
from xppm.distill.distill_policy import distill_policy


logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config).raw
    # Stub: this will raise NotImplementedError until you implement distillation.
    distill_policy(cfg)
    logger.info("08_distill_policy completed.")


if __name__ == "__main__":
    main()


