import argparse

from xppm.utils.logging import get_logger
from xppm.utils.config import Config
from xppm.xai.fidelity_tests import run_fidelity_tests


logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config).raw
    # Stub: this will raise NotImplementedError until you implement fidelity tests.
    run_fidelity_tests(cfg)
    logger.info("07_fidelity_tests completed.")


if __name__ == "__main__":
    main()


