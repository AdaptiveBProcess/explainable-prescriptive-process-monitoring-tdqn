import argparse

from xppm.utils.logging import get_logger
from xppm.utils.config import Config
from xppm.rl.train_tdqn import TDQNConfig, train_tdqn


logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config).raw
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
    logger.info("04_train_tdqn_offline completed.")


if __name__ == "__main__":
    main()


