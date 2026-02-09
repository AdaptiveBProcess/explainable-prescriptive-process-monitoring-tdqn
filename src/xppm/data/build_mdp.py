from __future__ import annotations

from pathlib import Path

import numpy as np

from xppm.utils.io import save_npz
from xppm.utils.logging import get_logger


logger = get_logger(__name__)


def build_mdp_dataset(prefixes_path: str | Path, output_path: str | Path, splits_path: str | Path) -> None:
    """Placeholder for 03_build_mdp_dataset logic."""
    logger.info("Building MDP dataset from prefixes at %s", prefixes_path)
    # TODO: implement trajectory assembly and MDP state-action construction
    states = np.zeros((1, 1), dtype="float32")
    actions = np.zeros((1, 1), dtype="int64")
    rewards = np.zeros((1, 1), dtype="float32")
    next_states = np.zeros((1, 1), dtype="float32")
    dones = np.zeros((1, 1), dtype="bool")
    save_npz(output_path, states=states, actions=actions, rewards=rewards, next_states=next_states, dones=dones)
    Path(splits_path).parent.mkdir(parents=True, exist_ok=True)
    Path(splits_path).write_text('{"train": [], "val": [], "test": []}\n', encoding="utf-8")


