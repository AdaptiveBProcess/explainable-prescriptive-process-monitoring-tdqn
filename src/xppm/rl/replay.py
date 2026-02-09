from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ReplayBuffer:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        idx = np.random.randint(0, len(self.states), size=batch_size)
        return {
            "states": self.states[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_states": self.next_states[idx],
            "dones": self.dones[idx],
        }


