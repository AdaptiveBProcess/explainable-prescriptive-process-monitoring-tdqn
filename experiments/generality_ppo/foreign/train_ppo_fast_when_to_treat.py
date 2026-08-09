"""
Fast PPO training for WhenToTreat using BPIC 2017.

Bypasses original envManager (hardcoded /home/zdashtbozorg/... paths)
and baselineEnv (TensorFlow dependency) by reimplementing env logic
directly from the CSV. Faithful to the original reward and state design.

Usage:
    python train_ppo_fast.py [--timesteps 10000]
"""

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

CSV_DEFAULT = Path(__file__).parent / "RL" / "data" / "results_adaptive_counterfacs_bpic2017.csv"
RESULTS_DIR = Path(__file__).parent / "RL" / "results"


class WTTEnvFast(gym.Env):
    """
    Minimal Gym env for WhenToTreat.

    State (3-D): [relative_position, lower_TE, upper_TE]
    Action (Discrete 2): 0 = do nothing, 1 = treat
    Reward: replicates compute_reward() from state_with_temp_costReward_withoutPreds.py
    """

    metadata = {"render_modes": []}

    def __init__(self, csv_path: Path | str = CSV_DEFAULT):
        super().__init__()
        df = pd.read_csv(csv_path)
        # sort event by event as original envManager does
        df = df.sort_values(["Case ID", "event_nr"]).reset_index(drop=True)
        self._df = df
        self._max_idx = len(df) - 1

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=np.array([0.0, -1e6, -1e6], dtype=np.float32),
            high=np.array([1.0, 1e6, 1e6], dtype=np.float32),
        )
        self._idx = 0

    # ── helpers ──────────────────────────────────────────────────────────────

    def _row(self):
        return self._df.iloc[self._idx]

    def _state(self, row=None) -> np.ndarray:
        if row is None:
            row = self._row()
        rel = float(row["event_nr"]) / max(float(row["case_length"]), 1.0)
        rel = float(np.clip(rel, 0.0, 1.0))
        return np.array(
            [rel, float(row["lower"]), float(row["upper"])],
            dtype=np.float32,
        )

    @staticmethod
    def _reward(adapted: bool, ite: float, actual_outcome: float) -> float:
        cost, gain = 25.0, 50.0
        if adapted:
            if ite > 0:
                return (gain * ite) - cost
            elif ite == 0:
                return -cost
            else:
                return -cost - gain
        else:
            if ite > 0:
                return -gain
            elif ite == 0:
                return gain if actual_outcome == 1 else 0.0
            else:
                return gain

    # ── gym API ───────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._idx = 0
        return self._state(), {}

    def step(self, action):
        row = self._row()
        adapted = bool(action == 1)

        y0, y1 = float(row["y0"]), float(row["y1"])
        ite = y1 - y0
        actual_outcome = y1 if adapted else y0

        reward = self._reward(adapted, ite, actual_outcome)

        is_last = int(row["event_nr"]) >= int(row["case_length"])
        terminated = adapted or is_last

        self._idx = min(self._idx + 1, self._max_idx)
        obs = self._state()
        return obs, reward, terminated, False, {}

    def render(self):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timesteps",
        type=int,
        default=10_000,
        help="Total PPO training timesteps (default 10 000 = fast test)",
    )
    parser.add_argument("--csv", type=str, default=str(CSV_DEFAULT))
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        print("Run preprocess_bpic2017.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading env from {csv_path} ...")
    env = WTTEnvFast(csv_path)
    check_env(env, warn=True)

    print(f"Training PPO for {args.timesteps:,} timesteps ...")
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=512,
        batch_size=64,
        n_epochs=5,
        learning_rate=3e-4,
        gamma=0.99,
        verbose=1,
    )
    model.learn(total_timesteps=args.timesteps)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = RESULTS_DIR / "ppo_bpic2017_fast"
    model.save(str(save_path))
    print(f"\n✅ Model saved → {save_path}.zip")
    print("   Load with: SB3PPOAgent.from_rl_ppm(results_dir='RL/results/')")


if __name__ == "__main__":
    main()
