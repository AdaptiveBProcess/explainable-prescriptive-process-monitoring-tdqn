"""
Fast PPO training for RL-prescriptive-monitoring using BPIC 2017.

Bypasses the original envManager_v2 (argv/hardcoded paths) and
baselineEnv (TensorFlow dependency) by reimplementing the env logic
directly from the CSV. Faithful to the original reward and state design.

Usage:
    python train_ppo_fast.py [--timesteps 10000] [--resources 3]
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

CSV_DEFAULT = Path(__file__).parent / "rl" / "data" / "ready_to_use_adaptive_bpic2017.csv"
RESULTS_DIR = Path(__file__).parent / "results"


class PPMEnvFast(gym.Env):
    """
    Minimal Gym env for RL-prescriptive-monitoring.

    State (4-D): [relative_position, reliability, deviation, n_resources]
    Action (Discrete 2): 0 = do nothing, 1 = intervene
    Reward: replicates compute_reward() from state_with_temp_costReward_withoutPreds.py
    """

    metadata = {"render_modes": []}

    def __init__(self, csv_path: Path | str = CSV_DEFAULT, resources: int = 3):
        super().__init__()
        df = pd.read_csv(csv_path, sep=";")
        # sort event-by-event as original envManager_v2 does
        df = df.sort_values(["orig_timestamp", "prefix_nr"]).reset_index(drop=True)
        self._df = df
        self._max_idx = len(df) - 1
        self._resources_init = resources

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=np.array([0.0, -1e6, -1e6, 0.0], dtype=np.float32),
            high=np.array([1.0, 1e6, 1e6, float(resources)], dtype=np.float32),
        )
        self._idx = 0
        self._nr_res = list(range(1, resources + 1))

    # ── helpers ──────────────────────────────────────────────────────────────

    def _row(self):
        return self._df.iloc[self._idx]

    def _state(self, row=None) -> np.ndarray:
        if row is None:
            row = self._row()
        rel = float(row["prefix_nr"]) / max(float(row["case_length"]), 1.0)
        rel = float(np.clip(rel, 0.0, 1.0))
        return np.array(
            [rel, float(row["reliability"]), float(row["deviation"]), float(len(self._nr_res))],
            dtype=np.float32,
        )

    @staticmethod
    def _reward(adapted: bool, ite: float, nr_res: list) -> float:
        cost, gain, gain_res = 25.0, 50.0, 50.0
        has_res = len(nr_res) > 0
        if adapted:
            if ite > 0:
                r = (gain * ite) - cost + (gain_res if has_res else -gain_res)
            elif ite == 0:
                r = -cost - gain_res
            else:
                r = -cost - gain - gain_res
        else:
            if ite > 0:
                r = -gain - (gain_res if has_res else gain_res)
            elif ite == 0:
                r = gain_res if has_res else 0.0
            else:
                r = gain + (gain_res if has_res else gain_res)
        return float(r)

    # ── gym API ───────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._idx = 0
        self._nr_res = list(range(1, self._resources_init + 1))
        return self._state(), {}

    def step(self, action):
        row = self._row()
        adapted = bool(action == 1)

        y0, y1 = float(row["y0"]), float(row["y1"])
        ite = y1 - y0

        if adapted and self._nr_res:
            self._nr_res.pop(0)

        reward = self._reward(adapted, ite, self._nr_res)

        # episode ends when case is treated OR prefix reaches end of trace
        is_last = int(row["prefix_nr"]) >= int(row["case_length"])
        terminated = adapted or is_last

        # advance index (wrap around to enable continuous collection)
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
    parser.add_argument("--resources", type=int, default=3)
    parser.add_argument("--csv", type=str, default=str(CSV_DEFAULT))
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading env from {csv_path} ...")
    env = PPMEnvFast(csv_path, resources=args.resources)
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
    print("   Load with: SB3PPOAgent.from_rl_ppm(results_dir='results/')")


if __name__ == "__main__":
    main()
