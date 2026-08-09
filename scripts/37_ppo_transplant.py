"""Sec. 4.5 generality study: dual-level explanations on external PPO agents.

Thin entry point for experiments/generality_ppo/run_dual_level_ppo.py, numbered
for discoverability next to the other paper scripts. Regenerates
artifacts/generality/{rl-prescriptive-monitoring,when-to-treat}.json.

Requires the sb3 environment (experiments/generality_ppo/requirements.txt),
NOT the repo's .venv — see experiments/generality_ppo/README.md.
"""

import runpy
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "experiments/generality_ppo/run_dual_level_ppo.py"

if __name__ == "__main__":
    sys.argv[0] = str(SCRIPT)
    runpy.run_path(str(SCRIPT), run_name="__main__")
