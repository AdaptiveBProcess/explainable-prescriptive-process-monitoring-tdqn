"""Run the pipeline fidelity tests (Q-drop, action-flip) on the baseline
attribution methods produced by scripts/21_baseline_attributions.py.

Reuses run_fidelity_tests unchanged; only the config paths are overridden so
each run reads the baseline XAI dir and the checkpoint-bundled vocabulary
(the per-dataset interim vocabs are stale for simbank).
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from xppm.utils.config import Config  # noqa: E402
from xppm.xai.fidelity_tests import run_fidelity_tests  # noqa: E402

OPE_PATHS = {
    "simbank": REPO / "artifacts/ope/ope_dr.json",
    "bpi2017": REPO / "artifacts/ope/bpi2017/ope_dr.json",
    "bpi2020-rfp": REPO / "artifacts/ope/bpi2020-rfp/ope_dr.json",
}


def main():
    datasets = sys.argv[1:] or list(OPE_PATHS)
    for ds in datasets:
        ope = json.load(open(OPE_PATHS[ds]))
        ckpt = REPO / ope["metadata"]["ckpt_path"]
        vocab = REPO / ope["metadata"]["vocab_path"]
        for method in ("saliency", "attention"):
            xai_dir = REPO / "artifacts/xai/baselines" / method / ds
            out_csv = REPO / "artifacts/fidelity/baselines" / method / ds / "fidelity.csv"
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            print(f"==== {ds} / {method}")

            config_obj = Config.for_dataset(str(REPO / "configs/config.yaml"), ds)
            cfg = config_obj.raw
            cfg.setdefault("xai", {})["checkpoint_path"] = str(ckpt)
            cfg["xai"]["out_dir"] = str(xai_dir)
            cfg.setdefault("encoding", {}).setdefault("output", {})["vocab_activity_path"] = str(
                vocab
            )
            fid = cfg.setdefault("fidelity", {})
            fid["out_csv"] = str(out_csv)
            fid.setdefault("tests", {}).setdefault("rank_consistency", {})["enabled"] = False

            run_fidelity_tests(cfg, config_obj=config_obj)


if __name__ == "__main__":
    main()
