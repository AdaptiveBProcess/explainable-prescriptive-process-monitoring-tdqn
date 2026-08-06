"""Final magnitude-based Q-drop gap (risk attribution, IG) on all five datasets,
with per-item paired standard errors. Source of the paper's phi^V numbers."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from xppm.rl.train_tdqn import load_dataset_with_splits  # noqa: E402
from xppm.xai.fidelity_tests import (  # noqa: E402
    _compute_q_values,
    _load_q_network,
    _perturb_states_mask_topk,
)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 123
NR = 20
DS = {
    "simbank": ("data/simbank/processed", "artifacts/ope/ope_dr.json", "artifacts/xai"),
    "bpi2017": (
        "data/bpi2017/processed",
        "artifacts/ope/bpi2017/ope_dr.json",
        "artifacts/xai/bpi2017",
    ),
    "bpi2020-rfp": (
        "data/bpi2020-rfp/processed",
        "artifacts/ope/bpi2020-rfp/ope_dr.json",
        "artifacts/xai/bpi2020-rfp",
    ),
    "bpi2020-int-decl": (
        "data/bpi2020-int-decl/processed",
        "artifacts/ope/bpi2020-int-decl/ope_dr.json",
        "artifacts/xai/bpi2020-int-decl",
    ),
    "bpi2020-travel": (
        "data/bpi2020-travel/processed",
        "artifacts/ope/bpi2020-travel/ope_dr.json",
        "artifacts/xai/bpi2020-travel",
    ),
}
out = {}
for ds, (proc, opep, xai) in DS.items():
    ope = json.load(open(REPO / opep))
    qn = _load_q_network(
        REPO / ope["metadata"]["ckpt_path"],
        REPO / proc / "D_offline.npz",
        REPO / ope["metadata"]["vocab_path"],
        {"training": {"transformer": {}}},
        DEV,
    )
    test = load_dataset_with_splits(
        str(REPO / proc / "D_offline.npz"), str(REPO / proc / "splits.json"), "test"
    )
    items = json.load(open(REPO / xai / "risk_explanations.json"))["items"]
    idx = np.array([it["transition_idx"] for it in items])
    s, sm, va = test["s"][idx], test["s_mask"][idx], test["valid_actions"][idx]
    _, v0, _ = _compute_q_values(qn, s, sm, va, DEV)
    out[ds] = {}
    for p in (0.1, 0.2, 0.3, 0.5):
        tp = []
        bp = []
        ks = []
        for i, it in enumerate(items):
            k = max(1, int(np.ceil(p * int(sm[i].sum()))))
            ks.append(k)
            pos = [
                t["position"]
                for t in it["top_tokens"]
                if 0 <= t["position"] < s.shape[1] and sm[i][t["position"]] > 0
            ]
            posb = [
                t["position"]
                for t in it.get("bottom_tokens", [])
                if 0 <= t["position"] < s.shape[1] and sm[i][t["position"]] > 0
            ]
            tp.append(pos[:k])
            bp.append(posb[:k])
        sg, mg = _perturb_states_mask_topk(s, sm, tp)
        _, vg, _ = _compute_q_values(qn, sg, mg, va, DEV)
        sa, ma = _perturb_states_mask_topk(s, sm, bp)
        _, vb, _ = _compute_q_values(qn, sa, ma, va, DEV)
        absr = []
        for r in range(NR):
            rnd = []
            for i in range(len(items)):
                np.random.seed(SEED + i * 1000 + r)
                nonpad = np.where(sm[i] > 0)[0]
                rnd.append(
                    list(np.random.choice(nonpad, size=min(ks[i], len(nonpad)), replace=False))
                )
            sr, mr = _perturb_states_mask_topk(s, sm, rnd)
            _, vr, _ = _compute_q_values(qn, sr, mr, va, DEV)
            absr.append(np.abs(v0 - vr))
        ar = np.stack(absr).mean(0)
        diff = np.abs(v0 - vg) - ar
        out[ds][str(p)] = {
            "abs_guided": float(np.abs(v0 - vg).mean()),
            "abs_random": float(ar.mean()),
            "abs_anti": float(np.abs(v0 - vb).mean()),
            "gap": float(diff.mean()),
            "gap_se": float(diff.std(ddof=1) / np.sqrt(len(diff))),
            "n": len(diff),
        }
    print(ds, {p: (round(v["gap"], 4), round(v["gap_se"], 4)) for p, v in out[ds].items()})
json.dump(out, open(REPO / "artifacts/fidelity/baselines/absgap_final_ig.json", "w"), indent=1)
