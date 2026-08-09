"""Magnitude-based Q-drop gap: g_abs(p) = E[|dV_guided|] - E[|dV_random|],
computed per item for IG / saliency / attention, plus anti-guided control.
Same selection, seeds and deletion apparatus as the pipeline fidelity tests.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(
    "/home/andrew/Documents/docs/3-resolver-problema-subtema/algorithms-explainability/xppm-tdqn"
)
sys.path.insert(0, str(REPO / "src"))
from xppm.rl.train_tdqn import load_dataset_with_splits  # noqa: E402
from xppm.xai.fidelity_tests import (  # noqa: E402
    _compute_q_values,
    _load_q_network,
    _perturb_states_mask_topk,
)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED, N_RANDOM, P_LIST = 123, 20, [0.1, 0.2]
DS = {
    "bpi2012": (
        "data/bpi2012/processed",
        "artifacts/ope/bpi2012/ope_dr.json",
        "artifacts/xai/bpi2012",
    ),
    "bpi2017ct": (
        "data/bpi2017ct/processed",
        "artifacts/ope/bpi2017ct/ope_dr.json",
        "artifacts/xai/bpi2017ct",
    ),
}


def importance_positions(items, key):
    return [[t["position"] for t in it.get(key, [])] for it in items]


def run(ds):
    proc, opep, xai_ig = DS[ds]
    ope = json.load(open(REPO / opep))
    q_net = _load_q_network(
        REPO / ope["metadata"]["ckpt_path"],
        REPO / proc / "D_offline.npz",
        REPO / ope["metadata"]["vocab_path"],
        {"training": {"transformer": {}}},
        DEV,
    )
    test = load_dataset_with_splits(
        str(REPO / proc / "D_offline.npz"), str(REPO / proc / "splits.json"), "test"
    )
    out = {}
    sources = {
        "ig": REPO / xai_ig,
        "saliency": Path("artifacts/xai/baselines/saliency") / ds,
        "attention": Path("artifacts/xai/baselines/attention") / ds,
    }
    for method, src in sources.items():
        risk = json.load(open(src / "risk_explanations.json"))
        dq = json.load(open(src / "deltaQ_explanations.json"))
        items = risk["items"]
        idx = np.array([it["transition_idx"] for it in items])
        s, sm, va = test["s"][idx], test["s_mask"][idx], test["valid_actions"][idx]
        _, v0, _ = _compute_q_values(q_net, s, sm, va, DEV)
        res = {}
        for attr_name, its, key_top, key_bot in (
            ("phiV", risk["items"], "top_tokens", "bottom_tokens"),
            ("phiDQ_on_V", dq["items"], "top_drivers", "bottom_drivers"),
        ):
            for p in P_LIST:
                k_list, top_pos, bot_pos = [], [], []
                for i, it in enumerate(its):
                    n_np = int(sm[i].sum())
                    k = max(1, int(np.ceil(p * n_np)))
                    k_list.append(k)
                    pos_t = [
                        t["position"]
                        for t in it.get(key_top, [])
                        if 0 <= t["position"] < s.shape[1] and sm[i][t["position"]] > 0
                    ]
                    pos_b = [
                        t["position"]
                        for t in it.get(key_bot, [])
                        if 0 <= t["position"] < s.shape[1] and sm[i][t["position"]] > 0
                    ]
                    top_pos.append(pos_t[:k])
                    bot_pos.append(pos_b[:k])
                sg, mg = _perturb_states_mask_topk(s, sm, top_pos)
                _, vg, _ = _compute_q_values(q_net, sg, mg, va, DEV)
                sa, ma = _perturb_states_mask_topk(s, sm, bot_pos)
                _, va2, _ = _compute_q_values(q_net, sa, ma, va, DEV)
                abs_r = []
                for r in range(N_RANDOM):
                    rnd = []
                    for i in range(len(its)):
                        np.random.seed(SEED + i * 1000 + r)
                        nonpad = np.where(sm[i] > 0)[0]
                        k = k_list[i]
                        rnd.append(
                            list(np.random.choice(nonpad, size=min(k, len(nonpad)), replace=False))
                        )
                    sr, mr = _perturb_states_mask_topk(s, sm, rnd)
                    _, vr, _ = _compute_q_values(q_net, sr, mr, va, DEV)
                    abs_r.append(np.abs(v0 - vr))
                abs_rand = np.stack(abs_r).mean(0)
                res[f"{attr_name}@{p}"] = {
                    "abs_guided": float(np.abs(v0 - vg).mean()),
                    "abs_random": float(abs_rand.mean()),
                    "abs_anti": float(np.abs(v0 - va2).mean()),
                    "gap_abs": float((np.abs(v0 - vg) - abs_rand).mean()),
                    "gap_abs_anti": float((np.abs(v0 - va2) - abs_rand).mean()),
                }
        out[method] = res
    return out


results = {ds: run(ds) for ds in DS}
json.dump(
    results,
    open(
        Path("%sabsgap_compare_fase0.json" % "artifacts/reports/fase0/"),
        "w",
    ),
    indent=1,
)
for ds, r in results.items():
    print(f"== {ds}")
    for m, res in r.items():
        for k, v in res.items():
            print(
                f"  {m:9s} {k:12s} guided={v['abs_guided']:9.3f} "
                f"random={v['abs_random']:8.3f} anti={v['abs_anti']:8.3f} "
                f"gap={v['gap_abs']:9.3f} gap_anti={v['gap_abs_anti']:8.3f}"
            )
