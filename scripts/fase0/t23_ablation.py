"""T2.3: brazo faltante de la ablacion — phi^V bajo el TEST DE MARGEN.
Identico a 23_margin_drop_compare pero rankeando por top_tokens/bottom_tokens de riesgo.
Reporta ambas formas: |E[.]| (Def.5) y E[|.|] (sin cancelacion, T2.4)."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(
    "/home/andrew/Documents/docs/3-resolver-problema-subtema/algorithms-explainability/xppm-tdqn"
)
sys.path.insert(0, str(REPO / "src"))
from xppm.rl.train_tdqn import load_dataset_with_splits
from xppm.xai.fidelity_tests import _compute_q_values, _load_q_network, _perturb_states_mask_topk

DS = {
    "simbank": ("data/processed", "artifacts/ope/ope_dr.json", "artifacts/xai"),
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
    "simbank-ir3": (
        "data/simbank-ir3/processed",
        "artifacts/ope/simbank-ir3/ope_dr.json",
        "artifacts/xai/simbank-ir3",
    ),
}
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
P_LIST = [0.1, 0.2]
N_RANDOM = 10
SEED = 123


def bq(q, s, sm, va, bs=512):
    out = []
    for i in range(0, len(s), bs):
        qq, _, _ = _compute_q_values(q, s[i : i + bs], sm[i : i + bs], va[i : i + bs], DEV)
        out.append(qq)
    return np.concatenate(out)


res = {}
for name, (proc, opep, xai) in DS.items():
    ope = json.load(open(REPO / opep))
    q = _load_q_network(
        REPO / ope["metadata"]["ckpt_path"],
        REPO / proc / "D_offline.npz",
        REPO / ope["metadata"]["vocab_path"],
        {"training": {"transformer": {}}},
        DEV,
    )
    t = load_dataset_with_splits(
        str(REPO / proc / "D_offline.npz"), str(REPO / proc / "splits.json"), "test"
    )
    s, sm, va = t["s"], t["s_mask"], t["valid_actions"]
    dq = json.load(open(REPO / xai / "deltaQ_explanations.json"))
    risk = {
        i["case_id"]: i for i in json.load(open(REPO / xai / "risk_explanations.json"))["items"]
    }
    items = [
        i
        for i in dq["items"]
        if i.get("a_star") != i.get("a_contrast") and abs(i.get("delta_q", 0)) > 0
    ]
    idx = np.array([i["transition_idx"] for i in items])
    s_i, sm_i, va_i = s[idx], sm[idx], va[idx]
    a_s = np.array([i["a_star"] for i in items])
    a_c = np.array([i["a_contrast"] for i in items])
    rows = np.arange(len(items))
    q0 = bq(q, s_i, sm_i, va_i)
    dq0 = q0[rows, a_s] - q0[rows, a_c]

    def margin(ms, mm):
        qm = bq(q, ms, mm, va_i)
        return qm[rows, a_s] - qm[rows, a_c]

    def pos(it, key, mx):
        return [
            x["position"]
            for x in it.get(key, [])
            if x.get("position") is not None and 0 <= x["position"] < mx
        ]

    out = {"n_items": len(items), "mean_abs_dq": float(np.abs(dq0).mean())}
    rng = np.random.default_rng(SEED)
    for p in P_LIST:
        k_l, top, bot, npad = [], [], [], []
        for j, it in enumerate(items):
            n = int(sm_i[j].sum())
            k = max(1, int(np.ceil(p * n)))
            k_l.append(k)
            r = risk.get(it["case_id"], {})  # <-- ranking por phi^V
            top.append(pos(r, "top_tokens", s.shape[1])[:k])
            bot.append(pos(r, "bottom_tokens", s.shape[1])[:k])
            npad.append(np.nonzero(sm_i[j])[0])
        sg, mg = _perturb_states_mask_topk(s_i, sm_i, top)
        dg = margin(sg, mg)
        sa, ma = _perturb_states_mask_topk(s_i, sm_i, bot)
        da = margin(sa, ma)
        rr = []
        for _ in range(N_RANDOM):
            rnd = [
                list(rng.choice(npad[j], size=min(k_l[j], len(npad[j])), replace=False))
                for j in range(len(items))
            ]
            sr, mr = _perturb_states_mask_topk(s_i, sm_i, rnd)
            rr.append(np.abs(dq0) - np.abs(margin(sr, mr)))
        rr = np.stack(rr).mean(0)
        out[str(p)] = {
            "red_guided_phiV": float((np.abs(dq0) - np.abs(dg)).mean()),
            "red_random": float(rr.mean()),
            "red_anti_phiV": float((np.abs(dq0) - np.abs(da)).mean()),
            "abs_red_guided_phiV": float(np.abs(np.abs(dq0) - np.abs(dg)).mean()),
            "abs_red_random": float(np.abs(rr).mean()),
            "abs_red_anti_phiV": float(np.abs(np.abs(dq0) - np.abs(da)).mean()),
        }
    res[name] = out
    print(name, json.dumps(out), flush=True)
json.dump(
    res,
    open(
        "artifacts/reports/fase0/t23_ablation.json",
        "w",
    ),
    indent=1,
)
print("saved")
