"""T1.3: caso representativo = mediana de |dQ| entre casos de intervencion explicados."""

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
from xppm.xai.fidelity_tests import _compute_q_values, _load_q_network

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
DEV = torch.device("cpu")


def batched_v(q_net, s, sm, va, bs=512):
    vs = []
    for i in range(0, len(s), bs):
        _, v, _ = _compute_q_values(q_net, s[i : i + bs], sm[i : i + bs], va[i : i + bs], DEV)
        vs.append(v)
    return np.concatenate(vs)


out = {}
for name, (proc, opep, xai) in DS.items():
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
    s, sm, va = test["s"], test["s_mask"], test["valid_actions"]
    v_all = batched_v(q_net, s, sm, va)
    cp = test.get("case_ptr")
    if cp is not None:
        last = {}
        for i, c in enumerate(cp):
            last[int(c)] = i
        v_case = v_all[np.array(sorted(last.values()))]
    else:
        v_case = v_all
    risk = json.load(open(REPO / xai / "risk_explanations.json"))
    dq = json.load(open(REPO / xai / "deltaQ_explanations.json"))
    dqby = {i["case_id"]: i for i in dq["items"]}
    riskby = {i["case_id"]: i for i in risk["items"]}
    # subpoblacion T1.5: par de acciones distinto y margen no nulo
    items = [
        i
        for i in dq["items"]
        if i.get("a_star") != i.get("a_contrast") and abs(i.get("delta_q", 0)) > 0
    ]
    if not items:
        out[name] = {"n_items": 0}
        continue
    mags = np.array([abs(i["delta_q"]) for i in items])
    med = float(np.median(mags))
    pick = items[int(np.argmin(np.abs(mags - med)))]
    r = riskby.get(pick["case_id"], {})
    vv = r.get("V") or r.get("q_star")
    out[name] = {
        "n_items": len(items),
        "median_abs_dq": med,
        "case_id": pick["case_id"],
        "V": vv,
        "pctile": float((v_case < vv).mean() * 100) if vv is not None else None,
        "median_V_case": float(np.median(v_case)),
        "delta_q": pick["delta_q"],
        "a_star": pick.get("a_star_name"),
        "why_at_risk": [t["token_name"] for t in r.get("top_tokens", [])[:3]],
        "why_act_now": [t["token_name"] for t in pick.get("top_drivers", [])[:3]],
    }
    print(name, json.dumps(out[name], default=str)[:400], flush=True)
json.dump(
    out,
    open(
        "artifacts/reports/fase0/t13_repr.json",
        "w",
    ),
    indent=1,
)
print("saved")
