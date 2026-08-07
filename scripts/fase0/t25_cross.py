"""T2.5: por que el desplazamiento de margen de SimBank cruza cero entre k=1 y k=2."""

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

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ope = json.load(open(REPO / "artifacts/ope/ope_dr.json"))
q = _load_q_network(
    REPO / ope["metadata"]["ckpt_path"],
    REPO / "data/processed/D_offline.npz",
    REPO / ope["metadata"]["vocab_path"],
    {"training": {"transformer": {}}},
    DEV,
)
t = load_dataset_with_splits(
    str(REPO / "data/processed/D_offline.npz"), str(REPO / "data/processed/splits.json"), "test"
)
s, sm, va = t["s"], t["s_mask"], t["valid_actions"]
dq = json.load(open(REPO / "artifacts/xai/deltaQ_explanations.json"))
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


def bq(ss, mm):
    o = []
    for i in range(0, len(ss), 512):
        qq, _, _ = _compute_q_values(q, ss[i : i + 512], mm[i : i + 512], va_i[i : i + 512], DEV)
        o.append(qq)
    return np.concatenate(o)


q0 = bq(s_i, sm_i)
dq0 = q0[rows, a_s] - q0[rows, a_c]


def masked(k):
    pos = [
        [x["position"] for x in it.get("top_drivers", []) if x.get("position") is not None][:k]
        for it in items
    ]
    ss, mm = _perturb_states_mask_topk(s_i, sm_i, pos)
    qm = bq(ss, mm)
    return qm[rows, a_s] - qm[rows, a_c]


d1, d2 = masked(1), masked(2)
r1 = np.abs(dq0) - np.abs(d1)
r2 = np.abs(dq0) - np.abs(d2)
print(f"n={len(items)}  mean|dQ|={np.abs(dq0).mean():.4g}")
for lbl, rr in [("k=1", r1), ("k=2", r2)]:
    print(
        f"{lbl}: media={rr.mean():+9.4g}  mediana={np.median(rr):+9.4g}  "
        f"contraen={(rr>0).sum():3d}  ensanchan={(rr<0).sum():3d}  max_ensanche={rr.min():+.4g}"
    )
# contribucion del 2do evento
delta = r2 - r1
print(f"\nefecto del 2do evento (r2-r1): media={delta.mean():+.4g} mediana={np.median(delta):+.4g}")
print(f"  casos donde el 2do evento ENSANCHA el margen: {(delta<0).sum()} de {len(delta)}")
worst = np.argsort(delta)[:5]
print("  5 casos que mas ensanchan al añadir el 2do evento:")
for j in worst:
    tk = [x["token_name"] for x in items[j].get("top_drivers", [])[:2]]
    print(
        f"    caso {items[j]['case_id']}: dQ={dq0[j]:.4g} r1={r1[j]:+.4g} r2={r2[j]:+.4g}  top2={tk}"
    )
