"""T2.1/T2.2 (C5): both fidelity tests for the Fase-2 datasets.

Usage: 26_fidelity_bpi2017ct.py [bpi2017ct|simbank-ir3]   (default: bpi2017ct)

Risk test: magnitude-based Q-drop gap |ΔV| with per-item paired SEs — same
apparatus, seeds and conventions as 25_absgap_final.py (SEED=123, NR=20).
Intervention test: margin drop + sign flips — same apparatus as
23_margin_drop_compare.py (shared null: evaluability.DEFAULT_N_RANDOM,
DEFAULT_SEED), plus paired SEs.
For simbank-ir3 the deltaQ contrast recorded per item is the runner-up rate
(a_star != a_contrast filter keeps only offer decision points).

Output: artifacts/fidelity/<dataset>/fidelity_ct.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from xppm.rl.train_tdqn import load_dataset_with_splits  # noqa: E402
from xppm.xai.evaluability import (  # noqa: E402
    DEFAULT_N_RANDOM,
    DEFAULT_SEED,
    _random_positions,
)
from xppm.xai.fidelity_tests import (  # noqa: E402
    _compute_q_values,
    _load_q_network,
    _perturb_states_mask_topk,
)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = DEFAULT_SEED
NR = DEFAULT_N_RANDOM
N_RANDOM_MARGIN = DEFAULT_N_RANDOM
P_LIST = (0.1, 0.2, 0.3, 0.5)
BATCH = 1024
CONFIG = {"training": {"transformer": {}}}

DATASET = sys.argv[1] if len(sys.argv) > 1 else "bpi2017ct"
PROC = REPO / f"data/{DATASET}/processed"
OPE = REPO / f"artifacts/ope/{DATASET}/ope_dr_boa.json"
XAI = REPO / f"artifacts/xai/{DATASET}"
OUT = REPO / f"artifacts/fidelity/{DATASET}/fidelity_ct.json"


def batched_q(q_net, s, sm, va):
    qs = []
    for i in range(0, len(s), BATCH):
        q, _, _ = _compute_q_values(
            q_net, s[i : i + BATCH], sm[i : i + BATCH], va[i : i + BATCH], DEV
        )
        qs.append(q)
    return np.concatenate(qs)


def batched_v(q_net, s, sm, va):
    vs = []
    for i in range(0, len(s), BATCH):
        _, v, _ = _compute_q_values(
            q_net, s[i : i + BATCH], sm[i : i + BATCH], va[i : i + BATCH], DEV
        )
        vs.append(v)
    return np.concatenate(vs)


def risk_absgap(q_net, test):
    items = json.load(open(XAI / "risk_explanations.json"))["items"]
    idx = np.array([it["transition_idx"] for it in items])
    s, sm, va = test["s"][idx], test["s_mask"][idx], test["valid_actions"][idx]
    v0 = batched_v(q_net, s, sm, va)
    out = {"n_items": len(items)}
    for p in P_LIST:
        tp, bp, ks = [], [], []
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
        vg = batched_v(q_net, sg, mg, va)
        sa, ma = _perturb_states_mask_topk(s, sm, bp)
        vb = batched_v(q_net, sa, ma, va)
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
            vr = batched_v(q_net, sr, mr, va)
            absr.append(np.abs(v0 - vr))
        ar = np.stack(absr).mean(0)
        diff = np.abs(v0 - vg) - ar
        out[str(p)] = {
            "abs_guided": float(np.abs(v0 - vg).mean()),
            "abs_random": float(ar.mean()),
            "abs_anti": float(np.abs(v0 - vb).mean()),
            "gap": float(diff.mean()),
            "gap_se": float(diff.std(ddof=1) / np.sqrt(len(diff))),
            "n": len(diff),
        }
    return out


def margin_drop(q_net, test):
    dq = json.load(open(XAI / "deltaQ_explanations.json"))
    items = [
        it
        for it in dq["items"]
        if it.get("a_star") != it.get("a_contrast") and abs(it.get("delta_q", 0)) > 0
    ]
    if not items:
        return {"n_items": 0}
    s_all = test["s"]
    idx = np.array([it["transition_idx"] for it in items])
    s, sm, va = test["s"][idx], test["s_mask"][idx], test["valid_actions"][idx]
    a_star = np.array([it["a_star"] for it in items])
    a_con = np.array([it["a_contrast"] for it in items])
    rows = np.arange(len(items))
    q0 = batched_q(q_net, s, sm, va)
    dq_orig = q0[rows, a_star] - q0[rows, a_con]

    def margin_after(ms, mm):
        qm = batched_q(q_net, ms, mm, va)
        return qm[rows, a_star] - qm[rows, a_con]

    def positions(it, key):
        mx = s_all.shape[1]
        return [
            t["position"]
            for t in it.get(key, [])
            if t.get("position") is not None and 0 <= t["position"] < mx
        ]

    out = {"n_items": len(items), "mean_abs_dq_orig": float(np.abs(dq_orig).mean())}
    for p_rm in P_LIST:
        k_list, top_pos, bot_pos, nonpad = [], [], [], []
        for j in range(len(items)):
            k = max(1, int(np.ceil(p_rm * int(sm[j].sum()))))
            k_list.append(k)
            top_pos.append(positions(items[j], "top_drivers")[:k])
            bot_pos.append(positions(items[j], "bottom_drivers")[:k])
            nonpad.append(np.nonzero(sm[j])[0])
        sg, mg = _perturb_states_mask_topk(s, sm, top_pos)
        dq_g = margin_after(sg, mg)
        sa, ma = _perturb_states_mask_topk(s, sm, bot_pos)
        dq_a = margin_after(sa, ma)
        red_r, flip_r = [], []
        for rep in range(N_RANDOM_MARGIN):
            rnd = _random_positions(sm, np.array(k_list), SEED, rep)
            sr, mr = _perturb_states_mask_topk(s, sm, rnd)
            dq_r = margin_after(sr, mr)
            red_r.append(np.abs(dq_orig) - np.abs(dq_r))
            flip_r.append((np.sign(dq_r) != np.sign(dq_orig)).mean())
        red_r = np.stack(red_r).mean(0)
        red_g = np.abs(dq_orig) - np.abs(dq_g)
        diff = red_g - red_r
        out[str(p_rm)] = {
            "red_guided": float(red_g.mean()),
            "red_random": float(red_r.mean()),
            "red_anti": float((np.abs(dq_orig) - np.abs(dq_a)).mean()),
            "gap": float(diff.mean()),
            "gap_se": float(diff.std(ddof=1) / np.sqrt(len(diff))),
            "signflip_guided": float((np.sign(dq_g) != np.sign(dq_orig)).mean()),
            "signflip_random": float(np.mean(flip_r)),
        }
    return out


def main():
    ope = json.load(open(OPE))
    ckpt = REPO / ope["metadata"]["ckpt_path"]
    vocab = REPO / ope["metadata"]["vocab_path"]
    q_net = _load_q_network(ckpt, PROC / "D_offline.npz", vocab, CONFIG, DEV)
    test = load_dataset_with_splits(str(PROC / "D_offline.npz"), str(PROC / "splits.json"), "test")
    results = {
        "risk_absgap": risk_absgap(q_net, test),
        "margin_drop": margin_drop(q_net, test),
        "ckpt": str(ckpt),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(OUT, "w"), indent=1)
    for name, block in results.items():
        if not isinstance(block, dict):
            continue
        print(f"== {name} (n={block.get('n_items')})")
        for p in P_LIST:
            b = block.get(str(p))
            if b:
                print(
                    f"  p={p}: "
                    + ", ".join(f"{k}={v:.4g}" for k, v in b.items() if isinstance(v, (int, float)))
                )


if __name__ == "__main__":
    main()
