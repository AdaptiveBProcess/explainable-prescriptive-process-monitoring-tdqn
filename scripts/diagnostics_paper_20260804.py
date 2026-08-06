"""Diagnostics for paper corrections: Priority 2.4 (V(s) percentiles, representative
case selection) and Priority 1.3 option 3 (corrected |Delta Q| margin-drop metric).

Standalone: reuses repo loaders, does NOT modify pipeline artifacts.
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

DATASETS = {
    "simbank": {
        "npz": REPO / "data/processed/D_offline.npz",
        "splits": REPO / "data/processed/splits.json",
        "xai": REPO / "artifacts/xai",
        "ope": REPO / "artifacts/ope/ope_dr.json",
    },
    "bpi2017": {
        "npz": REPO / "data/bpi2017/processed/D_offline.npz",
        "splits": REPO / "data/bpi2017/processed/splits.json",
        "xai": REPO / "artifacts/xai/bpi2017",
        "ope": REPO / "artifacts/ope/bpi2017/ope_dr.json",
    },
    "bpi2020-rfp": {
        "npz": REPO / "data/bpi2020-rfp/processed/D_offline.npz",
        "splits": REPO / "data/bpi2020-rfp/processed/splits.json",
        "xai": REPO / "artifacts/xai/bpi2020-rfp",
        "ope": REPO / "artifacts/ope/bpi2020-rfp/ope_dr.json",
    },
    "bpi2020-int-decl": {
        "npz": REPO / "data/bpi2020-int-decl/processed/D_offline.npz",
        "splits": REPO / "data/bpi2020-int-decl/processed/splits.json",
        "xai": REPO / "artifacts/xai/bpi2020-int-decl",
        "ope": REPO / "artifacts/ope/bpi2020-int-decl/ope_dr.json",
    },
    "bpi2020-travel": {
        "npz": REPO / "data/bpi2020-travel/processed/D_offline.npz",
        "splits": REPO / "data/bpi2020-travel/processed/splits.json",
        "xai": REPO / "artifacts/xai/bpi2020-travel",
        "ope": REPO / "artifacts/ope/bpi2020-travel/ope_dr.json",
    },
}

CONFIG = {"training": {"transformer": {}}}  # defaults match params.yaml (d_model=128 etc.)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH = 1024
P_LIST = [0.1, 0.2, 0.3, 0.5]
N_RANDOM = 10
RNG_SEED = 123


def batched_q(q_net, s, sm, va):
    """Return (q_vals, v_s, a_star) over batches."""
    qs, vs, ast = [], [], []
    for i in range(0, len(s), BATCH):
        q, v, a = _compute_q_values(
            q_net, s[i : i + BATCH], sm[i : i + BATCH], va[i : i + BATCH], DEVICE
        )
        qs.append(q)
        vs.append(v)
        ast.append(a)
    return np.concatenate(qs), np.concatenate(vs), np.concatenate(ast)


def main(ds_names):
    out = {}
    for name in ds_names:
        p = DATASETS[name]
        if not p["npz"].exists():
            print(f"== {name}: SKIP (no data at {p['npz']})")
            continue
        ope = json.load(open(p["ope"]))
        ckpt = REPO / ope["metadata"]["ckpt_path"]
        vocab = REPO / ope["metadata"]["vocab_path"]
        print(f"== {name}: ckpt={ckpt.name in ('Q_theta.ckpt',) and ckpt.parent.name or ckpt}")
        q_net = _load_q_network(ckpt, p["npz"], vocab, CONFIG, DEVICE)
        test = load_dataset_with_splits(str(p["npz"]), str(p["splits"]), "test")
        s, sm, va = test["s"], test["s_mask"], test["valid_actions"]
        case_ptr = test.get("case_ptr")
        keys = list(test.keys())
        print(f"   test keys: {keys}; n={len(s)}")

        _, v_all, _ = batched_q(q_net, s, sm, va)

        # ---------- Part A (2.4): case-level V distribution (last transition per case)
        res = {"n_test_transitions": int(len(s))}
        if case_ptr is not None:
            # last transition per case = last occurrence of each case id
            last_idx = {}
            for i, c in enumerate(case_ptr):
                last_idx[int(c)] = i
            li = np.array(sorted(last_idx.values()))
            v_case = v_all[li]
            res["n_test_cases"] = int(len(li))
        else:
            v_case = v_all
        pcts = [1, 5, 10, 25, 50, 75, 90, 99]
        res["v_case_percentiles"] = {str(q): float(np.percentile(v_case, q)) for q in pcts}
        res["v_case_mean"] = float(v_case.mean())
        res["v_case_std"] = float(v_case.std())

        # explained cases: percentile of each; find case near 5th pct with an intervention
        risk = json.load(open(p["xai"] / "risk_explanations.json"))
        dq = json.load(open(p["xai"] / "deltaQ_explanations.json"))
        dq_by_case = {it["case_id"]: it for it in dq["items"]}
        explained = []
        for it in risk["items"]:
            vv = it.get("V") or it.get("q_star")
            pctile = float((v_case < vv).mean() * 100)
            d = dq_by_case.get(it["case_id"], {})
            explained.append(
                {
                    "case_id": it["case_id"],
                    "V": vv,
                    "pctile": pctile,
                    "delta_q": d.get("delta_q"),
                    "a_star": d.get("a_star_name"),
                    "top_tokens": [t["token_name"] for t in it.get("top_tokens", [])[:3]],
                    "top_drivers": [t["token_name"] for t in d.get("top_drivers", [])[:3]],
                }
            )
        # representative candidates: lowest-V explained cases with a real intervention margin
        with_int = [e for e in explained if e["delta_q"] and abs(e["delta_q"]) > 0]
        pool = with_int if with_int else explained
        pool_sorted = sorted(pool, key=lambda e: e["V"])
        res["lowest_V_explained"] = pool_sorted[:3]
        near5 = min(pool, key=lambda e: abs(e["pctile"] - 5.0))
        res["nearest_5th_pct_explained"] = near5
        c552 = next((e for e in explained if e["case_id"] == 552), None)
        if c552:
            res["case_552"] = c552

        # ---------- Part B (1.3 opt 3): corrected |Delta Q| margin drop, guided vs random vs anti
        # transition_idx aligns with TEST split arrays
        items = [
            it
            for it in dq["items"]
            if it.get("a_star") != it.get("a_contrast") and abs(it.get("delta_q", 0)) > 0
        ]
        res["n_dq_items"] = len(items)
        if items:
            idx = np.array([it["transition_idx"] for it in items])
            s_i, sm_i, va_i = s[idx], sm[idx], va[idx]
            a_star = np.array([it["a_star"] for it in items])
            a_con = np.array([it["a_contrast"] for it in items])
            q0, _, _ = batched_q(q_net, s_i, sm_i, va_i)
            dq_orig = q0[np.arange(len(items)), a_star] - q0[np.arange(len(items)), a_con]

            def margin_after(masked_states, masked_masks):
                qm, _, _ = batched_q(q_net, masked_states, masked_masks, va_i)
                return qm[np.arange(len(items)), a_star] - qm[np.arange(len(items)), a_con]

            def positions(it, key):
                mx = s.shape[1]
                return [
                    t["position"]
                    for t in it.get(key, [])
                    if t.get("position") is not None and 0 <= t["position"] < mx
                ]

            part_b = {}
            rng = np.random.default_rng(RNG_SEED)
            for p_rm in P_LIST:
                k_list, top_pos, bot_pos, nonpad = [], [], [], []
                for j, it in enumerate(items):
                    n_np = int(sm_i[j].sum())
                    k = max(1, int(np.ceil(p_rm * n_np)))
                    k_list.append(k)
                    top_pos.append(positions(it, "top_drivers")[:k])
                    bot_pos.append(positions(it, "bottom_drivers")[:k])
                    nonpad.append(np.nonzero(sm_i[j])[0])
                sg, mg = _perturb_states_mask_topk(s_i, sm_i, top_pos)
                dq_g = margin_after(sg, mg)
                sa, ma = _perturb_states_mask_topk(s_i, sm_i, bot_pos)
                dq_a = margin_after(sa, ma)
                red_r, flip_r = [], []
                for _ in range(N_RANDOM):
                    rnd = [
                        list(
                            rng.choice(
                                nonpad[j], size=min(k_list[j], len(nonpad[j])), replace=False
                            )
                        )
                        for j in range(len(items))
                    ]
                    sr, mr = _perturb_states_mask_topk(s_i, sm_i, rnd)
                    dq_r = margin_after(sr, mr)
                    red_r.append(np.abs(dq_orig) - np.abs(dq_r))
                    flip_r.append((np.sign(dq_r) != np.sign(dq_orig)).mean())
                red_r = np.stack(red_r).mean(0)
                part_b[p_rm] = {
                    "red_guided": float((np.abs(dq_orig) - np.abs(dq_g)).mean()),
                    "red_random": float(red_r.mean()),
                    "red_anti": float((np.abs(dq_orig) - np.abs(dq_a)).mean()),
                    "signflip_guided": float((np.sign(dq_g) != np.sign(dq_orig)).mean()),
                    "signflip_random": float(np.mean(flip_r)),
                    "mean_abs_dq_orig": float(np.abs(dq_orig).mean()),
                }
            res["dq_margin_drop"] = part_b
        out[name] = res
        print(json.dumps(res, indent=1, default=str))

    outpath = Path(__file__).parent / "diag_results.json"
    json.dump(out, open(outpath, "w"), indent=1)
    print("saved ->", outpath)


if __name__ == "__main__":
    names = sys.argv[1:] or list(DATASETS)
    main(names)
