"""Margin-drop + sign-flip test (Def. margindrop of the paper) for IG and the
baseline attribution methods, same apparatus as diagnostics_paper_20260804.py
Part B, parameterized by XAI dir so IG and baselines are scored identically.
"""

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

DATASETS = {
    "simbank": {
        "npz": REPO / "data/simbank/processed/D_offline.npz",
        "splits": REPO / "data/simbank/processed/splits.json",
        "ope": REPO / "artifacts/ope/ope_dr_boa.json",
        "xai_ig": REPO / "artifacts/xai",
    },
    "bpi2017": {
        "npz": REPO / "data/bpi2017/processed/D_offline.npz",
        "splits": REPO / "data/bpi2017/processed/splits.json",
        "ope": REPO / "artifacts/ope/bpi2017/ope_dr_boa.json",
        "xai_ig": REPO / "artifacts/xai/bpi2017",
    },
    "bpi2020-rfp": {
        "npz": REPO / "data/bpi2020-rfp/processed/D_offline.npz",
        "splits": REPO / "data/bpi2020-rfp/processed/splits.json",
        "ope": REPO / "artifacts/ope/bpi2020-rfp/ope_dr_boa.json",
        "xai_ig": REPO / "artifacts/xai/bpi2020-rfp",
    },
    "bpi2012": {
        "npz": REPO / "data/bpi2012/processed/D_offline.npz",
        "splits": REPO / "data/bpi2012/processed/splits.json",
        "ope": REPO / "artifacts/ope/bpi2012/ope_dr_boa.json",
        "xai_ig": REPO / "artifacts/xai/bpi2012",
    },
    "bpi2017ct": {
        "npz": REPO / "data/bpi2017ct/processed/D_offline.npz",
        "splits": REPO / "data/bpi2017ct/processed/splits.json",
        "ope": REPO / "artifacts/ope/bpi2017ct/ope_dr_boa.json",
        "xai_ig": REPO / "artifacts/xai/bpi2017ct",
    },
    "simbank-ir3": {
        "npz": REPO / "data/simbank-ir3/processed/D_offline.npz",
        "splits": REPO / "data/simbank-ir3/processed/splits.json",
        "ope": REPO / "artifacts/ope/simbank-ir3/ope_dr_boa.json",
        "xai_ig": REPO / "artifacts/xai/simbank-ir3",
    },
    "bpi2012-offertes": {
        "npz": REPO / "data/bpi2012-offertes/processed/D_offline.npz",
        "splits": REPO / "data/bpi2012-offertes/processed/splits.json",
        "ope": REPO / "artifacts/ope/bpi2012-offertes/ope_dr_boa.json",
        "xai_ig": REPO / "artifacts/xai/bpi2012-offertes",
    },
    "sepsis": {
        "npz": REPO / "data/sepsis/processed/D_offline.npz",
        "splits": REPO / "data/sepsis/processed/splits.json",
        "ope": REPO / "artifacts/ope/sepsis/ope_dr_boa.json",
        "xai_ig": REPO / "artifacts/xai/sepsis",
    },
}
CONFIG = {"training": {"transformer": {}}}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH = 1024
P_LIST = [0.1, 0.2, 0.3, 0.5]
# Single random-null convention shared with 32_evaluability / 26_fidelity_*:
# same n_random, seed AND per-(item, rep) position draws, so the same null has
# one value per configuration across every artifact.
from xppm.xai.evaluability import (  # noqa: E402
    DEFAULT_N_RANDOM,
    DEFAULT_SEED,
    _random_positions,
)

N_RANDOM = DEFAULT_N_RANDOM
RNG_SEED = DEFAULT_SEED


def batched_q(q_net, s, sm, va):
    qs = []
    for i in range(0, len(s), BATCH):
        q, _, _ = _compute_q_values(
            q_net, s[i : i + BATCH], sm[i : i + BATCH], va[i : i + BATCH], DEVICE
        )
        qs.append(q)
    return np.concatenate(qs)


def margin_drop(q_net, test, dq_json_path, signed_scores=None):
    """signed_scores: optional (n_all_items, max_len) array of SIGNED per-token
    margin attributions (npz raw attr summed over embedding dims). When given,
    rankings come from it (descending = margin supporters first) instead of the
    JSON's magnitude-ranked top_drivers — the exploratory sign-aware variant of
    Def. 3 (v4 review, D1). Rows are aligned with dq["items"] order."""
    dq = json.load(open(dq_json_path))
    rows_items = [
        (row, it)
        for row, it in enumerate(dq["items"])
        if it.get("a_star") != it.get("a_contrast") and abs(it.get("delta_q", 0)) > 0
    ]
    items = [it for _, it in rows_items]
    npz_rows = [row for row, _ in rows_items]
    if not items:
        return {"n_items": 0}
    s, sm, va = test["s"], test["s_mask"], test["valid_actions"]
    idx = np.array([it["transition_idx"] for it in items])
    s_i, sm_i, va_i = s[idx], sm[idx], va[idx]
    a_star = np.array([it["a_star"] for it in items])
    a_con = np.array([it["a_contrast"] for it in items])
    rows = np.arange(len(items))
    q0 = batched_q(q_net, s_i, sm_i, va_i)
    dq_orig = q0[rows, a_star] - q0[rows, a_con]

    def margin_after(ms, mm):
        qm = batched_q(q_net, ms, mm, va_i)
        return qm[rows, a_star] - qm[rows, a_con]

    def positions(it, key):
        mx = s.shape[1]
        return [
            t["position"]
            for t in it.get(key, [])
            if t.get("position") is not None and 0 <= t["position"] < mx
        ]

    out = {"n_items": len(items), "mean_abs_dq_orig": float(np.abs(dq_orig).mean())}
    for p_rm in P_LIST:
        k_list, top_pos, bot_pos, nonpad = [], [], [], []
        for j, it in enumerate(items):
            k = max(1, int(np.ceil(p_rm * int(sm_i[j].sum()))))
            k_list.append(k)
            if signed_scores is not None:
                sc = signed_scores[npz_rows[j]]
                np_pos = np.nonzero(sm_i[j])[0]
                order = np_pos[np.argsort(-sc[np_pos])]
                top_pos.append(list(order[:k]))  # strongest margin supporters
                bot_pos.append(list(order[::-1][:k]))  # strongest compressors
            else:
                top_pos.append(positions(it, "top_drivers")[:k])
                bot_pos.append(positions(it, "bottom_drivers")[:k])
            nonpad.append(np.nonzero(sm_i[j])[0])
        sg, mg = _perturb_states_mask_topk(s_i, sm_i, top_pos)
        dq_g = margin_after(sg, mg)
        sa, ma = _perturb_states_mask_topk(s_i, sm_i, bot_pos)
        dq_a = margin_after(sa, ma)
        red_r, flip_r = [], []
        for rep in range(N_RANDOM):
            rnd = _random_positions(sm_i, np.array(k_list), RNG_SEED, rep)
            sr, mr = _perturb_states_mask_topk(s_i, sm_i, rnd)
            dq_r = margin_after(sr, mr)
            red_r.append(np.abs(dq_orig) - np.abs(dq_r))
            flip_r.append((np.sign(dq_r) != np.sign(dq_orig)).mean())
        red_r = np.stack(red_r).mean(0)
        out[str(p_rm)] = {
            "red_guided": float((np.abs(dq_orig) - np.abs(dq_g)).mean()),
            "red_random": float(red_r.mean()),
            "red_anti": float((np.abs(dq_orig) - np.abs(dq_a)).mean()),
            "signflip_guided": float((np.sign(dq_g) != np.sign(dq_orig)).mean()),
            "signflip_random": float(np.mean(flip_r)),
        }
    return out


def main():
    datasets = sys.argv[1:] or list(DATASETS)
    results = {}
    for name in datasets:
        p = DATASETS[name]
        ope = json.load(open(p["ope"]))
        ckpt = REPO / ope["metadata"]["ckpt_path"]
        vocab = REPO / ope["metadata"]["vocab_path"]
        q_net = _load_q_network(ckpt, p["npz"], vocab, CONFIG, DEVICE)
        test = load_dataset_with_splits(str(p["npz"]), str(p["splits"]), "test")
        results[name] = {}
        methods = {
            "ig": p["xai_ig"] / "deltaQ_explanations.json",
            "saliency": REPO
            / "artifacts/xai/baselines/saliency"
            / name
            / "deltaQ_explanations.json",
            "attention": REPO
            / "artifacts/xai/baselines/attention"
            / name
            / "deltaQ_explanations.json",
        }
        for method, path in methods.items():
            print(f"== {name} / {method}")
            if not Path(path).exists():
                print(f"   -- {name}/{method}: no artifact, skipped")
                continue
            results[name][method] = margin_drop(q_net, test, path)
        # exploratory sign-aware ranking of the IG margin attribution (unscored)
        npz_path = p["xai_ig"] / "ig_grad_attributions.npz"
        if npz_path.exists():
            print(f"== {name} / ig_signed")
            z = np.load(npz_path)
            signed = z["deltaq_attr"].sum(axis=-1)
            results[name]["ig_signed"] = margin_drop(
                q_net, test, methods["ig"], signed_scores=signed
            )
    outpath = REPO / "artifacts/fidelity/baselines/margin_drop_compare.json"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(outpath, "w"), indent=1)
    print("saved ->", outpath)


if __name__ == "__main__":
    main()
