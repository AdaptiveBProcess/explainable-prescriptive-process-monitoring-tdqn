"""Verdict stability across training seeds (v5, 1.6).

Given a freshly trained checkpoint (different repro.seed, same data and
splits), recompute the paper's two fidelity verdicts on the SAME explained
population (the transition indices of the baseline risk_explanations.json):

  - value test at p in {0.1, 0.2}: guided vs shared random null vs anti,
    paired gap and SE, ordering verdict;
  - corrected margin test at p = 0.1: sign-aware ranking, passes iff
    d_guided > d_random > d_anti and d_guided > 0.

Usage: 47_eval_seed.py --dataset simbank --ckpt-dir artifacts/models/tdqn/<run> \
                       --seed-tag 43 [--out artifacts/reports/seed_stability/...]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from xppm.rl.train_tdqn import load_dataset_with_splits  # noqa: E402
from xppm.utils.config import Config  # noqa: E402
from xppm.xai.attributions import compute_attributions  # noqa: E402
from xppm.xai.evaluability import DEFAULT_N_RANDOM, DEFAULT_SEED, _random_positions  # noqa: E402
from xppm.xai.fidelity_tests import (  # noqa: E402
    _compute_q_values,
    _load_q_network,
    _perturb_states_mask_topk,
)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG = {"training": {"transformer": {}}}

PROC = {"simbank": "data/simbank/processed", "bpi2017ct": "data/bpi2017ct/processed"}
XAI = {"simbank": "artifacts/xai", "bpi2017ct": "artifacts/xai/bpi2017ct"}


def value_test(q_net, s, sm, va, token_imp, p):
    _, v0, _ = _compute_q_values(q_net, s, sm, va, DEV)
    ks, top, bot = [], [], []
    for i in range(len(s)):
        k = max(1, int(np.ceil(p * int(sm[i].sum()))))
        ks.append(k)
        nonpad = np.nonzero(sm[i])[0]
        order = nonpad[np.argsort(-token_imp[i][nonpad])]
        top.append(list(order[:k]))
        bot.append(list(order[::-1][:k]))
    sg, mg = _perturb_states_mask_topk(s, sm, top)
    _, vg, _ = _compute_q_values(q_net, sg, mg, va, DEV)
    sa, ma = _perturb_states_mask_topk(s, sm, bot)
    _, vb, _ = _compute_q_values(q_net, sa, ma, va, DEV)
    absr = []
    for rep in range(DEFAULT_N_RANDOM):
        rnd = _random_positions(sm, np.array(ks), DEFAULT_SEED, rep)
        sr, mr = _perturb_states_mask_topk(s, sm, rnd)
        _, vr, _ = _compute_q_values(q_net, sr, mr, va, DEV)
        absr.append(np.abs(v0 - vr))
    dg = np.abs(v0 - vg)
    dr = np.stack(absr).mean(0)
    da = np.abs(v0 - vb)
    diff = dg - dr
    return {
        "gap": float(diff.mean()),
        "gap_se": float(diff.std(ddof=1) / np.sqrt(len(diff))),
        "se_ratio": float(diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))),
        "ordering_holds": bool(dg.mean() > dr.mean() > da.mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(PROC))
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--seed-tag", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ds = args.dataset
    ckpt_dir = Path(args.ckpt_dir)
    cfg = Config.for_dataset("configs/config.yaml", ds).raw
    n_steps = int(cfg["xai"]["methods"]["risk"].get("n_steps_ig", 128))
    xai_cfg = {"methods": {"risk": {"baseline": "pad", "n_steps_ig": n_steps}}}

    q_net = _load_q_network(
        ckpt_dir / "Q_theta.ckpt",
        REPO / PROC[ds] / "D_offline.npz",
        ckpt_dir / "vocab_activity.json",
        CONFIG,
        DEV,
    )
    test = load_dataset_with_splits(
        str(REPO / PROC[ds] / "D_offline.npz"), str(REPO / PROC[ds] / "splits.json"), "test"
    )
    base = json.load(open(REPO / XAI[ds] / "risk_explanations.json"))
    idx = np.array([it["transition_idx"] for it in base["items"]])
    s, sm, va = test["s"][idx], test["s_mask"][idx], test["valid_actions"][idx]

    # risk attribution from the NEW network on the SAME states
    risk = compute_attributions(
        q_net=q_net,
        states=s,
        state_masks=sm,
        valid_actions=va,
        config=xai_cfg,
        device=DEV,
        target="V",
        batch_size=64,
    )
    out = {
        "dataset": ds,
        "seed_tag": args.seed_tag,
        "ckpt": str(ckpt_dir),
        "n": int(len(idx)),
        "value_test": {
            str(p): value_test(q_net, s, sm, va, np.asarray(risk["token_importance"]), p)
            for p in (0.1, 0.2)
        },
    }

    # corrected margin test (sign-aware), NOOP contrast as in the paper (w=2)
    dq = compute_attributions(
        q_net=q_net,
        states=s,
        state_masks=sm,
        valid_actions=va,
        config=xai_cfg,
        device=DEV,
        target="deltaQ",
        contrast_action_id=0,
        batch_size=64,
    )
    a_star = np.asarray(dq["a_star"])
    keep = (a_star != 0) & (np.asarray(dq["delta_q"]) > 0)
    if keep.sum() >= 5:
        s2, sm2, va2 = s[keep], sm[keep], va[keep]
        signed = np.asarray(dq["attributions_emb"])[keep].sum(axis=2)
        rows = np.arange(int(keep.sum()))
        pair = np.stack([a_star[keep].astype(int), np.zeros(int(keep.sum()), dtype=int)], axis=1)
        q0, _, _ = _compute_q_values(q_net, s2, sm2, va2, DEV)
        dq0 = q0[rows, pair[:, 0]] - q0[rows, pair[:, 1]]
        ks, top, bot = [], [], []
        for j in range(len(rows)):
            k = max(1, int(np.ceil(0.1 * int(sm2[j].sum()))))
            ks.append(k)
            nonpad = np.nonzero(sm2[j])[0]
            order = nonpad[np.argsort(-signed[j][nonpad])]
            top.append(list(order[:k]))
            bot.append(list(order[::-1][:k]))
        sg, mg = _perturb_states_mask_topk(s2, sm2, top)
        qg, _, _ = _compute_q_values(q_net, sg, mg, va2, DEV)
        dq_g = qg[rows, pair[:, 0]] - qg[rows, pair[:, 1]]
        sa, ma = _perturb_states_mask_topk(s2, sm2, bot)
        qa, _, _ = _compute_q_values(q_net, sa, ma, va2, DEV)
        dq_a = qa[rows, pair[:, 0]] - qa[rows, pair[:, 1]]
        red_r = []
        for rep in range(DEFAULT_N_RANDOM):
            rnd = _random_positions(sm2, np.array(ks), DEFAULT_SEED, rep)
            sr, mr = _perturb_states_mask_topk(s2, sm2, rnd)
            qr, _, _ = _compute_q_values(q_net, sr, mr, va2, DEV)
            dq_r = qr[rows, pair[:, 0]] - qr[rows, pair[:, 1]]
            red_r.append(np.abs(dq0) - np.abs(dq_r))
        d_g = float((np.abs(dq0) - np.abs(dq_g)).mean())
        d_r = float(np.stack(red_r).mean(0).mean())
        d_a = float((np.abs(dq0) - np.abs(dq_a)).mean())
        out["margin_test_corrected"] = {
            "n_dq": int(keep.sum()),
            "d_guided": d_g,
            "d_random": d_r,
            "d_anti": d_a,
            "passes": bool(d_g > d_r > d_a and d_g > 0),
        }
    else:
        out["margin_test_corrected"] = {"n_dq": int(keep.sum()), "note": "too few margin cases"}

    out_path = Path(
        args.out or REPO / f"artifacts/reports/seed_stability/{ds}_seed{args.seed_tag}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
