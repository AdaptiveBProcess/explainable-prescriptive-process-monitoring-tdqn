"""Why wait? — attribution and fidelity test for the wait-side margin (v5, 1.3).

Def. 2.1 promises an account of "the alternative of not acting"; until now only
intervention states (a* != a0) carried a margin attribution. This script closes
the gap: on the explained-pool states where the policy WAITS (a* = a0), the
wait margin dQ_wait(s) = Q(s, a0) - max_{a != a0} Q(s, a) > 0 is attributed
with the existing runner-up contrast machinery (contrast_action_id = -1, which
for a* = a0 yields exactly Q(a0) - Q(best other)), and evaluated with the
corrected (sign-aware) margin test under the shared random null.

Output: artifacts/reports/wait_margin.json
"""

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
P_LIST = [0.1, 0.2]

DS = {
    "simbank": ("data/simbank/processed", "artifacts/ope/ope_dr_boa.json", "artifacts/xai"),
    "simbank-ir3": (
        "data/simbank-ir3/processed",
        "artifacts/ope/simbank-ir3/ope_dr_boa.json",
        "artifacts/xai/simbank-ir3",
    ),
    "bpi2012": (
        "data/bpi2012/processed",
        "artifacts/ope/bpi2012/ope_dr_boa.json",
        "artifacts/xai/bpi2012",
    ),
    "bpi2012-offertes": (
        "data/bpi2012-offertes/processed",
        "artifacts/ope/bpi2012-offertes/ope_dr_boa.json",
        "artifacts/xai/bpi2012-offertes",
    ),
    "bpi2017ct": (
        "data/bpi2017ct/processed",
        "artifacts/ope/bpi2017ct/ope_dr_boa.json",
        "artifacts/xai/bpi2017ct",
    ),
    "sepsis": (
        "data/sepsis/processed",
        "artifacts/ope/sepsis/ope_dr_boa.json",
        "artifacts/xai/sepsis",
    ),
}


def run(ds: str) -> dict:
    proc, opep, xaip = DS[ds]
    cfg = Config.for_dataset("configs/config.yaml", ds).raw
    n_steps = int(cfg["xai"]["methods"]["risk"].get("n_steps_ig", 128))
    xai_cfg = {"methods": {"risk": {"baseline": "pad", "n_steps_ig": n_steps}}}

    risk = json.load(open(REPO / xaip / "risk_explanations.json"))
    wait = [it for it in risk["items"] if it.get("a_star") == 0]
    if len(wait) == 0:
        return {"n_wait_states": 0, "note": "no wait-optimal states in the explained pool"}

    ope = json.load(open(REPO / opep))
    q_net = _load_q_network(
        REPO / ope["metadata"]["ckpt_path"],
        REPO / proc / "D_offline.npz",
        REPO / ope["metadata"]["vocab_path"],
        CONFIG,
        DEV,
    )
    test = load_dataset_with_splits(
        str(REPO / proc / "D_offline.npz"), str(REPO / proc / "splits.json"), "test"
    )
    idx = np.array([it["transition_idx"] for it in wait])
    s, sm, va = test["s"][idx], test["s_mask"][idx], test["valid_actions"][idx]

    # runner-up contrast: for wait states a*_fixed = a0, so the target is
    # Q(a0) - Q(best other) = the wait margin
    res = compute_attributions(
        q_net=q_net,
        states=s,
        state_masks=sm,
        valid_actions=va,
        config=xai_cfg,
        device=DEV,
        target="deltaQ",
        contrast_action_id=-1,
        batch_size=64,
    )
    a_star = np.asarray(res["a_star"])
    a_con = np.asarray(res["a_contrast"]) if "a_contrast" in res else None
    keep = (a_star == 0) & (np.asarray(res["delta_q"]) > 0)
    if a_con is not None:
        keep &= a_con != a_star
    if keep.sum() == 0:
        return {"n_wait_states": int(len(wait)), "n_evaluable": 0}

    s, sm, va = s[keep], sm[keep], va[keep]
    signed = np.asarray(res["attributions_emb"])[keep].sum(axis=2)
    a_pair = np.stack(
        [np.zeros(keep.sum(), dtype=int), np.asarray(res["a_contrast"])[keep].astype(int)], axis=1
    )
    rows = np.arange(int(keep.sum()))
    q0, _, _ = _compute_q_values(q_net, s, sm, va, DEV)
    dq0 = q0[rows, a_pair[:, 0]] - q0[rows, a_pair[:, 1]]

    out = {
        "n_wait_states": int(len(wait)),
        "n_evaluable": int(keep.sum()),
        "n_steps_ig": n_steps,
        "mean_wait_margin": float(dq0.mean()),
    }
    for p in P_LIST:
        ks, top, bot = [], [], []
        for j in range(len(rows)):
            k = max(1, int(np.ceil(p * int(sm[j].sum()))))
            ks.append(k)
            nonpad = np.nonzero(sm[j])[0]
            order = nonpad[np.argsort(-signed[j][nonpad])]
            top.append(list(order[:k]))
            bot.append(list(order[::-1][:k]))
        sg, mg = _perturb_states_mask_topk(s, sm, top)
        qg, _, _ = _compute_q_values(q_net, sg, mg, va, DEV)
        dq_g = qg[rows, a_pair[:, 0]] - qg[rows, a_pair[:, 1]]
        sa, ma = _perturb_states_mask_topk(s, sm, bot)
        qa, _, _ = _compute_q_values(q_net, sa, ma, va, DEV)
        dq_a = qa[rows, a_pair[:, 0]] - qa[rows, a_pair[:, 1]]
        red_r = []
        for rep in range(DEFAULT_N_RANDOM):
            rnd = _random_positions(sm, np.array(ks), DEFAULT_SEED, rep)
            sr, mr = _perturb_states_mask_topk(s, sm, rnd)
            qr, _, _ = _compute_q_values(q_net, sr, mr, va, DEV)
            dq_r = qr[rows, a_pair[:, 0]] - qr[rows, a_pair[:, 1]]
            red_r.append(np.abs(dq0) - np.abs(dq_r))
        d_r = float(np.stack(red_r).mean(0).mean())
        d_g = float((np.abs(dq0) - np.abs(dq_g)).mean())
        d_a = float((np.abs(dq0) - np.abs(dq_a)).mean())
        out[str(p)] = {
            "d_guided": d_g,
            "d_random": d_r,
            "d_anti": d_a,
            "passes_corrected": bool(d_g > d_r > d_a and d_g > 0),
            "drain_fraction": float(d_g / np.abs(dq0).mean()),
        }
    return out


def main() -> None:
    results = {}
    for ds in DS:
        print(f"== {ds}", flush=True)
        results[ds] = run(ds)
        print(json.dumps(results[ds], indent=1)[:300], flush=True)
    path = REPO / "artifacts/reports/wait_margin.json"
    path.write_text(json.dumps(results, indent=1))
    print("saved ->", path)


if __name__ == "__main__":
    main()
