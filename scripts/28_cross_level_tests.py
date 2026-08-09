"""Cross-level fidelity tests: each attribution level under BOTH tests.

Source of the paper's C4 claim that the levels are empirically distinct only
where the contrasted actions compete: for every configuration it runs

- the VALUE test (|V displacement|, guided vs shared random null) with the
  phi^V ranking and with the phi^dQ ranking, on the explained states where
  both rankings exist; and
- the MARGIN test (|dQ| drain + sign flips) with the phi^dQ ranking and with
  the phi^V ranking, on the margin-evaluable items.

Output: artifacts/fidelity/baselines/cross_level_tests.json
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

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 123
NR = 10
P = 0.1
DS = {
    "simbank": ("data/simbank/processed", "artifacts/ope/ope_dr.json", "artifacts/xai"),
    "simbank-ir3": (
        "data/simbank-ir3/processed",
        "artifacts/ope/simbank-ir3/ope_dr.json",
        "artifacts/xai/simbank-ir3",
    ),
    "bpi2012": (
        "data/bpi2012/processed",
        "artifacts/ope/bpi2012/ope_dr.json",
        "artifacts/xai/bpi2012",
    ),
    "bpi2012-offertes": (
        "data/bpi2012-offertes/processed",
        "artifacts/ope/bpi2012-offertes/ope_dr.json",
        "artifacts/xai/bpi2012-offertes",
    ),
    "bpi2017": (
        "data/bpi2017/processed",
        "artifacts/ope/bpi2017/ope_dr.json",
        "artifacts/xai/bpi2017",
    ),
    "bpi2017ct": (
        "data/bpi2017ct/processed",
        "artifacts/ope/bpi2017ct/ope_dr.json",
        "artifacts/xai/bpi2017ct",
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
    "sepsis": ("data/sepsis/processed", "artifacts/ope/sepsis/ope_dr.json", "artifacts/xai/sepsis"),
}
BATCH = 1024


def batched_v(qn, s, sm, va):
    vs = []
    for i in range(0, len(s), BATCH):
        _, v, _ = _compute_q_values(qn, s[i : i + BATCH], sm[i : i + BATCH], va[i : i + BATCH], DEV)
        vs.append(v)
    return np.concatenate(vs)


def batched_q(qn, s, sm, va):
    qs = []
    for i in range(0, len(s), BATCH):
        q, _, _ = _compute_q_values(qn, s[i : i + BATCH], sm[i : i + BATCH], va[i : i + BATCH], DEV)
        qs.append(q)
    return np.concatenate(qs)


def ranking_positions(item, key, s_width, sm_row, k):
    pos = [
        t["position"]
        for t in item.get(key, [])
        if t.get("position") is not None
        and 0 <= t["position"] < s_width
        and sm_row[t["position"]] > 0
    ]
    return pos[:k]


def main():
    rng = np.random.default_rng(SEED)
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
        risk = json.load(open(REPO / xai / "risk_explanations.json"))["items"]
        dq_all = json.load(open(REPO / xai / "deltaQ_explanations.json"))["items"]
        dq_by_tid = {it["transition_idx"]: it for it in dq_all}
        s_all, sm_all, va_all = test["s"], test["s_mask"], test["valid_actions"]
        width = s_all.shape[1]
        res = {}

        # ---- VALUE test on the risk-explained states where both rankings exist
        pairs = [
            (it, dq_by_tid[it["transition_idx"]])
            for it in risk
            if it["transition_idx"] in dq_by_tid
        ]
        if pairs:
            idx = np.array([it["transition_idx"] for it, _ in pairs])
            s, sm, va = s_all[idx], sm_all[idx], va_all[idx]
            v0 = batched_v(qn, s, sm, va)
            ks = [max(1, int(np.ceil(P * int(sm[i].sum())))) for i in range(len(pairs))]
            pos_v = [
                ranking_positions(r, "top_tokens", width, sm[i], ks[i])
                for i, (r, _) in enumerate(pairs)
            ]
            pos_q = [
                ranking_positions(d, "top_drivers", width, sm[i], ks[i])
                for i, (_, d) in enumerate(pairs)
            ]
            sv, mv = _perturb_states_mask_topk(s, sm, pos_v)
            sq, mq = _perturb_states_mask_topk(s, sm, pos_q)
            disp_v = np.abs(v0 - batched_v(qn, sv, mv, va))
            disp_q = np.abs(v0 - batched_v(qn, sq, mq, va))
            rnds = []
            for _ in range(NR):
                rnd = []
                for i in range(len(pairs)):
                    nonpad = np.where(sm[i] > 0)[0]
                    rnd.append(
                        list(rng.choice(nonpad, size=min(ks[i], len(nonpad)), replace=False))
                    )
                sr, mr = _perturb_states_mask_topk(s, sm, rnd)
                rnds.append(np.abs(v0 - batched_v(qn, sr, mr, va)))
            disp_r = np.stack(rnds).mean(0)
            res["value_test"] = {
                "n": len(pairs),
                "guided_phi_v": float(disp_v.mean()),
                "guided_phi_dq": float(disp_q.mean()),
                "random": float(disp_r.mean()),
                "gap_phi_v": float((disp_v - disp_r).mean()),
                "gap_phi_v_se": float((disp_v - disp_r).std(ddof=1) / np.sqrt(len(pairs))),
                "gap_phi_dq": float((disp_q - disp_r).mean()),
                "gap_phi_dq_se": float((disp_q - disp_r).std(ddof=1) / np.sqrt(len(pairs))),
            }

        # ---- MARGIN test on margin-evaluable items, both rankings
        margin_items = [
            it
            for it in dq_all
            if it.get("a_star") != it.get("a_contrast") and abs(it.get("delta_q", 0)) > 0
        ]
        risk_by_tid = {it["transition_idx"]: it for it in risk}
        margin_items = [it for it in margin_items if it["transition_idx"] in risk_by_tid]
        if margin_items:
            idx = np.array([it["transition_idx"] for it in margin_items])
            s, sm, va = s_all[idx], sm_all[idx], va_all[idx]
            a_star = np.array([it["a_star"] for it in margin_items])
            a_con = np.array([it["a_contrast"] for it in margin_items])
            rows = np.arange(len(margin_items))
            q0 = batched_q(qn, s, sm, va)
            dq0 = q0[rows, a_star] - q0[rows, a_con]
            ks = [max(1, int(np.ceil(P * int(sm[i].sum())))) for i in range(len(margin_items))]

            def margin_after(pos_lists):
                sp, mp = _perturb_states_mask_topk(s, sm, pos_lists)
                qm = batched_q(qn, sp, mp, va)
                return qm[rows, a_star] - qm[rows, a_con]

            pos_q = [
                ranking_positions(it, "top_drivers", width, sm[i], ks[i])
                for i, it in enumerate(margin_items)
            ]
            pos_v = [
                ranking_positions(
                    risk_by_tid[it["transition_idx"]], "top_tokens", width, sm[i], ks[i]
                )
                for i, it in enumerate(margin_items)
            ]
            dq_q = margin_after(pos_q)
            dq_v = margin_after(pos_v)
            red_r, flip_r = [], []
            for _ in range(NR):
                rnd = []
                for i in range(len(margin_items)):
                    nonpad = np.where(sm[i] > 0)[0]
                    rnd.append(
                        list(rng.choice(nonpad, size=min(ks[i], len(nonpad)), replace=False))
                    )
                dq_r = margin_after(rnd)
                red_r.append(np.abs(dq0) - np.abs(dq_r))
                flip_r.append((np.sign(dq_r) != np.sign(dq0)).mean())
            red_r = np.stack(red_r).mean(0)
            res["margin_test"] = {
                "n": len(margin_items),
                "mean_abs_dq": float(np.abs(dq0).mean()),
                "red_phi_dq": float((np.abs(dq0) - np.abs(dq_q)).mean()),
                "red_phi_v": float((np.abs(dq0) - np.abs(dq_v)).mean()),
                "red_random": float(red_r.mean()),
                "flip_phi_dq": float((np.sign(dq_q) != np.sign(dq0)).mean()),
                "flip_phi_v": float((np.sign(dq_v) != np.sign(dq0)).mean()),
                "flip_random": float(np.mean(flip_r)),
            }
        out[ds] = res
        print(ds, json.dumps(res, indent=1)[:400])
    outpath = REPO / "artifacts/fidelity/baselines/cross_level_tests.json"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(outpath, "w"), indent=1)
    print("saved ->", outpath)


if __name__ == "__main__":
    main()
