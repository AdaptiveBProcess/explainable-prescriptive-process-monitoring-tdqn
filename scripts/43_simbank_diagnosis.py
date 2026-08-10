"""Why is the risk test weakest on SimBank? (v4 review, point A3/B7)

Recomputes the p=0.1 value-test displacement per case (same apparatus, seeds
and null as 25_absgap_final) and tests the three obvious hypotheses:
  (i)   ties        gap restricted to cases NOT in the exact-median tie mass;
  (ii)  reference   gap by tercile of |V - V(s0)| (non-neutral IG reference);
  (iii) integral    Spearman correlation of per-case completeness error with
                    the per-case paired gap.

Output: artifacts/reports/simbank_diagnosis.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from xppm.rl.train_tdqn import load_dataset_with_splits  # noqa: E402
from xppm.xai.evaluability import DEFAULT_N_RANDOM, DEFAULT_SEED, _random_positions  # noqa: E402
from xppm.xai.fidelity_tests import (  # noqa: E402
    _compute_q_values,
    _load_q_network,
    _perturb_states_mask_topk,
)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
P = 0.1


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    ope = json.load(open(REPO / "artifacts/ope/ope_dr_boa.json"))
    q_net = _load_q_network(
        REPO / ope["metadata"]["ckpt_path"],
        REPO / "data/simbank/processed/D_offline.npz",
        REPO / ope["metadata"]["vocab_path"],
        {"training": {"transformer": {}}},
        DEV,
    )
    test = load_dataset_with_splits(
        str(REPO / "data/simbank/processed/D_offline.npz"),
        str(REPO / "data/simbank/processed/splits.json"),
        "test",
    )
    risk = json.load(open(REPO / "artifacts/xai/risk_explanations.json"))
    items = risk["items"]
    idx = np.array([it["transition_idx"] for it in items])
    s, sm, va = test["s"][idx], test["s_mask"][idx], test["valid_actions"][idx]
    _, v0, _ = _compute_q_values(q_net, s, sm, va, DEV)
    n = len(items)

    ks, top_pos = [], []
    for i, it in enumerate(items):
        k = max(1, int(np.ceil(P * int(sm[i].sum()))))
        ks.append(k)
        pos = [
            t["position"]
            for t in it["top_tokens"]
            if 0 <= t["position"] < s.shape[1] and sm[i][t["position"]] > 0
        ]
        top_pos.append(pos[:k])
    sg, mg = _perturb_states_mask_topk(s, sm, top_pos)
    _, vg, _ = _compute_q_values(q_net, sg, mg, va, DEV)
    absr = []
    for rep in range(DEFAULT_N_RANDOM):
        rnd = _random_positions(sm, np.array(ks), DEFAULT_SEED, rep)
        sr, mr = _perturb_states_mask_topk(s, sm, rnd)
        _, vr, _ = _compute_q_values(q_net, sr, mr, va, DEV)
        absr.append(np.abs(v0 - vr))
    disp_g = np.abs(v0 - vg)
    disp_r = np.stack(absr).mean(0)
    diff = disp_g - disp_r

    def gap_stats(mask: np.ndarray) -> dict:
        d = diff[mask]
        return {
            "n": int(mask.sum()),
            "gap": float(d.mean()),
            "gap_se": float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else None,
            "se_ratio": float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) if len(d) > 1 else None,
        }

    v_items = np.array([it["V"] for it in items])
    med = float(np.median(v_items))
    tied = v_items == med

    # (ii) distance to the IG reference value V(s0)
    stats_all = gap_stats(np.ones(n, dtype=bool))
    v_pad = json.load(open(REPO / "artifacts/reports/paper_stats.json"))["simbank"]["v_pad"]
    dist = np.abs(v_items - v_pad)
    terciles = np.quantile(dist, [1 / 3, 2 / 3])
    t_low = dist <= terciles[0]
    t_mid = (dist > terciles[0]) & (dist <= terciles[1])
    t_high = dist > terciles[1]

    # (iii) completeness error per case (risk target)
    comp = risk["metadata"].get("ig_completeness_risk", {})
    rel = np.array(comp.get("rel_err_per_sample", [np.nan] * n))[:n]
    rho = spearman(rel, diff) if np.isfinite(rel).all() else None

    out = {
        "p": P,
        "n_cases": n,
        "all": stats_all,
        "hypothesis_ties": {
            "median": med,
            "n_tied": int(tied.sum()),
            "tied": gap_stats(tied),
            "untied": gap_stats(~tied),
        },
        "hypothesis_reference": {
            "v_pad": v_pad,
            "tercile_bounds_abs_dist": [float(x) for x in terciles],
            "near_reference": gap_stats(t_low),
            "mid": gap_stats(t_mid),
            "far_from_reference": gap_stats(t_high),
        },
        "hypothesis_integral": {
            "spearman_rel_err_vs_gap": rho,
            "rel_err_median": float(np.nanmedian(rel)),
        },
    }
    path = REPO / "artifacts/reports/simbank_diagnosis.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
