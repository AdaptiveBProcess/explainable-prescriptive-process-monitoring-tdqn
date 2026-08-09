"""C3: quantify ranking separation between phi_V and phi_dQ per dataset.

For each explained case with a nonzero intervention margin, compare the two
attribution rankings over event positions: Spearman rank correlation (over
positions present in both lists) and Jaccard overlap of the top-3 positions.
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO = Path(
    "/home/andrew/Documents/docs/3-resolver-problema-subtema/algorithms-explainability/xppm-tdqn"
)

DATASETS = {
    "bpi2012": REPO / "artifacts/xai/bpi2012",
    "simbank": REPO / "artifacts/xai",
    "simbank-ir3": REPO / "artifacts/xai/simbank-ir3",
    "bpi2017": REPO / "artifacts/xai/bpi2017",
    "bpi2017-sla": REPO / "artifacts/xai/bpi2017ct",
    "bpi2020-rfp": REPO / "artifacts/xai/bpi2020-rfp",
    "bpi2020-int-decl": REPO / "artifacts/xai/bpi2020-int-decl",
    "bpi2020-travel": REPO / "artifacts/xai/bpi2020-travel",
}


def per_case_metrics(risk_item, dq_item):
    rv = {t["position"]: t["importance"] for t in risk_item.get("top_tokens", [])}
    qv = {t["position"]: t["importance"] for t in dq_item.get("top_drivers", [])}
    common = sorted(set(rv) & set(qv))
    out = {}
    if len(common) >= 3:
        rho, _ = spearmanr([rv[p] for p in common], [qv[p] for p in common])
        out["spearman"] = rho
    top_r = set(sorted(rv, key=rv.get, reverse=True)[:3])
    top_q = set(sorted(qv, key=qv.get, reverse=True)[:3])
    if top_r and top_q:
        out["jaccard3"] = len(top_r & top_q) / len(top_r | top_q)
    return out


def main():
    for name, xai in DATASETS.items():
        rf, qf = xai / "risk_explanations.json", xai / "deltaQ_explanations.json"
        if not rf.exists():
            print(f"{name:18s} MISSING {xai}")
            continue
        risk = {it["case_id"]: it for it in json.load(open(rf))["items"]}
        dq = [it for it in json.load(open(qf))["items"] if abs(it.get("delta_q", 0)) > 0]
        sp, jc = [], []
        for it in dq:
            r = risk.get(it["case_id"])
            if r is None:
                continue
            m = per_case_metrics(r, it)
            if "spearman" in m and np.isfinite(m["spearman"]):
                sp.append(m["spearman"])
            if "jaccard3" in m:
                jc.append(m["jaccard3"])
        print(
            f"{name:18s} n_margin_cases={len(dq):4d}  "
            f"spearman_median={np.median(sp) if sp else float('nan'):+.3f} (n={len(sp)})  "
            f"jaccard3_median={np.median(jc) if jc else float('nan'):.2f} "
            f"jaccard3_mean={np.mean(jc) if jc else float('nan'):.2f} (n={len(jc)})"
        )


if __name__ == "__main__":
    main()
