"""Sensitivity of Def. 4's constants (v5, 1.4). Pure post-processing.

Sweeps the two 3x factors and the 30-case minimum over the stored evaluability
evidence and reports which verdicts change:
  - margin condition (iii): E[|dQ|] >= f * |d_random(0.1)| for f in {2, 3, 4}
  - band resolvability:     IQR(V)  >= f * E[|delta_random|] for f in {2, 3, 4}
  - sample condition (ii):  n >= m for m in {20, 30, 50}

Output: artifacts/reports/threshold_sensitivity.json
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FACTORS = [2.0, 3.0, 4.0]
MINS = [20, 30, 50]


def main() -> None:
    ev = json.loads((REPO / "artifacts/fidelity/evaluability.json").read_text())
    out: dict = {"factors": FACTORS, "min_cases": MINS, "configs": {}}
    for ds, d in ev.items():
        if not isinstance(d, dict) or "margin_test" not in d:
            continue
        mt, band = d["margin_test"], d.get("at_risk_band", {})
        entry: dict = {}
        if mt.get("mean_abs_dq") is not None and mt.get("m_random_0.1"):
            ratio = mt["mean_abs_dq"] / mt["m_random_0.1"]
            entry["target_over_null"] = round(ratio, 2)
            entry["cond_iii"] = {str(f): bool(ratio >= f) for f in FACTORS}
        if mt.get("n_cases") is not None:
            entry["n_margin_cases"] = mt["n_cases"]
            entry["cond_ii"] = {str(m): bool(mt["n_cases"] >= m) for m in MINS}
        if band.get("ratio") is not None:
            entry["band_ratio"] = round(band["ratio"], 2)
            entry["band_resolvable"] = {str(f): bool(band["ratio"] >= f) for f in FACTORS}
        # margin evaluability combining (i) [factor-independent] with swept (ii)+(iii)
        gran = mt.get("granularity")
        if gran is not None and "cond_iii" in entry and "cond_ii" in entry:
            ok_ii, ok_iii = entry["cond_ii"], entry["cond_iii"]
            entry["margin_evaluable"] = {
                f"f={f},m={m}": bool(gran and ok_ii[str(m)] and ok_iii[str(f)])
                for f in FACTORS
                for m in MINS
            }
        out["configs"][ds] = entry

    # headline: does the margin-evaluable set change at the declared m=30?
    cfgs = out["configs"]
    base = {ds for ds, e in cfgs.items() if e.get("margin_evaluable", {}).get("f=3.0,m=30")}
    flips = {}
    for f in FACTORS:
        cur = {ds for ds, e in cfgs.items() if e.get("margin_evaluable", {}).get(f"f={f},m=30")}
        flips[str(f)] = sorted(cur ^ base)
    out["margin_set_changes_vs_declared"] = flips
    out["band_changes_vs_declared"] = {
        str(f): sorted(
            ds
            for ds, e in cfgs.items()
            if "band_resolvable" in e
            and e["band_resolvable"][str(f)] != e["band_resolvable"]["3.0"]
        )
        for f in FACTORS
    }

    path = REPO / "artifacts/reports/threshold_sensitivity.json"
    path.write_text(json.dumps(out, indent=1))
    keys = ("margin_set_changes_vs_declared", "band_changes_vs_declared")
    print(json.dumps({k: out[k] for k in keys}, indent=1))


if __name__ == "__main__":
    main()
