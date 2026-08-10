"""OPE robustness post-processing for the v4 review (no model runs needed).

(a) Support-gate sensitivity: on/off-support verdict per arm for
    epsilon in {0.05, 0.10, 0.15, 0.20}, from the stored ESS fractions of the
    primary (boa) and variant (encoder) runs.
(b) Ratio-cap incidence: bracket of the fraction of per-step ratios sitting at
    the cap (=20), derived from the stored rho_step_percentiles
    [p50, p75, p90, p95, p99] (a percentile equal to the cap means at least
    that tail fraction was clipped).

Output: artifacts/reports/ope_sensitivity.json
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EPS = [0.05, 0.10, 0.15, 0.20]
CAP = 20.0
PCTS = [50, 75, 90, 95, 99]

DATASETS = [
    "simbank",
    "simbank-ir3",
    "bpi2012",
    "bpi2017",
    "bpi2017ct",
    "bpi2020-rfp",
    "bpi2020-int-decl",
    "bpi2020-travel",
    "bpi2012-offertes",
    "sepsis",
]


def ope_path(ds: str, kind: str) -> Path:
    name = f"ope_dr_{kind}.json"
    return REPO / ("artifacts/ope" if ds == "simbank" else f"artifacts/ope/{ds}") / name


def clip_bracket(pcts: list[float]) -> str:
    """Smallest clipped-tail bound implied by which percentiles sit at the cap."""
    at_cap = [p for p, v in zip(PCTS, pcts) if v >= CAP - 1e-9]
    if not at_cap:
        return "<1%"
    return f">={100 - min(at_cap)}%"


def main() -> None:
    out: dict = {"epsilons": EPS, "cap": CAP, "configs": {}}
    for ds in DATASETS:
        entry: dict = {}
        for kind in ("boa", "encoder"):
            p = ope_path(ds, kind)
            if not p.exists():
                continue
            d = json.load(open(p))
            pol_ess = d["diagnostics"]["ess_fraction"]
            noop = d["results"].get("baselines", {}).get("noop", {})
            noop_ess = noop.get("ess_fraction")
            verdicts = {}
            for eps in EPS:
                pol_on = pol_ess >= eps
                noop_on = noop_ess is not None and noop_ess >= eps
                verdicts[str(eps)] = {
                    "policy_on_support": bool(pol_on),
                    "noop_on_support": bool(noop_on),
                    "comparison_licensed": bool(pol_on and noop_on),
                }
            entry[kind] = {
                "policy_ess": pol_ess,
                "noop_ess": noop_ess,
                "verdicts": verdicts,
                "rho_step_percentiles": d["diagnostics"].get("rho_step_percentiles"),
                "clipped_fraction_bracket": clip_bracket(
                    d["diagnostics"].get("rho_step_percentiles", [0] * 5)
                ),
            }
        out["configs"][ds] = entry

    # headline: which comparisons flip between eps=0.10 and eps=0.15 (primary)
    flips = [
        ds
        for ds, e in out["configs"].items()
        if "boa" in e
        and e["boa"]["verdicts"]["0.1"]["comparison_licensed"]
        != e["boa"]["verdicts"]["0.15"]["comparison_licensed"]
    ]
    out["licensed_flips_0.10_to_0.15_primary"] = flips

    path = REPO / "artifacts/reports/ope_sensitivity.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps({"flips_0.10_to_0.15": flips}, indent=1))
    for ds, e in out["configs"].items():
        if "boa" in e:
            print(
                "%-18s policy_ess=%.3f clip=%s"
                % (ds, e["boa"]["policy_ess"], e["boa"]["clipped_fraction_bracket"])
            )


if __name__ == "__main__":
    main()
