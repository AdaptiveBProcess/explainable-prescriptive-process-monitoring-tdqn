"""Backfill metadata.risk_threshold + per-item at_risk into existing
risk_explanations.json artifacts.

06_explain_policy now computes tau = configured percentile of V over the
explained pool (xai.risk_threshold_percentile, paper: p50) and stamps each item
with at_risk. Regenerating the explanations would recompute identical V values
at real GPU cost, so this script derives exactly the fields 06 would emit from
the V values already in each artifact. Idempotent.
"""

import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

XAI_DIRS = {
    "simbank": REPO / "artifacts/xai",
    "simbank-ir3": REPO / "artifacts/xai/simbank-ir3",
    "bpi2012": REPO / "artifacts/xai/bpi2012",
    "bpi2012-offertes": REPO / "artifacts/xai/bpi2012-offertes",
    "bpi2017": REPO / "artifacts/xai/bpi2017",
    "bpi2017ct": REPO / "artifacts/xai/bpi2017ct",
    "bpi2020-rfp": REPO / "artifacts/xai/bpi2020-rfp",
    "bpi2020-int-decl": REPO / "artifacts/xai/bpi2020-int-decl",
    "bpi2020-travel": REPO / "artifacts/xai/bpi2020-travel",
    "sepsis": REPO / "artifacts/xai/sepsis",
}


def main() -> None:
    cfg = yaml.safe_load((REPO / "configs/config.yaml").read_text())
    pct = float(cfg["xai"]["risk_threshold_percentile"])
    for ds, d in XAI_DIRS.items():
        path = d / "risk_explanations.json"
        if not path.exists():
            print(f"{ds}: missing, skipped")
            continue
        doc = json.loads(path.read_text())
        pool = np.array([it["V"] for it in doc["items"]], dtype=float)
        tau = float(np.percentile(pool, pct))
        doc["metadata"]["risk_threshold"] = {
            "percentile": pct,
            "tau_value": tau,
            "n_at_risk": int((pool < tau).sum()),
            "n_pool": int(pool.shape[0]),
        }
        for it in doc["items"]:
            it["at_risk"] = bool(it["V"] < tau)
        path.write_text(json.dumps(doc, indent=1))
        print(f"{ds}: tau(p{pct:.0f}) = {tau:.6g}, at_risk {int((pool < tau).sum())}/{pool.size}")


if __name__ == "__main__":
    main()
