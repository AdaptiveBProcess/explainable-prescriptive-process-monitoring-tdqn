"""Base-rate selection-effect check for the BPI 2012 use sketch (paper Sec. 4.2).

Conditional approval rates by whether the case ever received the declared
intervention (W_Nabellen incomplete dossiers). A large conditional gap with an
off-support policy arm reads as *which cases get called*, not *what calling
does*. Output: artifacts/reports/selection_effect_bpi2012.json
"""

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]

df = pd.read_parquet(REPO / "data/bpi2012/interim/clean.parquet")
g = df.groupby("case_id").agg(
    outcome=("outcome", "first"),
    called=("activity", lambda a: any("Nabellen incomplete" in str(x) for x in a)),
)
splits = json.load(open(REPO / "data/bpi2012/processed/splits.json"))["cases"]
test_ids = {str(x) for x in splits["test"]}
gt = g.loc[g.index.astype(str).isin(test_ids)]

out = {
    "full_log": {
        "n_cases": int(len(g)),
        "called_rate": float(g.called.mean()),
        "approve_given_called": float(g[g.called].outcome.mean()),
        "approve_given_not_called": float(g[~g.called].outcome.mean()),
    },
    "test_split": {
        "n_cases": int(len(gt)),
        "called_rate": float(gt.called.mean()),
        "approve_given_called": float(gt[gt.called].outcome.mean()),
        "approve_given_not_called": float(gt[~gt.called].outcome.mean()),
    },
}
outpath = REPO / "artifacts/reports/selection_effect_bpi2012.json"
outpath.write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
print("saved ->", outpath)
