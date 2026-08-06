"""T2.2: prepare the SimBank w=3 log (set_ir_3_levels, delta=0.5 mixture) for the
xppm pipeline.

Relabels each 'calculate_offer' event with the interest rate the bank/RCT chose
(calculate_offer_ir7 / _ir8 / _ir9) so the chosen action is readable from the
next event's activity label — the same convention the binary time_contact_HQ
setting uses (action = next event is 'contact_headquarters').

Input : SimBank/data/loan_log_ir3_100000_delta0.5_{train,val} (pickled DataFrames)
Output: data/raw/loan_log_ir3_delta05 (single pickled DataFrame, train+val cases
        renumbered consecutively; the pipeline does its own case-level split)
"""

import pickle
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SIMBANK = REPO.parent / "SimBank" / "data"
OUT = REPO / "data" / "raw" / "loan_log_ir3_delta05"


def relabel(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    is_offer = df["activity"] == "calculate_offer"
    ir = (df.loc[is_offer, "interest_rate"] * 100).round().astype(int)
    assert set(ir.unique()) <= {7, 8, 9}, f"unexpected rates: {sorted(ir.unique())}"
    df.loc[is_offer, "activity"] = "calculate_offer_ir" + ir.astype(str)
    return df


def main() -> None:
    train = pickle.load(open(SIMBANK / "loan_log_ir3_100000_delta0.5_train", "rb"))
    val = pickle.load(open(SIMBANK / "loan_log_ir3_100000_delta0.5_val", "rb"))
    val = val.copy()
    val["case_nr"] = val["case_nr"] + train["case_nr"].max() + 1
    df = pd.concat([relabel(train), relabel(val)], ignore_index=True)
    # Simulator clock spans centuries (sequential case arrivals): keep microsecond
    # precision so timestamps stay in-range (ns precision caps at year 2262 and
    # preprocess would coerce anything beyond to NaT).
    df["timestamp"] = df["timestamp"].astype("datetime64[us]")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "wb") as f:
        pickle.dump(df, f)
    n_offer = df["activity"].str.startswith("calculate_offer_ir").sum()
    print(
        f"{len(df):,} events, {df['case_nr'].nunique():,} cases, "
        f"{n_offer:,} offer events -> {OUT}"
    )
    print(
        df[df["activity"].str.startswith("calculate_offer")]["activity"].value_counts().to_string()
    )


if __name__ == "__main__":
    main()
