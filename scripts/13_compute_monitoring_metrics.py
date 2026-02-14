#!/usr/bin/env python3
"""
Compute daily monitoring metrics from decisions.jsonl.

Metrics computed:
- total_requests
- coverage (% surrogate)
- fallback_rate (% baseline)
- override_rate (% override)
- ood_rate (% ood=true)
- latency (p50, p95, p99)
- action_distribution
- confidence stats
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def load_decisions_jsonl(log_path: Path, since_date: str = None):
    """Carga decisiones desde JSONL."""
    decisions = []
    with open(log_path) as f:
        for line in f:
            if line.strip():
                dec = json.loads(line)
                if since_date:
                    if dec["timestamp"] >= since_date:
                        decisions.append(dec)
                else:
                    decisions.append(dec)
    return pd.DataFrame(decisions)


def compute_daily_metrics(df: pd.DataFrame, date: str):
    """
    Compute métricas diarias desde decisions.jsonl.

    Métricas:
    - total_requests
    - coverage (% surrogate)
    - fallback_rate (% baseline)
    - override_rate (% override)
    - ood_rate (% ood=true)
    - latency (p50, p95, p99)
    - action_distribution
    - schema_error_rate (si loggeamos errores)
    """
    total = len(df)

    if total == 0:
        return {"date": date, "total_requests": 0, "error": "No decisions found"}

    # Coverage
    coverage = (df["source"] == "surrogate").sum() / total
    fallback_rate = (df["source"] == "baseline").sum() / total
    override_rate = (df["source"] == "override").sum() / total

    # OOD rate
    ood_rate = df["ood"].sum() / total if "ood" in df.columns else 0.0

    # Latency percentiles
    latency = df["latency_ms"].values
    latency_p50 = np.percentile(latency, 50)
    latency_p95 = np.percentile(latency, 95)
    latency_p99 = np.percentile(latency, 99)

    # Action distribution
    action_dist = df["action_id"].value_counts().to_dict()

    # Confidence stats
    confidence_mean = df["confidence"].mean() if "confidence" in df.columns else 0.0
    confidence_std = df["confidence"].std() if "confidence" in df.columns else 0.0

    metrics = {
        "date": date,
        "total_requests": total,
        "coverage": coverage,
        "fallback_rate": fallback_rate,
        "override_rate": override_rate,
        "ood_rate": ood_rate,
        "latency_p50": latency_p50,
        "latency_p95": latency_p95,
        "latency_p99": latency_p99,
        "confidence_mean": confidence_mean,
        "confidence_std": confidence_std,
        "action_distribution": action_dist,
    }

    return metrics


def compute_drift_psi(current_df: pd.DataFrame, baseline_df: pd.DataFrame, features: list):
    """
    Population Stability Index (PSI) por feature.

    PSI = sum((actual% - expected%) * ln(actual% / expected%))

    Interpretation:
    - PSI < 0.1: No significant change
    - 0.1 <= PSI < 0.2: Moderate change (warning)
    - PSI >= 0.2: Significant change (critical)
    """
    psi_scores = {}

    for feat in features:
        if feat not in current_df.columns or feat not in baseline_df.columns:
            continue

        # Binning (10 bins)
        combined = pd.concat([baseline_df[feat], current_df[feat]])
        _, bins = pd.cut(combined, bins=10, retbins=True, duplicates="drop")

        # Distributions
        baseline_dist = pd.cut(baseline_df[feat], bins=bins).value_counts(normalize=True)
        current_dist = pd.cut(current_df[feat], bins=bins).value_counts(normalize=True)

        # PSI calculation
        psi = 0
        for bin_label in baseline_dist.index:
            expected = baseline_dist.get(bin_label, 1e-6)
            actual = current_dist.get(bin_label, 1e-6)
            psi += (actual - expected) * np.log(actual / expected)

        psi_scores[feat] = psi

    return psi_scores


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--log-path", default="artifacts/deploy/v1/decisions.jsonl")
    parser.add_argument("--output-dir", default="artifacts/monitoring")
    parser.add_argument("--date", default=None, help="YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    # Date
    if args.date:
        date = args.date
    else:
        date = datetime.now().strftime("%Y-%m-%d")

    # Load decisions
    log_path = Path(args.log_path)
    if not log_path.exists():
        print(f"❌ Log file not found: {log_path}")
        return

    # Filter by date
    since = f"{date}T00:00:00"
    until = f"{date}T23:59:59"

    df = load_decisions_jsonl(log_path)
    df_day = df[(df["timestamp"] >= since) & (df["timestamp"] <= until)]

    print(f"📊 Computing metrics for {date}")
    print(f"   Total decisions: {len(df_day)}")

    # Compute metrics
    metrics = compute_daily_metrics(df_day, date)

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON (daily)
    json_path = output_dir / f"metrics_{date}.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"✅ Saved: {json_path}")

    # CSV (append)
    csv_path = output_dir / "daily_metrics.csv"
    df_metrics = pd.DataFrame([metrics])

    if csv_path.exists():
        df_existing = pd.read_csv(csv_path)
        df_metrics = pd.concat([df_existing, df_metrics], ignore_index=True)

    df_metrics.to_csv(csv_path, index=False)
    print(f"✅ Updated: {csv_path}")

    # Print summary
    print("\n📈 Metrics summary:")
    print(f"   Coverage: {metrics['coverage']:.2%}")
    print(f"   Fallback rate: {metrics['fallback_rate']:.2%}")
    print(f"   Override rate: {metrics['override_rate']:.2%}")
    print(f"   OOD rate: {metrics['ood_rate']:.2%}")
    print(
        f"   Latency p50/p95/p99: {metrics['latency_p50']:.1f} / "
        f"{metrics['latency_p95']:.1f} / {metrics['latency_p99']:.1f} ms"
    )


if __name__ == "__main__":
    main()
