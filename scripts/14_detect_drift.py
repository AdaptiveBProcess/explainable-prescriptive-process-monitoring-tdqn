#!/usr/bin/env python3
"""
Detect drift using:
1. OOD rate trend
2. Coverage drop
3. Override rate spike
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


def detect_drift(metrics_dir: Path, lookback_days: int = 7):
    """
    Detect drift usando:
    1. OOD rate trend
    2. Coverage drop
    3. Override rate spike
    """
    # Load recent metrics
    csv_path = metrics_dir / "daily_metrics.csv"
    if not csv_path.exists():
        print("❌ No metrics found")
        return

    df = pd.read_csv(csv_path)
    if len(df) < lookback_days:
        print(f"⚠️ Only {len(df)} days of metrics, need {lookback_days}")
        return

    df = df.tail(lookback_days)

    # Baseline (primeros 3 días del periodo)
    baseline = df.head(3)
    current = df.tail(1)

    # Drift indicators
    drift_signals = {}

    # 1. OOD rate spike
    ood_baseline = baseline["ood_rate"].mean()
    ood_current = current["ood_rate"].iloc[0]
    ood_spike = (ood_current - ood_baseline) / (ood_baseline + 1e-6)

    drift_signals["ood_spike"] = {
        "baseline": float(ood_baseline),
        "current": float(ood_current),
        "spike_pct": float(ood_spike),
        "alert": ood_spike > 2.0,  # 200% increase
    }

    # 2. Coverage drop
    coverage_baseline = baseline["coverage"].mean()
    coverage_current = current["coverage"].iloc[0]
    coverage_drop = (coverage_baseline - coverage_current) / (coverage_baseline + 1e-6)

    drift_signals["coverage_drop"] = {
        "baseline": float(coverage_baseline),
        "current": float(coverage_current),
        "drop_pct": float(coverage_drop),
        "alert": coverage_drop > 0.2,  # 20% drop
    }

    # 3. Override rate spike
    override_baseline = baseline["override_rate"].mean()
    override_current = current["override_rate"].iloc[0]
    override_spike = (override_current - override_baseline) / (override_baseline + 1e-6)

    drift_signals["override_spike"] = {
        "baseline": float(override_baseline),
        "current": float(override_current),
        "spike_pct": float(override_spike),
        "alert": override_spike > 1.0,  # 100% increase
    }

    # Overall drift alert
    any_alert = any(s["alert"] for s in drift_signals.values())

    drift_report = {
        "timestamp": datetime.now().isoformat(),
        "lookback_days": lookback_days,
        "drift_detected": any_alert,
        "signals": drift_signals,
    }

    # Save
    report_path = metrics_dir / "drift_report.json"
    with open(report_path, "w") as f:
        json.dump(drift_report, f, indent=2)

    # Print
    print("\n🔍 Drift Detection Report")
    print(f"   Lookback: {lookback_days} days")
    print(f"   Drift detected: {'⚠️ YES' if any_alert else '✅ NO'}")

    for signal_name, signal in drift_signals.items():
        status = "⚠️ ALERT" if signal["alert"] else "✅ OK"
        print(f"\n   {signal_name}: {status}")
        for k, v in signal.items():
            if k != "alert":
                print(f"     {k}: {v}")

    return drift_report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", default="artifacts/monitoring")
    parser.add_argument("--lookback-days", type=int, default=7)
    args = parser.parse_args()

    detect_drift(Path(args.metrics_dir), args.lookback_days)
