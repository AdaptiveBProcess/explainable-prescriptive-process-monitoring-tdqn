#!/usr/bin/env python3
"""
Generate HTML dashboard from daily_metrics.csv using Plotly.
"""

import json
from pathlib import Path

import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️ Plotly not available. Install with: pip install plotly")


def generate_dashboard(metrics_dir: Path):
    """Generate HTML dashboard from daily_metrics.csv."""
    if not PLOTLY_AVAILABLE:
        print("❌ Plotly not available. Dashboard generation skipped.")
        print("   Install with: pip install plotly")
        return

    # Load metrics
    csv_path = metrics_dir / "daily_metrics.csv"
    if not csv_path.exists():
        print("❌ No metrics found")
        return

    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])

    # Create subplots
    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "Coverage over time",
            "Latency (p95)",
            "OOD Rate",
            "Override Rate",
            "Action Distribution",
            "Confidence",
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "scatter"}],
        ],
    )

    # Plot 1: Coverage
    fig.add_trace(
        go.Scatter(x=df["date"], y=df["coverage"], name="Coverage", mode="lines+markers"),
        row=1,
        col=1,
    )

    # Plot 2: Latency p95
    fig.add_trace(
        go.Scatter(x=df["date"], y=df["latency_p95"], name="Latency p95", mode="lines+markers"),
        row=1,
        col=2,
    )

    # Plot 3: OOD Rate
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["ood_rate"],
            name="OOD Rate",
            mode="lines+markers",
            line=dict(color="red"),
        ),
        row=2,
        col=1,
    )

    # Plot 4: Override Rate
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["override_rate"],
            name="Override Rate",
            mode="lines+markers",
            line=dict(color="orange"),
        ),
        row=2,
        col=2,
    )

    # Plot 5: Action distribution (last day)
    if len(df) > 0:
        last_day = df.iloc[-1]
        # Parse action_distribution (stored as string dict)
        action_dist_str = last_day["action_distribution"]
        if isinstance(action_dist_str, str):
            action_dist = json.loads(action_dist_str.replace("'", '"'))
        else:
            action_dist = action_dist_str if isinstance(action_dist_str, dict) else {}

        actions = list(action_dist.keys())
        counts = list(action_dist.values())

        fig.add_trace(go.Bar(x=[str(a) for a in actions], y=counts, name="Actions"), row=3, col=1)

    # Plot 6: Confidence
    fig.add_trace(
        go.Scatter(
            x=df["date"], y=df["confidence_mean"], name="Confidence mean", mode="lines+markers"
        ),
        row=3,
        col=2,
    )

    # Layout
    fig.update_layout(title_text="XPPM DSS Monitoring Dashboard", height=1200, showlegend=False)

    # Update axes
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=2)

    # Save
    html_path = metrics_dir / "dashboard.html"
    fig.write_html(html_path)

    print(f"✅ Dashboard saved: {html_path}")
    print(f"   Open in browser: file://{html_path.absolute()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", default="artifacts/monitoring")
    args = parser.parse_args()

    generate_dashboard(Path(args.metrics_dir))
