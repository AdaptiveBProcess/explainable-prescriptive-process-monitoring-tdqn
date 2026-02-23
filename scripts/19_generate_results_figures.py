#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


def plot_qdrop(fidelity_csv: Path, out_path: Path) -> None:
    df = pd.read_csv(fidelity_csv)
    qdrop = df[
        (df["test"] == "q_drop") & (df["metric"].isin(["drop_topk_mean", "drop_rand_mean", "gap"]))
    ]
    # Pivot to columns: metric x p_remove
    pivot = qdrop.pivot_table(index="p_remove", columns="metric", values="value")
    p_vals = pivot.index.values

    fig, ax = plt.subplots(figsize=(6, 4))
    width = 0.25
    x = np.arange(len(p_vals))

    ax.bar(x - width, pivot["drop_topk_mean"], width, label="Top-k removal")
    ax.bar(x, pivot["drop_rand_mean"], width, label="Random removal")
    ax.bar(x + width, pivot["gap"], width, label="Gap (Top−Random)")

    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p:.1f}" for p in p_vals])
    ax.set_xlabel("Removal rate $p_{\\text{remove}}$")
    ax.set_ylabel("Q-drop (normalized units)")
    ax.set_title("Q-drop fidelity: Top-k vs random token removal")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_distillation(distill_metrics: Path, out_path: Path) -> None:
    with distill_metrics.open() as f:
        metrics = json.load(f)

    # Expect keys like: action_agreement_global, action_agreement_high_impact,
    # margin_correlation, margin_correlation_high_impact
    global_acc = metrics.get("action_agreement_global", 0.0) * 100.0
    hi_acc = metrics.get("action_agreement_high_impact", 0.0) * 100.0
    global_rho = metrics.get("margin_correlation", 0.0)
    hi_rho = metrics.get("margin_correlation_high_impact", 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True)

    # Action agreement
    axes[0].bar(["Global", "High-impact"], [global_acc, hi_acc], color=["steelblue", "darkgreen"])
    axes[0].set_ylabel("Action agreement (\%)")
    axes[0].set_ylim(90, 102)
    axes[0].set_title("Surrogate fidelity (actions)")
    for idx, val in enumerate([global_acc, hi_acc]):
        axes[0].text(idx, val + 0.5, f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)

    # Margin correlation
    axes[1].bar(["Global", "High-impact"], [global_rho, hi_rho], color=["steelblue", "darkgreen"])
    axes[1].set_ylabel("Spearman $\\rho$ (margin correlation)")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Surrogate fidelity (confidence)")
    for idx, val in enumerate([global_rho, hi_rho]):
        axes[1].text(idx, val + 0.03, f"$\\rho$={val:.2f}", ha="center", va="bottom", fontsize=9)
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_latency_from_decisions(decisions_jsonl: Path, out_path: Path) -> None:
    latencies = []
    with decisions_jsonl.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            # Expect latency in milliseconds under a standard key; adjust here if different.
            if "latency_ms" in rec:
                latencies.append(rec["latency_ms"])
            elif "latency" in rec:
                latencies.append(rec["latency"])
    if not latencies:
        return

    lat = np.array(latencies)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(lat, bins=min(10, len(lat)), color="steelblue", alpha=0.8, edgecolor="black")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Count")
    p50, p95, p99 = np.percentile(lat, [50, 95, 99])
    ax.set_title(
        f"Inference latency distribution (p50={p50:.2f}ms, p95={p95:.2f}ms, p99={p99:.2f}ms)"
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper result figures")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset name (e.g. bpi2020-rfp). If omitted, uses SimBank defaults.",
    )
    args = parser.parse_args()

    ds = args.dataset
    prefix = Path(ds) if ds else Path("")

    fidelity_csv = ARTIFACTS / "fidelity" / prefix / "fidelity.csv"
    distill_metrics = ARTIFACTS / "distill" / prefix / "final" / "fidelity_metrics.json"
    decisions_jsonl = ARTIFACTS / "deploy" / prefix / "v1" / "decisions.jsonl"

    if fidelity_csv.exists():
        plot_qdrop(fidelity_csv, ARTIFACTS / "fidelity" / prefix / "q_drop_gap_final.png")

    if distill_metrics.exists():
        plot_distillation(
            distill_metrics, ARTIFACTS / "distill" / prefix / "fidelity_stratified.png"
        )

    if decisions_jsonl.exists():
        plot_latency_from_decisions(
            decisions_jsonl, ARTIFACTS / "deploy" / prefix / "latency_hist.png"
        )


if __name__ == "__main__":
    main()
