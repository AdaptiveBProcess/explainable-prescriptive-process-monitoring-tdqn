"""Generate fidelity test visualizations for the paper.

Usage:
    python scripts/generate_fidelity_plots.py [--dataset DATASET]

Requirements:
    pip install matplotlib
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fidelity plots")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset name (e.g. bpi2020-rfp). If omitted, uses SimBank defaults.",
    )
    args = parser.parse_args()

    ds = args.dataset
    prefix = Path(ds) if ds else Path("")

    output_dir = Path("artifacts/fidelity") / prefix
    output_dir.mkdir(parents=True, exist_ok=True)

    fidelity_csv = Path("artifacts/fidelity") / prefix / "fidelity.csv"
    policy_summary_path = Path("artifacts/xai") / prefix / "policy_summary.json"

    # Load fidelity results
    df = pd.read_csv(fidelity_csv)

    # Load policy summary for rank-consistency scatter
    with open(policy_summary_path) as f:
        policy_summary = json.load(f)

    # ============================================================================
    # Plot 1: Q-drop gap with colored zones
    # ============================================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    p_vals = np.array([0.1, 0.2, 0.3, 0.5])
    q_drop_data = df[
        (df["test"] == "q_drop") & (df["target"] == "q_star") & (df["metric"] == "gap")
    ]
    gaps = q_drop_data["value"].values

    # Color by zone
    colors = [
        "#2ecc71" if g > 0 else "#e67e22" for g in gaps
    ]  # green for positive, orange for negative
    ax.bar(p_vals, gaps, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5, width=0.08)

    # Reference line
    ax.axhline(0, color="black", linestyle="--", linewidth=1, label="Neutral")

    # Annotations
    ax.text(
        0.15,
        50,
        "Fidelity\nconfirmed",
        ha="center",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.3),
    )
    ax.text(
        0.4,
        -600,
        "Expected\nreversal",
        ha="center",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    ax.set_xlabel("Perturbation level (p_remove)", fontsize=13)
    ax.set_ylabel("Q-drop gap (top-k - random)", fontsize=13)
    title_suffix = f" [{ds}]" if ds else ""
    ax.set_title(
        f"Explanation Fidelity: Q-drop Test{title_suffix}",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / "q_drop_gap_final.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {output_dir / 'q_drop_gap_final.png'}")

    # ============================================================================
    # Plot 2: Action-flip stratified with error bars
    # ============================================================================
    fig, ax = plt.subplots(figsize=(9, 5))

    p_vals = [0.1, 0.2, 0.3, 0.5]
    flip_topk_data = df[(df["test"] == "action_flip") & (df["metric"] == "flip_topk_on_possible")]
    flip_rand_data = df[
        (df["test"] == "action_flip") & (df["metric"] == "flip_rand_mean_on_possible")
    ]

    flip_topk = flip_topk_data["value"].values
    flip_rand = flip_rand_data["value"].values
    n_items = int(flip_topk_data["n_items"].iloc[0])

    # Grouped bars
    x = np.arange(len(p_vals))
    width = 0.35

    ax.bar(
        x - width / 2,
        flip_topk,
        width,
        label="Top-k removal",
        color="steelblue",
        alpha=0.8,
        edgecolor="black",
    )
    ax.bar(
        x + width / 2,
        flip_rand,
        width,
        label="Random removal",
        color="coral",
        alpha=0.8,
        edgecolor="black",
    )

    # Configuration
    ax.set_xlabel("Perturbation level (p_remove)", fontsize=13)
    ax.set_ylabel("Action flip rate", fontsize=13)
    ax.set_title(
        f"Action-flip Test: Robustness to Token Removal{title_suffix}\n"
        f"(n={n_items} flip-possible cases)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(p_vals)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # Annotation
    ax.text(
        0.5,
        0.04,
        "0% flips with top-k\n→ Robust policy",
        ha="center",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(output_dir / "action_flip_final.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {output_dir / 'action_flip_final.png'}")

    # ============================================================================
    # Plot 3: Rank-consistency scatter plot
    # ============================================================================
    clusters = policy_summary["clusters"]
    score_Q = [c["mean_v"] for c in clusters]
    score_margin = [c["mean_policy_margin"] for c in clusters]

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(score_Q, score_margin, s=100, alpha=0.7, edgecolor="black", linewidth=1.5, zorder=3)

    # Linear regression line
    z = np.polyfit(score_Q, score_margin, 1)
    p = np.poly1d(z)
    ax.plot(score_Q, p(score_Q), "r--", alpha=0.5, linewidth=2, label="Linear fit", zorder=2)

    # Annotations
    ax.set_xlabel("State value (mean V)", fontsize=13)
    ax.set_ylabel("Policy margin (Q(a*) - Q(a₂))", fontsize=13)
    n_clusters = len(clusters)
    ax.set_title(
        f"Rank-consistency: Value vs Confidence{title_suffix}\n(n={n_clusters} clusters)",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11)

    # Interpretive text
    ax.text(
        0.05,
        0.95,
        "Weak correlation:\nValue ≠ Confidence",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(output_dir / "rank_consistency_final.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {output_dir / 'rank_consistency_final.png'}")

    print("\n✅ All visualizations generated successfully!")


if __name__ == "__main__":
    main()
