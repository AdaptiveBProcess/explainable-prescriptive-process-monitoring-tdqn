"""Generate the manager-facing explanation card for SimBank case 552 (paper Fig. 2).

Reads real pipeline output (risk_explanations.json, deltaQ_explanations.json) and
renders the dual-level explanation card. All value displays are ordinal: V and
Delta-Q come from a non-conservatively trained Q_theta (see paper, Threats), so
the card never presents them as calibrated currency.

Usage:
    python scripts/generate_explanation_card.py [--case-id 552] [--out PATH.pdf]
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch

REPO = Path(__file__).resolve().parents[1]

NAVY = "#1e2735"
RED = "#e74c3c"
DARKRED = "#8f1f1a"
BLUE = "#3b76d1"
CYAN = "#5bc8e8"
LIGHT = "#f7f9fb"
PANEL = "#eef1f5"
GRAY = "#8a93a3"

LABELS = {
    "initiate_application": "initiate\napplication",
    "start_standard": "start\nstandard",
    "start_priority": "start\npriority",
    "validate_application": "validate\napplication",
    "email_customer": "email\ncustomer",
    "skip_contact": "SKIP\nCONTACT",
    "contact_headquarters": "contact\nheadquarters",
    "call_customer": "call\ncustomer",
}


def load_case(case_id: int):
    risk = json.load(open(REPO / "artifacts/xai/risk_explanations.json"))
    dq = json.load(open(REPO / "artifacts/xai/deltaQ_explanations.json"))
    r = next(it for it in risk["items"] if it["case_id"] == case_id)
    q = next(it for it in dq["items"] if it["case_id"] == case_id)
    # mid-rank percentile of this case's V within the explained pool (ties are
    # heavy; the paper reports all three tie conventions in paper_stats.json)
    pool = [it["V"] for it in risk["items"]]
    below = sum(v < r["V"] for v in pool)
    at = sum(v == r["V"] for v in pool)
    r["_percentile_midrank"] = (below + at / 2) / len(pool) * 100
    ev_r = sorted(r["top_tokens"], key=lambda t: t["position"])
    ev_q = {t["position"]: t["importance"] for t in q["top_drivers"]}
    events = [
        {
            "name": t["token_name"],
            "phi_v": t["importance"],
            "phi_dq": ev_q.get(t["position"], 0.0),
        }
        for t in ev_r
    ]
    return r, q, events


def pct_share(vals):
    tot = sum(vals)
    return [v / tot * 100 for v in vals]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-id", type=int, default=552)
    ap.add_argument("--out", type=Path, default=REPO / "artifacts/explanation_example.pdf")
    args = ap.parse_args()

    r, q, events = load_case(args.case_id)
    v_case, dq_case = r["V"], q["delta_q"]
    share_v = pct_share([e["phi_v"] for e in events])
    share_q = pct_share([e["phi_dq"] for e in events])

    def top3(key):
        return sorted(range(len(events)), key=lambda i: -events[i][key])[:3]

    # The narrative strings below state three facts about case 552; fail loudly
    # if a regenerated artifact stops supporting them instead of printing a
    # story the data contradicts.
    top3_v, top3_q = top3("phi_v"), top3("phi_dq")
    names_v = [events[i]["name"] for i in top3_v]
    assert names_v.count("validate_application") == 2, names_v
    assert top3_v[:2] == top3_q[:2], (top3_v, top3_q)
    assert events[top3_q[2]]["name"] == "email_customer", events[top3_q[2]]["name"]

    fig = plt.figure(figsize=(11.0, 4.9), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 46)
    ax.axis("off")
    fig.patch.set_facecolor(LIGHT)

    def box(x, y, w, h, fc, ec="none", lw=1.0, r_pad=0.6, z=1):
        p = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={r_pad}",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            zorder=z,
        )
        ax.add_patch(p)
        return p

    # ---------------- header band ----------------
    box(1, 33.5, 98, 11.5, NAVY, r_pad=0.9)
    box(3.2, 37.5, 11.5, 4.6, RED, r_pad=0.5, z=2)
    ax.text(
        8.95,
        39.8,
        "⚠ AT RISK",
        ha="center",
        va="center",
        color="white",
        fontsize=9.5,
        fontweight="bold",
        zorder=3,
    )
    ax.text(
        17,
        40.6,
        f"Case {args.case_id}  ·  Loan Application",
        color="white",
        fontsize=11.5,
        va="center",
    )
    ax.text(
        17,
        37.3,
        f"Value estimate V = {v_case:,.0f} (ordinal)  ·  percentile "
        f"{r['_percentile_midrank']:.1f} (mid-rank), at the $\\tau = p_{{50}}$ threshold",
        color="#b8c0cd",
        fontsize=8.2,
        va="center",
    )
    ax.plot([59.0, 59.0], [35.0, 43.5], color="#39445a", lw=1.0)
    ax.text(61.0, 42.6, "RECOMMENDED ACTION", color=GRAY, fontsize=7.5, va="center")
    box(61.0, 35.3, 37, 6.2, "#16202e", ec=CYAN, lw=1.4, r_pad=0.5, z=2)
    ax.text(
        79.5,
        39.8,
        "contact headquarters · intervention strongly preferred",
        ha="center",
        va="center",
        color=CYAN,
        fontsize=9.4,
        fontweight="bold",
        zorder=3,
    )
    ax.text(
        79.5,
        37.0,
        f"margin $\\Delta Q$ = +{dq_case:,.0f} (ordinal units)",
        ha="center",
        va="center",
        color="#7fa8c9",
        fontsize=7.4,
        zorder=3,
    )

    # ---------------- callout: dual-level summary ----------------
    box(33, 24.5, 44, 7.6, "white", ec="#c9d0da", lw=1.0, r_pad=0.7, z=4)
    ax.text(
        38.5,
        30.6,
        "Same lead events — the levels diverge at rank 3",
        fontsize=8.4,
        fontweight="bold",
        va="center",
        zorder=5,
    )
    box(34.5, 27.6, 6.6, 2.2, "white", ec=RED, lw=1.2, r_pad=0.3, z=5)
    ax.text(
        37.8,
        28.7,
        "$\\phi^V$ risk:",
        ha="center",
        va="center",
        color=RED,
        fontsize=7.6,
        fontweight="bold",
        zorder=6,
    )
    ax.text(
        42.2,
        28.7,
        f"validate_appl. {share_v[2]+share_v[4]+share_v[6]+share_v[9]:.0f}% "
        f"(top-3: two of its occurrences)",
        va="center",
        fontsize=8,
        zorder=6,
    )
    box(34.5, 25.1, 6.6, 2.2, "white", ec=BLUE, lw=1.2, r_pad=0.3, z=5)
    ax.text(
        37.8,
        26.2,
        "$\\phi^{\\Delta Q}$ act:",
        ha="center",
        va="center",
        color=BLUE,
        fontsize=7.6,
        fontweight="bold",
        zorder=6,
    )
    ax.text(
        42.2,
        26.2,
        f"validate_appl. {share_q[2]+share_q[4]+share_q[6]+share_q[9]:.0f}% "
        f"(top-3: email_customer enters)",
        va="center",
        fontsize=8,
        zorder=6,
    )
    ax.annotate(
        "",
        xy=(91.5, 23.6),
        xytext=(77, 25.6),
        arrowprops=dict(arrowstyle="->", color="#4a5468", lw=1.4),
        zorder=5,
    )

    # ---------------- event strip ----------------
    x0, bw, gap = 3.0, 8.2, 1.35
    yb, bh = 14.5, 5.2
    bar_w = 1.7
    max_bar = 5.5
    sv = np.sqrt([e["phi_v"] for e in events])
    sq = np.sqrt([e["phi_dq"] for e in events])
    sv = sv / sv.max() * max_bar
    sq = sq / sq.max() * max_bar
    ax.text(1.2, 23.2, "Why\nat risk?", color=RED, fontsize=7.2, fontweight="bold", va="top")
    ax.text(1.2, 13.2, "Why\nact now?", color=BLUE, fontsize=7.2, fontweight="bold", va="top")
    for i, e in enumerate(events):
        x = x0 + i * (bw + gap)
        hot = e["name"] == "skip_contact"
        box(
            x,
            yb,
            bw,
            bh,
            DARKRED if hot else "white",
            ec=RED if hot else "#d5dae2",
            lw=1.6 if hot else 1.0,
            r_pad=0.4,
            z=3,
        )
        ax.text(
            x + bw / 2,
            yb + bh / 2,
            LABELS.get(e["name"], e["name"]),
            ha="center",
            va="center",
            fontsize=6.7,
            color="white" if hot else "#2a3140",
            fontweight="bold",
            zorder=4,
        )
        cx = x + bw / 2
        ax.add_patch(
            plt.Rectangle(
                (cx - bar_w / 2, yb + bh + 0.25),
                bar_w,
                sv[i],
                facecolor=RED,
                alpha=1.0 if hot else 0.45,
                zorder=2,
            )
        )
        ax.add_patch(
            plt.Rectangle(
                (cx - bar_w / 2, yb - 0.25 - sq[i]),
                bar_w,
                sq[i],
                facecolor=BLUE,
                alpha=1.0 if hot else 0.45,
                zorder=2,
            )
        )

    # ---------------- bottom panel ----------------
    box(1, 1.6, 98, 6.4, PANEL, r_pad=0.8)
    ax.add_patch(Circle((3.6, 6.1), 0.55, facecolor=RED, zorder=3))
    ax.text(
        5.2,
        6.1,
        f"Why at risk?   The validate/email review cycle carries the risk signal "
        f"(validate_application {share_v[2]+share_v[4]+share_v[6]+share_v[9]:.0f}%); two of its "
        "occurrences rank in the top three — position matters, not just the name.",
        fontsize=8.2,
        va="center",
        zorder=3,
    )
    ax.add_patch(Circle((3.6, 3.4), 0.55, facecolor=BLUE, zorder=3))
    ax.text(
        5.2,
        3.4,
        f"Why act now?   The same events top the margin "
        f"(validate_application {share_q[2]+share_q[4]+share_q[6]+share_q[9]:.0f}%), but the "
        "third-ranked driver changes: email_customer matters for timing, a repeat validation for risk.",
        fontsize=8.2,
        va="center",
        zorder=3,
    )
    ax.text(
        99,
        0.35,
        "V and $\\Delta Q$ are the model's ordinal value estimates "
        "(uncalibrated; see Threats to Validity) — read magnitudes comparatively, "
        "not as currency.",
        ha="right",
        va="bottom",
        fontsize=6.6,
        color=GRAY,
        style="italic",
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", facecolor=LIGHT)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
