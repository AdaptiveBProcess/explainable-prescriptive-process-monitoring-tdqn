"""Generate the manager-facing explanation card (paper Fig. 2).

Reads real pipeline output (risk_explanations.json, deltaQ_explanations.json)
and renders the dual-level explanation card. All value displays are ordinal: V
and Delta-Q come from a non-conservatively trained Q_theta (see paper,
Threats), so the card never presents them as calibrated currency.

Two dataset modes:
  - bpi2017ct (paper Fig. 1 since v5): case 11005, a real-log case whose
    at-risk membership is unambiguous (percentile 8.8, ties 2.5%); the levels
    share the lead driver and diverge at rank 2.
  - simbank: case 552, kept in the repository as the tie-caveat illustration
    (its V sits exactly at the 30.5%-tied median).

Narrative strings are computed from the artifact and ASSERTED against it at
generation time, so a regenerated artifact cannot silently contradict the card.

Usage:
    python scripts/generate_explanation_card.py [--dataset bpi2017ct]
        [--case-id N] [--out PATH.pdf]
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

DATASETS = {
    "simbank": {"xai": "artifacts/xai", "default_case": 552},
    "bpi2017ct": {"xai": "artifacts/xai/bpi2017ct", "default_case": 11005},
}

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


def label(name: str) -> str:
    if name in LABELS:
        return LABELS[name]
    parts = name.replace("_", " ").split()
    if len(parts) <= 1:
        return name
    mid = (len(parts) + 1) // 2
    return " ".join(parts[:mid]) + "\n" + " ".join(parts[mid:])


def fmt(v: float) -> str:
    return f"{v:,.0f}" if abs(v) >= 100 else f"{v:.2f}"


def load_case(case_id: int, xai_dir: Path):
    risk = json.load(open(xai_dir / "risk_explanations.json"))
    dq = json.load(open(xai_dir / "deltaQ_explanations.json"))
    r = next(it for it in risk["items"] if it["case_id"] == case_id)
    q = next(it for it in dq["items"] if it["case_id"] == case_id)
    pool = np.array([it["V"] for it in risk["items"]])
    below = float((pool < r["V"]).mean())
    at = float((pool == r["V"]).mean())
    r["_pct_midrank"] = (below + at / 2) * 100
    r["_tie_pct"] = at * 100
    r["_below_median"] = bool(r["V"] < np.median(pool))
    ev_r = sorted(r["top_tokens"], key=lambda t: t["position"])
    ev_q = {t["position"]: t["importance"] for t in q["top_drivers"]}
    events = [
        {"name": t["token_name"], "phi_v": t["importance"], "phi_dq": ev_q.get(t["position"], 0.0)}
        for t in ev_r
    ]
    return r, q, events


def top3(events, key):
    return sorted(range(len(events)), key=lambda i: -events[i][key])[:3]


def pct_share(vals):
    tot = sum(vals)
    return [v / tot * 100 for v in vals]


def build_narrative(ds, events, share_v, share_q, r):
    """Dataset-specific narrative, asserted against the artifact."""
    t3v, t3q = top3(events, "phi_v"), top3(events, "phi_dq")
    nv = [events[i]["name"] for i in t3v]
    nq = [events[i]["name"] for i in t3q]
    if ds == "simbank":
        assert nv.count("validate_application") == 2, nv
        assert t3v[:2] == t3q[:2], (t3v, t3q)
        assert events[t3q[2]]["name"] == "email_customer", nq
        va_v = sum(share_v[i] for i, e in enumerate(events) if e["name"] == "validate_application")
        va_q = sum(share_q[i] for i, e in enumerate(events) if e["name"] == "validate_application")
        return {
            "title": "Same lead events — the levels diverge at rank 3",
            "line_v": f"validate_appl. {va_v:.0f}% (top-3: two of its occurrences)",
            "line_q": f"validate_appl. {va_q:.0f}% (top-3: email_customer enters)",
            "red": (
                f"Why at risk?   The validate/email review cycle carries the risk signal "
                f"(validate_application {va_v:.0f}%); two of its occurrences rank in the top "
                "three — position matters, not just the name."
            ),
            "blue": (
                f"Why act now?   The same events top the margin (validate_application "
                f"{va_q:.0f}%), but the third-ranked driver changes: email_customer matters "
                "for timing, a repeat validation for risk."
            ),
            "hot": "skip_contact",
            "arrow_to_event": max(range(len(events)), key=lambda i: events[i]["phi_v"]),
        }
    if ds == "bpi2017ct":
        # facts the strings below state — fail loudly if a regeneration breaks them
        assert nv[0] == nq[0], (nv, nq)  # shared lead driver
        assert nv[1] != nq[1], (nv, nq)  # divergence at rank 2
        assert r["_below_median"] and r["_tie_pct"] < 5, (r["_pct_midrank"], r["_tie_pct"])
        lead = nv[0]
        return {
            "title": "Same lead driver — the levels diverge at rank 2",
            "line_v": f"{lead} {share_v[t3v[0]]:.0f}%  +  {nv[1]} {share_v[t3v[1]]:.0f}%",
            "line_q": f"{lead} {share_q[t3q[0]]:.0f}%  +  {nq[1]} {share_q[t3q[1]]:.0f}%",
            "red": (
                f"Why at risk?   {lead} leads ({share_v[t3v[0]]:.0f}%), and {nv[1]} — the "
                "application left incomplete — is what keeps this accepted offer at risk of "
                "missing its 21-day service target."
            ),
            "blue": (
                f"Why act now?   The same lead driver ({share_q[t3q[0]]:.0f}%), but rank 2 "
                f"changes to {nq[1]}: for timing, what matters is how far the application has "
                "progressed, not what is missing."
            ),
            "hot": None,
            "arrow_to_event": max(range(len(events)), key=lambda i: events[i]["phi_v"]),
        }
    raise SystemExit(f"no narrative for {ds}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), default="bpi2017ct")
    ap.add_argument("--case-id", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    ds = args.dataset
    case_id = args.case_id or DATASETS[ds]["default_case"]
    out = args.out or REPO / (
        "artifacts/explanation_example.pdf"
        if ds == "bpi2017ct"
        else f"artifacts/explanation_example_{ds}.pdf"
    )

    r, q, events = load_case(case_id, REPO / DATASETS[ds]["xai"])
    v_case, dq_case = r["V"], q["delta_q"]
    share_v = pct_share([e["phi_v"] for e in events])
    share_q = pct_share([e["phi_dq"] for e in events])
    story = build_narrative(ds, events, share_v, share_q, r)

    action = q.get("a_star_name", "intervene").replace("_", " ")
    thresh_txt = (
        f"percentile {r['_pct_midrank']:.1f} (mid-rank), at the $\\tau = p_{{50}}$ threshold"
        if not r["_below_median"]
        else f"percentile {r['_pct_midrank']:.1f}, below the $\\tau = p_{{50}}$ threshold"
    )

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
        17, 40.6, f"Case {case_id}  ·  Loan Application", color="white", fontsize=11.5, va="center"
    )
    ax.text(
        17,
        37.3,
        f"Value estimate V = {fmt(v_case)} (ordinal)  ·  {thresh_txt}",
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
        f"{action} · intervention preferred",
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
        f"margin $\\Delta Q$ = +{fmt(dq_case)} (ordinal units)",
        ha="center",
        va="center",
        color="#7fa8c9",
        fontsize=7.4,
        zorder=3,
    )

    # ---------------- callout: dual-level summary ----------------
    box(33, 24.5, 44, 7.6, "white", ec="#c9d0da", lw=1.0, r_pad=0.7, z=4)
    ax.text(38.5, 30.6, story["title"], fontsize=8.4, fontweight="bold", va="center", zorder=5)
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
    ax.text(42.2, 28.7, story["line_v"], va="center", fontsize=8, zorder=6)
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
    ax.text(42.2, 26.2, story["line_q"], va="center", fontsize=8, zorder=6)

    # ---------------- event strip ----------------
    n = len(events)
    x0, gap = 3.0, 1.35
    bw = (97.0 - x0 - gap * (n - 1)) / n
    yb, bh = 14.5, 5.2
    bar_w = min(1.7, bw * 0.22)
    max_bar = 5.5
    sv = np.sqrt([max(e["phi_v"], 0) for e in events])
    sq = np.sqrt([max(e["phi_dq"], 0) for e in events])
    sv = sv / sv.max() * max_bar
    sq = sq / sq.max() * max_bar
    ax.text(1.2, 23.2, "Why\nat risk?", color=RED, fontsize=7.2, fontweight="bold", va="top")
    ax.text(1.2, 13.2, "Why\nact now?", color=BLUE, fontsize=7.2, fontweight="bold", va="top")
    arrow_i = story["arrow_to_event"]
    for i, e in enumerate(events):
        x = x0 + i * (bw + gap)
        hot = story["hot"] is not None and e["name"] == story["hot"]
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
            label(e["name"]),
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
                alpha=1.0 if (hot or i == arrow_i) else 0.45,
                zorder=2,
            )
        )
        ax.add_patch(
            plt.Rectangle(
                (cx - bar_w / 2, yb - 0.25 - sq[i]),
                bar_w,
                sq[i],
                facecolor=BLUE,
                alpha=1.0 if (hot or i == arrow_i) else 0.45,
                zorder=2,
            )
        )
    ax.annotate(
        "",
        xy=(x0 + arrow_i * (bw + gap) + bw / 2, yb + bh + 0.6 + sv[arrow_i]),
        xytext=(77, 25.6),
        arrowprops=dict(arrowstyle="->", color="#4a5468", lw=1.4),
        zorder=5,
    )

    # ---------------- bottom panel ----------------
    box(1, 1.6, 98, 6.4, PANEL, r_pad=0.8)
    ax.add_patch(Circle((3.6, 6.1), 0.55, facecolor=RED, zorder=3))
    ax.text(5.2, 6.1, story["red"], fontsize=8.2, va="center", zorder=3)
    ax.add_patch(Circle((3.6, 3.4), 0.55, facecolor=BLUE, zorder=3))
    ax.text(5.2, 3.4, story["blue"], fontsize=8.2, va="center", zorder=3)
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

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor=LIGHT)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
