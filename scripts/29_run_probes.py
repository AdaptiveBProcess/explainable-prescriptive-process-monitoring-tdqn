"""Paper 2 (MI) - Step 29: linear probes over the cached residual stream.

Tests the Phase-1 hypotheses: per-activity counts (H1), discrete V-level
separability (H2) and last-activity decodability, each against majority
baseline and shuffled-label control. Reads the cache from step 28.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from xppm.interp.probing import (
    activity_counts,
    discretize_values,
    last_activity,
    probe_sweep,
)
from xppm.utils.io import load_npz, save_json
from xppm.utils.logging import ensure_dir, get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run linear probes on cached activations")
    parser.add_argument(
        "--activations", type=str, default="artifacts/interp/activations/pooled_activations.npz"
    )
    parser.add_argument("--npz", type=str, default="data/processed/D_offline.npz")
    parser.add_argument("--out-dir", type=str, default="artifacts/interp/probes")
    parser.add_argument("--n-v-levels", type=int, default=5, help="Clusters for the V collapse")
    parser.add_argument(
        "--max-activities", type=int, default=12, help="Most frequent activities to probe counts"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_csv = Path(args.out_dir) / "probe_results.csv"
    if out_csv.exists() and not args.overwrite:
        raise SystemExit(f"{out_csv} exists; pass --overwrite to regenerate")

    cache = load_npz(args.activations)
    states = load_npz(args.npz)["s"][cache["indices"]]
    vocab_size = int(states.max()) + 1

    hooks = [k for k in cache.keys() if k.startswith("resid_") or k in ("proj", "pooled")]
    activations_by_hook = {name: cache[name] for name in hooks}

    counts = activity_counts(states, vocab_size)
    frequent = np.argsort(counts.sum(axis=0))[::-1][: args.max_activities]
    labels: dict[str, tuple[np.ndarray, str]] = {
        f"count_tok{tok}": (counts[:, tok].astype("float64"), "regression")
        for tok in frequent
        if counts[:, tok].sum() > 0
    }
    labels["last_activity"] = (last_activity(states), "classification")
    labels["v_level"] = (
        discretize_values(cache["v"], n_levels=args.n_v_levels, seed=args.seed),
        "classification",
    )

    logger.info("Probing %d hooks x %d tasks", len(activations_by_hook), len(labels))
    results = probe_sweep(activations_by_hook, labels, seed=args.seed)

    ensure_dir(args.out_dir)
    results.to_csv(out_csv, index=False)
    best = results.loc[results.groupby("task")["score"].idxmax()]
    save_json(
        {
            "n_states": int(len(states)),
            "n_v_levels": args.n_v_levels,
            "best_by_task": best.set_index("task")[["hook", "score", "baseline", "control"]]
            .round(4)
            .to_dict(orient="index"),
        },
        Path(args.out_dir) / "probe_summary.json",
    )
    logger.info("29_run_probes completed -> %s", out_csv)


if __name__ == "__main__":
    main()
