"""Paper 2 (MI) - Step 30: causal tracing over twin pairs (Phase 2).

For each twin pair from step 27, patches healthy activations into the risk
run (hook point x position) and measures recovery of the intervention margin
DeltaQ. The aggregate map is the in-distribution causal counterpart of the
IG attributions phi^V / phi^DeltaQ (headline comparison of the paper).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from xppm.interp.hooked_model import HookedTDQN
from xppm.interp.patching import DEFAULT_A0, DEFAULT_A_STAR, aggregate_traces, causal_trace_pair
from xppm.utils.config import Config
from xppm.utils.io import load_npz, load_parquet, save_json, save_parquet
from xppm.utils.logging import ensure_dir, get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Causal tracing over twin pairs")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--ckpt", type=str, required=True, help="Path to Q_theta.ckpt")
    parser.add_argument("--npz", type=str, default="data/processed/D_offline.npz")
    parser.add_argument("--vocab", type=str, default="data/interim/vocab_activity.json")
    parser.add_argument("--pairs", type=str, default="artifacts/interp/pairs.parquet")
    parser.add_argument("--out-dir", type=str, default="artifacts/interp/tracing")
    parser.add_argument("--max-pairs", type=int, default=100)
    parser.add_argument("--a-star", type=int, default=DEFAULT_A_STAR)
    parser.add_argument("--a0", type=int, default=DEFAULT_A0)
    parser.add_argument("--device", type=str, default=None, help="cpu/cuda (default: auto)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.out_dir) / "traces.parquet"
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"{out_path} exists; pass --overwrite to regenerate")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = Config.for_dataset(args.config, args.dataset).raw

    states = load_npz(args.npz)["s"]
    pairs = load_parquet(args.pairs).head(args.max_pairs)
    hooked = HookedTDQN.from_checkpoint(args.ckpt, args.npz, args.vocab, cfg, device)

    traces = []
    for pair_id, row in enumerate(pairs.itertuples(index=False)):
        trace = causal_trace_pair(
            hooked,
            states[row.risk_idx],
            states[row.healthy_idx],
            a_star=args.a_star,
            a0=args.a0,
            device=device,
        )
        trace["pair_id"] = pair_id
        trace["risk_idx"] = row.risk_idx
        trace["healthy_idx"] = row.healthy_idx
        traces.append(trace)
        if (pair_id + 1) % 10 == 0:
            logger.info("Traced %d/%d pairs", pair_id + 1, len(pairs))

    all_traces = pd.concat(traces, ignore_index=True)
    aggregate = aggregate_traces(all_traces)

    ensure_dir(args.out_dir)
    save_parquet(all_traces, out_path)
    aggregate.to_csv(Path(args.out_dir) / "traces_aggregate.csv", index=False)
    top = aggregate.sort_values("recovery_mean", ascending=False).head(10)
    save_json(
        {
            "n_pairs": int(len(pairs)),
            "ckpt": args.ckpt,
            "a_star": args.a_star,
            "a0": args.a0,
            "top_recovery": top.round(4).to_dict(orient="records"),
        },
        Path(args.out_dir) / "tracing_summary.json",
    )
    logger.info("30_causal_tracing completed: %d pairs -> %s", len(pairs), out_path)


if __name__ == "__main__":
    main()
