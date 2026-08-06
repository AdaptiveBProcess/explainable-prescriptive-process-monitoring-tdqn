"""Paper 2 (MI) - Step 27: build counterfactual twin pairs (risk vs. healthy).

Pairs real logged prefixes with the same length and minimal edit distance but
opposite V(s) tails; input for activation patching (step 30). Numbers 20-26
are taken by paper-1 scripts, hence the numbering.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from xppm.interp.hooked_model import compute_state_values
from xppm.interp.pairs import build_twin_pairs
from xppm.rl.factory import AgentFactory
from xppm.utils.config import Config
from xppm.utils.io import load_npz, save_json, save_parquet
from xppm.utils.logging import ensure_dir, get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build twin pairs for activation patching")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--ckpt", type=str, required=True, help="Path to Q_theta.ckpt")
    parser.add_argument("--npz", type=str, default="data/processed/D_offline.npz")
    parser.add_argument("--vocab", type=str, default="data/interim/vocab_activity.json")
    parser.add_argument("--out-dir", type=str, default="artifacts/interp")
    parser.add_argument("--risk-pct", type=float, default=10.0)
    parser.add_argument("--healthy-pct", type=float, default=60.0)
    parser.add_argument("--max-edit-distance", type=int, default=2)
    parser.add_argument("--max-pairs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None, help="cpu/cuda (default: auto)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.out_dir) / "pairs.parquet"
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"{out_path} exists; pass --overwrite to regenerate")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = Config.for_dataset(args.config, args.dataset).raw

    data = load_npz(args.npz)
    states, valid = data["s"], data["valid_actions"]
    mask = (states != 0).astype("float32")

    model = AgentFactory.load(args.ckpt, args.npz, args.vocab, cfg, device)
    logger.info("Computing V(s) for %d states on %s", len(states), device)
    values = compute_state_values(model, states, mask, valid_actions=valid, device=device)

    pairs = build_twin_pairs(
        states,
        values,
        risk_percentile=args.risk_pct,
        healthy_percentile=args.healthy_pct,
        max_edit_distance=args.max_edit_distance,
        max_pairs=args.max_pairs,
        seed=args.seed,
    )

    ensure_dir(args.out_dir)
    save_parquet(pairs, out_path)
    save_json(
        {
            "n_pairs": int(len(pairs)),
            "n_states": int(len(states)),
            "ckpt": args.ckpt,
            "risk_pct": args.risk_pct,
            "healthy_pct": args.healthy_pct,
            "max_edit_distance": args.max_edit_distance,
            "seed": args.seed,
        },
        Path(args.out_dir) / "pairs_report.json",
    )
    logger.info("27_build_interp_pairs completed: %d pairs -> %s", len(pairs), out_path)


if __name__ == "__main__":
    main()
