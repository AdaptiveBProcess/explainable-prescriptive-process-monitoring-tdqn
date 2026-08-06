"""Paper 2 (MI) - Step 31: train a sparse autoencoder on cached activations (Phase 4).

Learns a feature dictionary over one hook point of the residual stream
(default: the pooled representation). Feature labeling against the SimBank
simulator happens downstream in analysis notebooks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from xppm.interp.sae import train_sae
from xppm.utils.io import load_npz, save_json
from xppm.utils.logging import ensure_dir, get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an SAE over cached TDQN activations")
    parser.add_argument(
        "--activations", type=str, default="artifacts/interp/activations/pooled_activations.npz"
    )
    parser.add_argument("--hook", type=str, default="pooled", help="Hook point to train on")
    parser.add_argument("--out-dir", type=str, default="artifacts/interp/sae")
    parser.add_argument("--expansion", type=int, default=8)
    parser.add_argument("--l1", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None, help="cpu/cuda (default: auto)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.out_dir) / f"sae_{args.hook}.pt"
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"{out_path} exists; pass --overwrite to regenerate")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cache = load_npz(args.activations)
    if args.hook not in cache:
        raise SystemExit(f"Hook '{args.hook}' not in {args.activations}; found {sorted(cache)}")
    activations = cache[args.hook]

    logger.info(
        "Training SAE on '%s': %d x %d, expansion x%d, l1=%g",
        args.hook,
        activations.shape[0],
        activations.shape[1],
        args.expansion,
        args.l1,
    )
    sae, history = train_sae(
        activations,
        expansion=args.expansion,
        l1_coeff=args.l1,
        lr=args.lr,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        seed=args.seed,
        device=device,
    )

    ensure_dir(args.out_dir)
    torch.save(
        {
            "state_dict": sae.state_dict(),
            "d_in": sae.d_in,
            "d_hidden": sae.d_hidden,
            "hook": args.hook,
        },
        out_path,
    )
    save_json(
        {
            "hook": args.hook,
            "expansion": args.expansion,
            "l1_coeff": args.l1,
            "epochs": args.epochs,
            "seed": args.seed,
            "history": history,
        },
        Path(args.out_dir) / f"sae_{args.hook}_history.json",
    )
    logger.info("31_train_sae completed -> %s (final: %s)", out_path, history[-1])


if __name__ == "__main__":
    main()
