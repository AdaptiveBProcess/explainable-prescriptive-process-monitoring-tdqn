"""Paper 2 (MI) - Step 28: cache internal activations at the pooling position.

For each state, stores the residual stream gathered at the last non-PAD
position (the pooling rule of the model), plus the pooled/state_repr vectors,
Q-values and V(s). Output feeds probing (step 29) and SAE training (step 31).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from xppm.interp.cache import gather_last_position, residual_hook_names
from xppm.interp.hooked_model import HookedTDQN
from xppm.utils.config import Config
from xppm.utils.io import load_npz, save_json, save_npz
from xppm.utils.logging import ensure_dir, get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache TDQN activations for MI analysis")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--ckpt", type=str, required=True, help="Path to Q_theta.ckpt")
    parser.add_argument("--npz", type=str, default="data/processed/D_offline.npz")
    parser.add_argument("--vocab", type=str, default="data/interim/vocab_activity.json")
    parser.add_argument("--out-dir", type=str, default="artifacts/interp/activations")
    parser.add_argument("--max-states", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None, help="cpu/cuda (default: auto)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.out_dir) / "pooled_activations.npz"
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"{out_path} exists; pass --overwrite to regenerate")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = Config.for_dataset(args.config, args.dataset).raw

    data = load_npz(args.npz)
    states, valid = data["s"], data["valid_actions"]
    rng = np.random.default_rng(args.seed)
    indices = np.arange(len(states))
    if len(indices) > args.max_states:
        indices = np.sort(rng.choice(indices, size=args.max_states, replace=False))
    states, valid = states[indices], valid[indices]
    mask = (states != 0).astype("float32")

    hooked = HookedTDQN.from_checkpoint(args.ckpt, args.npz, args.vocab, cfg, device)
    resid_hooks = residual_hook_names(hooked.n_layers)
    capture = resid_hooks + ["pooled", "state_repr"]

    collected: dict[str, list[np.ndarray]] = {name: [] for name in capture + ["q"]}
    logger.info("Caching %s for %d states on %s", capture, len(states), device)
    with torch.no_grad():
        for start in range(0, len(states), args.batch_size):
            end = start + args.batch_size
            s = torch.as_tensor(states[start:end], dtype=torch.long, device=device)
            m = torch.as_tensor(mask[start:end], dtype=torch.float32, device=device)
            q, cache = hooked.run_with_cache(s, m, names=capture)
            for name in resid_hooks:
                collected[name].append(gather_last_position(cache[name], m).cpu().numpy())
            for name in ("pooled", "state_repr"):
                collected[name].append(cache[name].cpu().numpy())
            collected["q"].append(q.cpu().numpy())

    arrays = {name: np.concatenate(chunks) for name, chunks in collected.items()}
    q_masked = np.where(valid.astype(bool), arrays["q"], -np.inf)
    arrays["v"] = q_masked.max(axis=1).astype("float32")
    arrays["indices"] = indices

    ensure_dir(args.out_dir)
    save_npz(out_path, **arrays)
    save_json(
        {
            "n_states": int(len(states)),
            "hooks": capture,
            "ckpt": args.ckpt,
            "npz": args.npz,
            "seed": args.seed,
        },
        Path(args.out_dir) / "cache_report.json",
    )
    logger.info("28_cache_activations completed -> %s", out_path)


if __name__ == "__main__":
    main()
