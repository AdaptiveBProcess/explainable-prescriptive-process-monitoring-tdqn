"""Does the trained policy actually use event order?

Permutes each test prefix and measures how much Q, and the recommended action,
move. A v1 (multiset) encoder is invariant by construction; a v2 encoder should
not be.

    python scripts/33_check_order_sensitivity.py --ckpt <Q_theta.ckpt> --data <proc_dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from xppm.rl.train_tdqn import load_dataset_with_splits  # noqa: E402
from xppm.xai.fidelity_tests import _load_q_network  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True, help="processed dir with D_offline.npz + splits.json")
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument(
        "--out",
        default=None,
        help="Write the result JSON here (e.g. artifacts/reports/order_sensitivity/<ds>.json). "
        "Stdout-only runs leave no artifact for the paper to cite.",
    )
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc = Path(args.data)
    q_net = _load_q_network(
        args.ckpt, proc / "D_offline.npz", args.vocab, {"training": {"transformer": {}}}, dev
    )
    test = load_dataset_with_splits(str(proc / "D_offline.npz"), str(proc / "splits.json"), "test")

    rng = np.random.default_rng(args.seed)
    # only prefixes with >= 2 distinct events can be permuted meaningfully
    lengths = test["s_mask"].sum(axis=1).astype(int)
    cand = np.nonzero(lengths >= 3)[0]
    idx = rng.choice(cand, size=min(args.n, cand.shape[0]), replace=False)

    s = test["s"][idx].copy()
    m = test["s_mask"][idx]
    va = test["valid_actions"][idx]

    s_perm = s.copy()
    for i in range(s.shape[0]):
        pos = np.nonzero(m[i] > 0)[0]
        s_perm[i, pos] = s[i, rng.permutation(pos)]

    with torch.no_grad():
        t = lambda x, d: torch.from_numpy(x).to(dev).to(d)  # noqa: E731
        q0 = q_net(t(s, torch.long), t(m, torch.float)).cpu().numpy()
        q1 = q_net(t(s_perm, torch.long), t(m, torch.float)).cpu().numpy()

    masked0 = np.where(va > 0, q0, -np.inf)
    masked1 = np.where(va > 0, q1, -np.inf)
    a0, a1 = masked0.argmax(1), masked1.argmax(1)

    scale = float(np.abs(q0).mean()) or 1.0
    out = {
        "encoder_version": int(getattr(q_net, "encoder_version", 1)),
        "n_prefixes": int(s.shape[0]),
        "mean_abs_delta_q": float(np.abs(q0 - q1).mean()),
        "relative_delta_q": float(np.abs(q0 - q1).mean() / scale),
        "action_flip_rate": float((a0 != a1).mean()),
        "identical_q_fraction": float((np.abs(q0 - q1).max(axis=1) < 1e-5).mean()),
    }
    # A permutation-equivariant encoder is invariant up to floating-point noise,
    # not bit-for-bit: reduction order changes when the inputs are reordered.
    # Judge on the relative move, not on exact equality.
    out["verdict"] = (
        "USES ORDER" if out["relative_delta_q"] > 1e-4 else "ORDER-INVARIANT (multiset)"
    )
    out["ckpt"] = str(args.ckpt)
    out["n_requested"] = int(args.n)
    out["seed"] = int(args.seed)
    print(json.dumps(out, indent=1))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=1))
        print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
