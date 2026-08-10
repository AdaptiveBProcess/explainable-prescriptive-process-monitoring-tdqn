"""Deletion-operator control for the value test (v5, 1.5).

The paper's deletion replaces events with PAD — IG's own reference — which the
text concedes favours reference-based methods. This control reruns the value
test with a non-PAD operator: each deleted event's token is SUBSTITUTED by a
deterministic pseudo-random real activity (never the original, mask unchanged),
so IG gets no reference advantage. If guided > random > anti survives, the
verdicts are not artifacts of the PAD operator.

Output: artifacts/reports/alt_deletion_check.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from xppm.rl.train_tdqn import load_dataset_with_splits  # noqa: E402
from xppm.xai.evaluability import DEFAULT_N_RANDOM, DEFAULT_SEED, _random_positions  # noqa: E402
from xppm.xai.fidelity_tests import _compute_q_values, _load_q_network  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG = {"training": {"transformer": {}}}
P = 0.1

DS = {
    "simbank": ("data/simbank/processed", "artifacts/ope/ope_dr_boa.json", "artifacts/xai"),
    "simbank-ir3": (
        "data/simbank-ir3/processed",
        "artifacts/ope/simbank-ir3/ope_dr_boa.json",
        "artifacts/xai/simbank-ir3",
    ),
    "bpi2012": (
        "data/bpi2012/processed",
        "artifacts/ope/bpi2012/ope_dr_boa.json",
        "artifacts/xai/bpi2012",
    ),
    "bpi2012-offertes": (
        "data/bpi2012-offertes/processed",
        "artifacts/ope/bpi2012-offertes/ope_dr_boa.json",
        "artifacts/xai/bpi2012-offertes",
    ),
    "bpi2017ct": (
        "data/bpi2017ct/processed",
        "artifacts/ope/bpi2017ct/ope_dr_boa.json",
        "artifacts/xai/bpi2017ct",
    ),
    "sepsis": (
        "data/sepsis/processed",
        "artifacts/ope/sepsis/ope_dr_boa.json",
        "artifacts/xai/sepsis",
    ),
}


def substitute(states: np.ndarray, positions: list, vocab_size: int, tag: int) -> np.ndarray:
    """Replace tokens at the given positions with a deterministic pseudo-random
    real activity id != original (mask untouched)."""
    out = states.copy()
    for i, pos_list in enumerate(positions):
        for p_ in pos_list:
            orig = int(out[i, p_])
            rng = np.random.default_rng(DEFAULT_SEED + i * 7919 + int(p_) * 31 + tag)
            repl = orig
            while repl == orig:
                repl = int(rng.integers(1, vocab_size))
            out[i, p_] = repl
    return out


def run(ds: str) -> dict:
    proc, opep, xaip = DS[ds]
    ope = json.load(open(REPO / opep))
    q_net = _load_q_network(
        REPO / ope["metadata"]["ckpt_path"],
        REPO / proc / "D_offline.npz",
        REPO / ope["metadata"]["vocab_path"],
        CONFIG,
        DEV,
    )
    test = load_dataset_with_splits(
        str(REPO / proc / "D_offline.npz"), str(REPO / proc / "splits.json"), "test"
    )
    risk = json.load(open(REPO / xaip / "risk_explanations.json"))
    items = risk["items"]
    idx = np.array([it["transition_idx"] for it in items])
    s, sm, va = test["s"][idx], test["s_mask"][idx], test["valid_actions"][idx]
    vocab_size = int(np.asarray(test["s"]).max()) + 1
    _, v0, _ = _compute_q_values(q_net, s, sm, va, DEV)

    ks, top, bot = [], [], []
    for i, it in enumerate(items):
        k = max(1, int(np.ceil(P * int(sm[i].sum()))))
        ks.append(k)
        pos_t = [
            t["position"]
            for t in it["top_tokens"]
            if 0 <= t["position"] < s.shape[1] and sm[i][t["position"]] > 0
        ][:k]
        pos_b = [
            t["position"]
            for t in it.get("bottom_tokens", [])
            if 0 <= t["position"] < s.shape[1] and sm[i][t["position"]] > 0
        ][:k]
        top.append(pos_t)
        bot.append(pos_b)

    _, vg, _ = _compute_q_values(q_net, substitute(s, top, vocab_size, 1), sm, va, DEV)
    _, vb, _ = _compute_q_values(q_net, substitute(s, bot, vocab_size, 2), sm, va, DEV)
    absr = []
    for rep in range(DEFAULT_N_RANDOM):
        rnd = _random_positions(sm, np.array(ks), DEFAULT_SEED, rep)
        _, vr, _ = _compute_q_values(q_net, substitute(s, rnd, vocab_size, 100 + rep), sm, va, DEV)
        absr.append(np.abs(v0 - vr))
    disp_g = np.abs(v0 - vg)
    disp_r = np.stack(absr).mean(0)
    disp_a = np.abs(v0 - vb)
    diff = disp_g - disp_r
    return {
        "p": P,
        "n": len(items),
        "operator": "token substitution (non-PAD)",
        "abs_guided": float(disp_g.mean()),
        "abs_random": float(disp_r.mean()),
        "abs_anti": float(disp_a.mean()),
        "gap": float(diff.mean()),
        "gap_se": float(diff.std(ddof=1) / np.sqrt(len(diff))),
        "ordering_holds": bool(disp_g.mean() > disp_r.mean() > disp_a.mean()),
    }


def main() -> None:
    results = {}
    for ds in DS:
        print(f"== {ds}", flush=True)
        results[ds] = run(ds)
        print(json.dumps(results[ds]), flush=True)
    path = REPO / "artifacts/reports/alt_deletion_check.json"
    path.write_text(json.dumps(results, indent=1))
    print("saved ->", path)


if __name__ == "__main__":
    main()
