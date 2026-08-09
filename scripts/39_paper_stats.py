"""Per-configuration statistics cited in the paper's prose (M3 of the 2nd-round
review): every number in the .tex must trace to an artifact, and these had none.

Emits artifacts/reports/paper_stats.json, namespaced by dataset:
  - l_bar:      mean non-PAD test prefix length (Table 1a column)
  - v_pool:     min/max/median/quartiles of V over the explained pool, tie
                fraction at the median (exact-equality)
  - v_pad:      V of the all-PAD reference state (Sec. 3.4's V(s_0))
  - argmax_off_support: fraction of test transitions whose logged action
                differs from argmax_a Q (the mass a deterministic pi_e drops)
  - simbank only: case-552 V, fraction of the pool sharing that exact value,
                and its percentile under the three tie conventions
  - --latency: dual-level IG wall-clock at 128 steps (batch 1 and 64), simbank

Reads checkpoints from ope_dr_boa.json metadata (primary estimator).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from xppm.rl.train_tdqn import load_dataset_with_splits  # noqa: E402
from xppm.xai.fidelity_tests import _compute_q_values, _load_q_network  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG = {"training": {"transformer": {}}}
BATCH = 1024

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
    "bpi2017": (
        "data/bpi2017/processed",
        "artifacts/ope/bpi2017/ope_dr_boa.json",
        "artifacts/xai/bpi2017",
    ),
    "bpi2017ct": (
        "data/bpi2017ct/processed",
        "artifacts/ope/bpi2017ct/ope_dr_boa.json",
        "artifacts/xai/bpi2017ct",
    ),
    "bpi2020-rfp": (
        "data/bpi2020-rfp/processed",
        "artifacts/ope/bpi2020-rfp/ope_dr_boa.json",
        "artifacts/xai/bpi2020-rfp",
    ),
    "bpi2020-int-decl": (
        "data/bpi2020-int-decl/processed",
        "artifacts/ope/bpi2020-int-decl/ope_dr_boa.json",
        "artifacts/xai/bpi2020-int-decl",
    ),
    "bpi2020-travel": (
        "data/bpi2020-travel/processed",
        "artifacts/ope/bpi2020-travel/ope_dr_boa.json",
        "artifacts/xai/bpi2020-travel",
    ),
    "sepsis": (
        "data/sepsis/processed",
        "artifacts/ope/sepsis/ope_dr_boa.json",
        "artifacts/xai/sepsis",
    ),
}

CASE_552 = 552  # the paper's worked example (SimBank)


def percentile_of(v: float, pool: np.ndarray) -> dict:
    below = float((pool < v).mean() * 100)
    at = float((pool == v).mean() * 100)
    return {
        "ties_below": below,  # ties count above this value
        "ties_midrank": below + at / 2,
        "ties_above": below + at,  # ties count below this value
        "frac_exact_tie_pct": at,
    }


def run(ds: str, do_argmax: bool) -> dict:
    proc, opep, xaip = DS[ds]
    out: dict = {}

    test = load_dataset_with_splits(
        str(REPO / proc / "D_offline.npz"), str(REPO / proc / "splits.json"), "test"
    )
    sm = np.asarray(test["s_mask"])
    out["l_bar"] = float(sm.sum(axis=1).mean())
    out["n_test_transitions"] = int(sm.shape[0])

    risk = json.load(open(REPO / xaip / "risk_explanations.json"))
    pool = np.array([it["V"] for it in risk["items"]], dtype=float)
    med = float(np.median(pool))
    out["v_pool"] = {
        "n": int(pool.size),
        "min": float(pool.min()),
        "p25": float(np.percentile(pool, 25)),
        "median": med,
        "p75": float(np.percentile(pool, 75)),
        "max": float(pool.max()),
        "frac_exact_tie_at_median": float((pool == med).mean()),
    }

    ope = json.load(open(REPO / opep))
    ckpt = REPO / ope["metadata"]["ckpt_path"]
    vocab = REPO / ope["metadata"]["vocab_path"]
    q_net = _load_q_network(ckpt, REPO / proc / "D_offline.npz", vocab, CONFIG, DEV)

    # V of the all-PAD reference (IG baseline state)
    max_len = int(np.asarray(test["s"]).shape[1])
    s_pad = np.zeros((1, max_len), dtype=np.asarray(test["s"]).dtype)
    m_pad = np.zeros((1, max_len), dtype=sm.dtype)
    va_any = np.ones((1, np.asarray(test["valid_actions"]).shape[1]), dtype=float)
    _, v_pad, _ = _compute_q_values(q_net, s_pad, m_pad, va_any, DEV)
    out["v_pad"] = float(v_pad[0])

    if ds == "simbank":
        items = {it["case_id"]: it for it in risk["items"]}
        if CASE_552 in items:
            v552 = float(items[CASE_552]["V"])
            out["case_552"] = {
                "V": v552,
                "is_median": bool(v552 == med),
                "frac_pool_sharing_value": float((pool == v552).mean()),
                "percentile": percentile_of(v552, pool),
            }

    if do_argmax:
        s = np.asarray(test["s"])
        va = np.asarray(test["valid_actions"])
        a_log = np.asarray(test["a"]).astype(int)
        agree = []
        for i in range(0, s.shape[0], BATCH):
            _, _, a_star = _compute_q_values(
                q_net, s[i : i + BATCH], sm[i : i + BATCH], va[i : i + BATCH], DEV
            )
            agree.append(a_star == a_log[i : i + BATCH])
        agree_arr = np.concatenate(agree)
        out["argmax_off_support"] = {
            "frac_logged_action_differs": float(1.0 - agree_arr.mean()),
            "n_transitions": int(agree_arr.size),
        }

    return out


def latency_benchmark() -> dict:
    """Dual-level IG wall-clock on simbank (paper Sec. 3.4 footnote).

    Measured at 128 steps (the default) AND at 512 (what simbank and bpi2017ct
    actually run, incl. the case-552 card), so the cost cited for the
    explanations the paper exhibits has an artifact behind it.
    """
    from xppm.xai.attributions import compute_attributions

    proc, opep, _ = DS["simbank"]
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
    s = np.asarray(test["s"])
    sm = np.asarray(test["s_mask"])
    va = np.asarray(test["valid_actions"])

    def dual(n: int, batch_size: int, n_steps: int) -> float:
        xai_cfg = {"methods": {"risk": {"baseline": "pad", "n_steps_ig": n_steps}}}
        t0 = time.perf_counter()
        for tgt, cid in (("V", None), ("deltaQ", -1)):
            compute_attributions(
                q_net=q_net,
                states=s[:n],
                state_masks=sm[:n],
                valid_actions=va[:n],
                config=xai_cfg,
                device=DEV,
                target=tgt,
                contrast_action_id=cid,
                batch_size=batch_size,
            )
        return time.perf_counter() - t0

    dual(4, 4, 128)  # warmup
    out: dict = {"device": str(DEV)}
    for n_steps in (128, 512):
        t1 = dual(8, 1, n_steps)
        t64 = dual(64, 64, n_steps)
        suffix = "" if n_steps == 128 else f"_{n_steps}"
        out[f"n_steps_ig{suffix}"] = n_steps
        out[f"seconds_per_prefix_batch1{suffix}"] = t1 / 8
        out[f"seconds_per_prefix_batch64{suffix}"] = t64 / 64
        out[f"explanations_per_second_batch64{suffix}"] = 64 / t64
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DS))
    ap.add_argument("--skip-argmax", action="store_true")
    ap.add_argument("--latency", action="store_true")
    ap.add_argument("--out", default="artifacts/reports/paper_stats.json")
    args = ap.parse_args()

    outpath = REPO / args.out
    results = json.loads(outpath.read_text()) if outpath.exists() else {}
    for ds in args.datasets:
        print(f"== {ds}", flush=True)
        results[ds] = run(ds, do_argmax=not args.skip_argmax)
    if args.latency:
        results["latency_simbank"] = latency_benchmark()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
