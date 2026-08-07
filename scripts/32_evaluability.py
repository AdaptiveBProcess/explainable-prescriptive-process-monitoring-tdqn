"""Compute the paper's Def. 4 (test resolution and explanandum) for every
configuration and write a single reproducible verdict file.

    python scripts/32_evaluability.py [--datasets simbank bpi2012 ...]

Output: artifacts/fidelity/evaluability.json
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
from xppm.xai.evaluability import evaluate_configuration  # noqa: E402
from xppm.xai.fidelity_tests import _compute_q_values, _load_q_network  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG = {"training": {"transformer": {}}}


def _entry(proc: str, ope: str, xai: str) -> dict[str, str]:
    return {"proc": proc, "ope": ope, "xai": xai}


DATASETS: dict[str, dict[str, str]] = {
    "simbank": _entry("data/simbank/processed", "artifacts/ope/ope_dr.json", "artifacts/xai"),
    "simbank-ir3": _entry(
        "data/simbank-ir3/processed",
        "artifacts/ope/simbank-ir3/ope_dr.json",
        "artifacts/xai/simbank-ir3",
    ),
    "bpi2012": _entry(
        "data/bpi2012/processed", "artifacts/ope/bpi2012/ope_dr.json", "artifacts/xai/bpi2012"
    ),
    "bpi2017ct": _entry(
        "data/bpi2017ct/processed", "artifacts/ope/bpi2017ct/ope_dr.json", "artifacts/xai/bpi2017ct"
    ),
    "bpi2017": _entry(
        "data/bpi2017/processed", "artifacts/ope/bpi2017/ope_dr.json", "artifacts/xai/bpi2017"
    ),
    "bpi2020-rfp": _entry(
        "data/bpi2020-rfp/processed",
        "artifacts/ope/bpi2020-rfp/ope_dr.json",
        "artifacts/xai/bpi2020-rfp",
    ),
    "bpi2020-int-decl": _entry(
        "data/bpi2020-int-decl/processed",
        "artifacts/ope/bpi2020-int-decl/ope_dr.json",
        "artifacts/xai/bpi2020-int-decl",
    ),
    "bpi2020-travel": _entry(
        "data/bpi2020-travel/processed",
        "artifacts/ope/bpi2020-travel/ope_dr.json",
        "artifacts/xai/bpi2020-travel",
    ),
}


def run(name: str) -> dict:
    p = DATASETS[name]
    ope = json.load(open(REPO / p["ope"]))
    q_net = _load_q_network(
        REPO / ope["metadata"]["ckpt_path"],
        REPO / p["proc"] / "D_offline.npz",
        REPO / ope["metadata"]["vocab_path"],
        CONFIG,
        DEVICE,
    )
    test = load_dataset_with_splits(
        str(REPO / p["proc"] / "D_offline.npz"), str(REPO / p["proc"] / "splits.json"), "test"
    )

    risk_items = json.load(open(REPO / p["xai"] / "risk_explanations.json"))["items"]
    idx = np.array([it["transition_idx"] for it in risk_items])
    s, sm, va = test["s"][idx], test["s_mask"][idx], test["valid_actions"][idx]

    # explained intervention cases: contrasted pair differs with a nonzero margin
    dq_items = json.load(open(REPO / p["xai"] / "deltaQ_explanations.json"))["items"]
    keep = [
        i
        for i, it in enumerate(dq_items)
        if it.get("a_contrast") is not None
        and int(it["a_star"]) != int(it["a_contrast"])
        and abs(float(it.get("delta_q", 0.0))) > 0.0
    ]
    if keep:
        didx = np.array([dq_items[i]["transition_idx"] for i in keep])
        dq_s, dq_sm, dq_va = test["s"][didx], test["s_mask"][didx], test["valid_actions"][didx]
        pairs = np.array(
            [[int(dq_items[i]["a_star"]), int(dq_items[i]["a_contrast"])] for i in keep]
        )
    else:
        dq_s = dq_sm = dq_va = pairs = None

    # case-level V: one value per case, taken at the case's last decision point,
    # which is the population Def. 4's at-risk band is defined over.
    _, v_all, _ = _compute_q_values(q_net, test["s"], test["s_mask"], test["valid_actions"], DEVICE)
    case_ptr = np.asarray(test["case_ptr"])
    t_ptr = np.asarray(test["t_ptr"]) if "t_ptr" in test else np.arange(case_ptr.shape[0])
    order = np.lexsort((t_ptr, case_ptr))
    cs = case_ptr[order]
    last_of_case = order[np.append(np.nonzero(cs[1:] != cs[:-1])[0], cs.shape[0] - 1)]
    v_case = v_all[last_of_case]

    return evaluate_configuration(
        q_net,
        s,
        sm,
        va,
        DEVICE,
        dq_states=dq_s,
        dq_state_masks=dq_sm,
        dq_valid_actions=dq_va,
        dq_action_pairs=pairs,
        case_level_v=v_case,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--out", default="artifacts/fidelity/evaluability.json")
    args = ap.parse_args()

    results = {}
    for name in args.datasets:
        try:
            results[name] = run(name)
        except FileNotFoundError as exc:
            print(f"-- {name}: skipped ({exc})")
            continue
        r, m, b = (
            results[name]["risk_test"],
            results[name]["margin_test"],
            results[name]["at_risk_band"],
        )
        print(
            f"{name:18s} risk={'OK ' if r['evaluable'] else 'n/e'}"
            f"{'' if r['evaluable'] else '(' + ','.join(r['failing']) + ')':10s}"
            f"  margin={'OK ' if m['evaluable'] else 'n/e'}"
            f"{'' if m['evaluable'] else '(' + ','.join(m['failing']) + ')':12s}"
            f"  band={b.get('ratio', float('nan')):8.2f}"
            f" {'resolvable' if b['resolvable'] else 'not resolvable'}"
        )

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out, "w"), indent=1)
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
