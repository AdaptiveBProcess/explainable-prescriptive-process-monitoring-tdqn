"""Compute cycle time (CT) as an operational side-effect audit metric.

CT is defined as max(timestamp) - min(timestamp) per case, in days, computed
from the historical event log (clean.parquet).  It is NOT optimized—it is
audited to verify that TDQN reward improvements do not incur disproportionate
process duration costs.

Stratification (offline, no simulation):
  - CT No-op cases:  test cases where TDQN recommends no intervention at any
                     decision point (argmax Q = 0 everywhere).
  - CT TDQN cases:   test cases where TDQN recommends >= 1 intervention
                     (argmax Q = 1 at some decision point).

Output: artifacts/cycle_time/{prefix}cycle_time.json

Usage
-----
    python scripts/compute_cycle_time.py                          # SimBank
    python scripts/compute_cycle_time.py --dataset bpi2017
    python scripts/compute_cycle_time.py --dataset bpi2020-rfp
    python scripts/compute_cycle_time.py --dataset bpi2020-int-decl
    python scripts/compute_cycle_time.py --dataset bpi2020-travel
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from xppm.rl.factory import AgentFactory
from xppm.utils.config import Config
from xppm.utils.logging import get_logger

logger = get_logger(__name__)

BATCH_SIZE = 2048


def _resolve_paths(cfg: dict, dataset: str | None) -> dict[str, Path]:
    """Derive all file paths for the given dataset.

    For SimBank (dataset=None) the base config leaves ``{dataset_name}``
    as a literal placeholder, so we fall back to the canonical SimBank paths
    rather than reading the unresolved template strings from config.
    For named datasets the pattern is ``data/{dataset}/...``.
    """
    if dataset:
        processed_dir = Path(f"data/{dataset}/processed")
        interim_dir = Path(f"data/{dataset}/interim")
        npz_path = processed_dir / "D_offline.npz"
        splits_path = processed_dir / "splits.json"
        clean_path = interim_dir / "clean.parquet"
        ope_path = Path(f"artifacts/ope/{dataset}/ope_dr.json")
        out_dir = Path(f"artifacts/cycle_time/{dataset}")
    else:
        # SimBank defaults (no {dataset_name} substitution needed)
        npz_path = Path("data/processed/D_offline.npz")
        splits_path = Path("data/processed/splits.json")
        clean_path = Path("data/interim/clean.parquet")
        ope_path = Path("artifacts/ope/ope_dr.json")
        out_dir = Path("artifacts/cycle_time")

    return {
        "npz": npz_path,
        "splits": splits_path,
        "clean": clean_path,
        "ope": ope_path,
        "out_dir": out_dir,
    }


def _load_ckpt_vocab_from_ope(ope_path: Path) -> tuple[Path, Path]:
    """Read ckpt_path and vocab_path from the OPE report JSON."""
    with ope_path.open() as f:
        report = json.load(f)
    meta = report.get("metadata", {})
    ckpt_path = Path(meta["ckpt_path"])
    vocab_path = Path(meta["vocab_path"])
    return ckpt_path, vocab_path


def _compute_ct_days(clean_path: Path) -> pd.Series:
    """Return cycle time in days indexed by case_id."""
    df = pd.read_parquet(clean_path, columns=["case_id", "timestamp"])
    ct = df.groupby("case_id")["timestamp"].agg(
        lambda x: (x.max() - x.min()).total_seconds() / 86400.0
    )
    return ct


def _get_tdqn_interventions(
    npz_path: Path,
    splits_path: Path,
    ckpt_path: Path,
    vocab_path: Path,
    cfg: dict,
    device: torch.device,
) -> dict[int, bool]:
    """Return {case_id: tdqn_intervenes} for all test cases.

    A case is marked intervenes=True if TDQN recommends action=1 (intervene)
    at any of its decision points in the test split.
    """
    # Load dataset
    data = np.load(str(npz_path), allow_pickle=True)
    case_ptr: np.ndarray = data["case_ptr"]  # (N,) int32
    s: np.ndarray = data["s"]  # (N, max_len) int
    s_mask: np.ndarray = data["s_mask"]  # (N, max_len) binary
    valid_actions: np.ndarray = data["valid_actions"]  # (N, n_actions) binary

    # Load test case IDs
    with splits_path.open() as f:
        splits = json.load(f)
    test_ids = np.array(splits["cases"]["test"], dtype=np.int32)
    test_id_set = set(test_ids.tolist())

    # Filter test transitions
    test_mask = np.isin(case_ptr, test_ids)
    s_test = s[test_mask]
    sm_test = s_mask[test_mask]
    va_test = valid_actions[test_mask]
    cp_test = case_ptr[test_mask]

    logger.info("Test transitions: %d  (from %d test cases)", s_test.shape[0], len(test_ids))

    # Load TDQN
    q_net = AgentFactory.load(ckpt_path, npz_path, vocab_path, cfg, device)
    q_net.eval()

    # Forward pass in batches
    n = s_test.shape[0]
    all_actions = np.empty(n, dtype=np.int64)

    with torch.no_grad():
        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)
            s_batch = torch.from_numpy(s_test[start:end]).long().to(device)
            sm_batch = torch.from_numpy(sm_test[start:end]).float().to(device)
            va_batch = va_test[start:end]  # numpy, used for masking

            q_vals = q_net(s_batch, sm_batch)  # (batch, n_actions)
            q_np = q_vals.cpu().numpy()

            # Mask invalid actions: set Q = -inf where valid_actions == 0
            invalid = va_batch == 0
            q_np[invalid] = -np.inf

            all_actions[start:end] = np.argmax(q_np, axis=1)

    # Aggregate per case: intervenes = any(action == 1)
    intervenes: dict[int, bool] = {cid: False for cid in test_id_set}
    for i, cid in enumerate(cp_test.tolist()):
        if all_actions[i] == 1:
            intervenes[cid] = True

    return intervenes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute cycle time as operational side-effect audit (post-hoc)."
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name. Loads configs/datasets/{name}.yaml on top of --config.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    config_obj = Config.for_dataset(args.config, args.dataset)
    cfg = config_obj.raw

    device = torch.device(args.device)
    ds_label = args.dataset or "simbank"

    paths = _resolve_paths(cfg, args.dataset)

    # Validate required files exist
    for key in ("npz", "splits", "clean", "ope"):
        p = paths[key]
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}  (key={key})")

    logger.info("=== Cycle Time Audit: %s ===", ds_label)
    logger.info("  clean.parquet : %s", paths["clean"])
    logger.info("  D_offline.npz : %s", paths["npz"])
    logger.info("  splits.json   : %s", paths["splits"])
    logger.info("  ope_dr.json   : %s", paths["ope"])

    # 1. Compute CT per case from historical log
    logger.info("Computing CT from event log...")
    ct_days = _compute_ct_days(paths["clean"])

    # 2. Load checkpoint/vocab paths from OPE report
    ckpt_path, vocab_path = _load_ckpt_vocab_from_ope(paths["ope"])
    logger.info("Checkpoint : %s", ckpt_path)
    logger.info("Vocab      : %s", vocab_path)

    # 3. Get TDQN intervention decisions per test case
    logger.info("Running TDQN forward pass on test transitions...")
    intervenes = _get_tdqn_interventions(
        paths["npz"], paths["splits"], ckpt_path, vocab_path, cfg, device
    )

    # 4. Load test case IDs
    with paths["splits"].open() as f:
        splits = json.load(f)
    test_ids = splits["cases"]["test"]

    # 5. Compute CT statistics
    ct_test = ct_days.reindex(test_ids)

    # Flag any missing CTs (edge case: cases without events shouldn't happen)
    n_missing = ct_test.isna().sum()
    if n_missing > 0:
        logger.warning("%d test cases have no CT (no events in log); excluding.", n_missing)
        ct_test = ct_test.dropna()

    intervenes_arr = pd.Series({cid: intervenes.get(cid, False) for cid in ct_test.index})

    ct_noop = ct_test[~intervenes_arr].values
    ct_tdqn = ct_test[intervenes_arr].values
    intervention_rate = float(intervenes_arr.mean())

    ct_noop_mean = float(ct_noop.mean()) if len(ct_noop) > 0 else float("nan")
    ct_tdqn_mean = float(ct_tdqn.mean()) if len(ct_tdqn) > 0 else float("nan")
    ct_overall = float(ct_test.mean())

    if ct_noop_mean > 0 and not np.isnan(ct_noop_mean) and not np.isnan(ct_tdqn_mean):
        delta_days = ct_tdqn_mean - ct_noop_mean
        delta_pct = 100.0 * delta_days / ct_noop_mean
    else:
        delta_days = float("nan")
        delta_pct = float("nan")

    result = {
        "dataset": ds_label,
        "n_test_cases": len(test_ids),
        "n_test_cases_with_ct": int(len(ct_test)),
        "intervention_rate_tdqn": round(intervention_rate, 4),
        "n_noop_cases": int((~intervenes_arr).sum()),
        "n_tdqn_cases": int(intervenes_arr.sum()),
        "ct_noop_cases_days": round(ct_noop_mean, 4),
        "ct_tdqn_cases_days": round(ct_tdqn_mean, 4),
        "delta_days": round(delta_days, 4),
        "delta_pct": round(delta_pct, 4),
        "ct_overall_days": round(ct_overall, 4),
    }

    # Log warnings if delta_pct is extreme
    if not np.isnan(delta_pct) and abs(delta_pct) > 50:
        logger.warning(
            "delta_pct=%.1f%% is large (>50%%). Investigate before using in paper.", delta_pct
        )

    logger.info("Results:")
    for k, v in result.items():
        logger.info("  %-35s %s", k, v)

    # 6. Save JSON artifact (replace float NaN/inf with null for strict JSON compliance)
    def _nan_to_null(v: object) -> object:
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return None
        return v

    result_serializable = {k: _nan_to_null(v) for k, v in result.items()}

    out_dir: Path = paths["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cycle_time.json"
    with out_path.open("w") as f:
        json.dump(result_serializable, f, indent=2)
    logger.info("Saved: %s", out_path)


if __name__ == "__main__":
    main()
