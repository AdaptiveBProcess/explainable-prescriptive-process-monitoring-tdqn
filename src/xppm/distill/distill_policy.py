"""Policy distillation: TDQN -> Decision Tree surrogate."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from xppm.rl.factory import AgentFactory
from xppm.rl.models.masking import apply_action_mask
from xppm.utils.io import fingerprint_data, load_json, load_npz, load_parquet, save_json
from xppm.utils.logging import ensure_dir, get_logger
from xppm.utils.seed import set_seed

logger = get_logger(__name__)


def select_common_states(
    dataset: dict[str, np.ndarray],
    splits: dict[str, Any],
    n_target: int,
    seed: int,
) -> np.ndarray:
    """Select common states with stratified sampling.

    Estratifica por:
    - Longitud del prefijo (bins: <5, 5-10, >10)
    - Acción recomendada (para balancear do_nothing vs intervención)

    Args:
        dataset: Full dataset dict
        splits: Splits dict with 'test' case IDs
        n_target: Target number of states
        seed: Random seed

    Returns:
        Selected transition indices
    """
    test_case_ids = set(splits["cases"]["test"])
    test_mask = np.isin(dataset["case_ptr"], list(test_case_ids))
    test_indices = np.where(test_mask)[0]

    if len(test_indices) == 0:
        raise ValueError("No test transitions found")

    # Get test data
    seq_lens = dataset["t_ptr"][test_indices] if "t_ptr" in dataset else np.ones(len(test_indices))
    # For now, use behavior actions as proxy for recommended actions
    # (we'll recompute with teacher later)
    actions = dataset["a"][test_indices]

    # Create bins for prefix length
    len_bins = np.digitize(seq_lens, bins=[5, 10])

    # Combine strata: (len_bin, action)
    strata = [f"{lb}_{a}" for lb, a in zip(len_bins, actions)]

    # Stratified sampling
    rng = np.random.default_rng(seed)
    unique_strata = list(set(strata))
    n_per_stratum = max(1, n_target // len(unique_strata))

    selected_list: list[int] = []
    for stratum in unique_strata:
        stratum_mask = np.array([s == stratum for s in strata])
        stratum_indices = test_indices[stratum_mask]
        if len(stratum_indices) > 0:
            n_select = min(n_per_stratum, len(stratum_indices))
            chosen = rng.choice(stratum_indices, size=n_select, replace=False)
            selected_list.extend(chosen.tolist())

    # If we need more, fill randomly
    if len(selected_list) < n_target:
        remaining = n_target - len(selected_list)
        remaining_indices = [idx for idx in test_indices if idx not in selected_list]
        if len(remaining_indices) > 0:
            additional = rng.choice(
                remaining_indices, size=min(remaining, len(remaining_indices)), replace=False
            )
            selected_list.extend(additional.tolist())

    return np.array(selected_list[:n_target])


def select_high_impact_states(
    deltaq_explanations_path: str | Path,
    dataset: dict[str, np.ndarray],
    n_target: int,
) -> np.ndarray:
    """Select states with highest deltaQ (top n_target).

    Args:
        deltaq_explanations_path: Path to deltaQ_explanations.json
        dataset: Full dataset dict (for matching case_id and t)
        n_target: Target number of states

    Returns:
        Selected transition indices
    """
    explanations_data = load_json(deltaq_explanations_path)

    # Handle both formats: {"items": [...]} or direct list
    if isinstance(explanations_data, dict) and "items" in explanations_data:
        explanations = explanations_data["items"]
    elif isinstance(explanations_data, list):
        explanations = explanations_data
    else:
        raise ValueError(f"Unexpected format in {deltaq_explanations_path}")

    # Sort by delta_q descending
    sorted_exp = sorted(explanations, key=lambda x: x.get("delta_q", 0.0), reverse=True)

    # Build mapping from (case_id, t) to transition index
    case_ptrs = dataset["case_ptr"]
    t_ptrs = dataset.get("t_ptr", np.ones(len(case_ptrs), dtype=np.int32))
    case_t_to_idx: dict[tuple[int, int], int] = {}
    for idx in range(len(case_ptrs)):
        key = (int(case_ptrs[idx]), int(t_ptrs[idx]))
        if key not in case_t_to_idx:  # Take first occurrence
            case_t_to_idx[key] = idx

    # Take top n_target, matching by case_id and t
    high_impact_idx: list[int] = []
    for exp in sorted_exp[:n_target]:
        case_id = exp.get("case_id")
        t = exp.get("t")
        if case_id is not None and t is not None:
            key = (int(case_id), int(t))
            if key in case_t_to_idx:
                high_impact_idx.append(case_t_to_idx[key])
            else:
                logger.debug("No match for case_id=%d, t=%d", case_id, t)
        elif "transition_idx" in exp:
            # Fallback: use transition_idx if available
            high_impact_idx.append(int(exp["transition_idx"]))

    if len(high_impact_idx) < n_target:
        logger.warning(
            "Only found %d/%d high-impact states (missing case_id/t or transition_idx)",
            len(high_impact_idx),
            n_target,
        )

    return np.array(high_impact_idx)


def build_distillation_dataset(
    dataset: dict[str, np.ndarray],
    splits: dict[str, Any],
    deltaq_path: str | Path | None,
    n_common: int,
    n_high_impact: int,
    seed: int,
    output_path: str | Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build distillation dataset: 70% common + 30% high-impact.

    Args:
        dataset: Full dataset dict
        splits: Splits dict
        deltaq_path: Path to deltaQ_explanations.json (optional)
        n_common: Number of common states
        n_high_impact: Number of high-impact states
        seed: Random seed
        output_path: Path to save selection metadata

    Returns:
        (selected_indices, selection_metadata)
    """
    # Select common states
    common_idx = select_common_states(dataset, splits, n_common, seed)

    # Select high-impact states
    if deltaq_path and Path(deltaq_path).exists():
        high_impact_idx = select_high_impact_states(deltaq_path, dataset, n_high_impact)
    else:
        logger.warning("deltaQ explanations not found, using random for high-impact")
        test_case_ids = set(splits["cases"]["test"])
        test_mask = np.isin(dataset["case_ptr"], list(test_case_ids))
        test_indices = np.where(test_mask)[0]
        rng = np.random.default_rng(seed + 1)
        high_impact_idx = rng.choice(
            test_indices, size=min(n_high_impact, len(test_indices)), replace=False
        )

    # Combine (remove duplicates)
    all_idx = np.unique(np.concatenate([common_idx, high_impact_idx]))

    # Create metadata with detailed statistics
    requested_high_impact = n_high_impact
    matched_high_impact = len(high_impact_idx)
    match_rate = matched_high_impact / requested_high_impact if requested_high_impact > 0 else 0.0

    selection_metadata = {
        "n_common": len(common_idx),
        "n_high_impact": len(high_impact_idx),
        "n_total": len(all_idx),
        "n_overlap": len(common_idx) + len(high_impact_idx) - len(all_idx),
        "selection_strategy": "70% stratified common + 30% top deltaQ",
        "high_impact_selection": {
            "requested": requested_high_impact,
            "matched": matched_high_impact,
            "match_rate": match_rate,
            "fallback_used": match_rate < 0.8,  # Flag if significant fallback
        },
        "indices": all_idx.tolist(),
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
    }

    # Save
    ensure_dir(Path(output_path).parent)
    save_json(selection_metadata, output_path)

    return all_idx, selection_metadata


def extract_tabular_features(
    dataset: dict[str, np.ndarray],
    clean_df: pd.DataFrame,
    indices: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    """Extract tabular features for distillation.

    Args:
        dataset: Dataset dict with case_ptr, t_ptr
        clean_df: Clean event log DataFrame
        indices: Transition indices to extract features for
        config: Config dict

    Returns:
        (features_array, feature_names)
    """
    case_ptrs = dataset["case_ptr"][indices]
    # t_ptr is 1-indexed prefix length (t_ptr=1 means first event, t_ptr=2 means second event, etc.)
    if "t_ptr" in dataset:
        t_ptrs = dataset["t_ptr"][indices]
    else:
        # Fallback: compute from s_mask
        logger.warning("t_ptr not found in dataset, computing from s_mask")
        s_mask = dataset["s_mask"][indices]
        t_ptrs = s_mask.sum(axis=1).astype(np.int32)

    # Get vocabulary for activity counts
    vocab_path = (
        config.get("encoding", {})
        .get("output", {})
        .get("vocab_activity_path", "data/interim/vocab_activity.json")
    )
    vocab = load_json(vocab_path)
    id2token = vocab.get("id2token", [])
    activity_names = [id2token[i] for i in range(len(id2token)) if isinstance(id2token[i], str)]

    # Map case_id -> sorted events DataFrame
    case_events_map: dict[int, pd.DataFrame] = {}
    for case_id in clean_df["case_id"].unique():
        case_events = (
            clean_df[clean_df["case_id"] == case_id].sort_values("timestamp").reset_index(drop=True)
        )
        case_events_map[case_id] = case_events

    features_list: list[list[float]] = []
    feature_names = [
        "amount",
        "est_quality",
        "unc_quality",
        "cum_cost",
        "elapsed_time",
        "prefix_len",
    ]

    # Add activity count features
    for act_name in activity_names[:10]:  # Limit to top 10 activities
        feature_names.append(f"count_{act_name}")

    for i, idx in enumerate(indices):
        case_id = int(case_ptrs[i])
        t = int(t_ptrs[i])  # t_ptr is 1-indexed (prefix length)

        # Get case events
        if case_id not in case_events_map:
            logger.warning("Case %d not found in clean_df, using defaults", case_id)
            feat = [0.0] * len(feature_names)
            features_list.append(feat)
            continue

        case_events = case_events_map[case_id]

        # Get event at position t-1 (0-indexed)
        # t_ptr=1 means first event (position 0), t_ptr=2 means second event (position 1), etc.
        event_pos = max(0, min(t - 1, len(case_events) - 1))

        event = case_events.iloc[event_pos]

        # Extract features
        feat: list[float] = [
            float(event.get("amount", 0.0)),
            float(event.get("est_quality", 0.0)),
            float(event.get("unc_quality", 0.0)),
            float(event.get("cum_cost", 0.0)),
            float(event.get("elapsed_time", 0.0)),
            float(t),  # prefix_len
        ]

        # Activity counts (from prefix up to this point)
        prefix_events = case_events.iloc[: event_pos + 1]
        for act_name in activity_names[:10]:
            count = int((prefix_events["activity"] == act_name).sum())
            feat.append(float(count))

        features_list.append(feat)

    return np.array(features_list, dtype=np.float32), feature_names


def generate_teacher_labels(
    q_net: torch.nn.Module,
    dataset: dict[str, np.ndarray],
    indices: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate teacher labels (a_star, q_star, margin) from TDQN.

    Args:
        q_net: Trained Q-network
        dataset: Dataset dict
        indices: Transition indices
        device: Device
        batch_size: Batch size for inference

    Returns:
        (y_action, y_qstar, y_margin)
    """
    q_net.eval()
    y_action: list[int] = []
    y_qstar: list[float] = []
    y_margin: list[float] = []

    s = dataset["s"][indices]
    s_mask = dataset["s_mask"][indices]
    valid_actions = dataset["valid_actions"][indices]

    n = len(indices)
    with torch.no_grad():
        for start_idx in range(0, n, batch_size):
            end_idx = min(start_idx + batch_size, n)
            batch_s = torch.from_numpy(s[start_idx:end_idx]).long().to(device)
            batch_sm = torch.from_numpy(s_mask[start_idx:end_idx]).float().to(device)
            batch_va = torch.from_numpy(valid_actions[start_idx:end_idx]).float().to(device)

            # Forward pass
            q_values = q_net(batch_s, batch_sm)  # (batch, n_actions)

            # Apply action mask
            q_masked = apply_action_mask(q_values, batch_va)

            # Get a_star, q_star, margin
            q_masked_np = q_masked.cpu().numpy()
            for j in range(q_masked_np.shape[0]):
                q_row = q_masked_np[j]
                a_star = int(np.argmax(q_row))
                q_star = float(q_row[a_star])

                # Margin: Q(a*) - Q(a_2nd)
                q_sorted = np.sort(q_row)[::-1]
                margin = float(q_sorted[0] - q_sorted[1]) if len(q_sorted) > 1 else 0.0

                y_action.append(a_star)
                y_qstar.append(q_star)
                y_margin.append(margin)

    return np.array(y_action), np.array(y_qstar), np.array(y_margin)


def train_surrogate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_margin_test: np.ndarray,
    high_impact_mask: np.ndarray | None = None,
    max_depth: int = 5,
    min_samples_leaf: int = 50,
    min_samples_split: int = 100,
    random_state: int = 42,
) -> tuple[DecisionTreeClassifier, dict[str, Any]]:
    """Train decision tree surrogate and evaluate fidelity.

    Args:
        X_train: Training features
        y_train: Training labels (actions)
        X_test: Test features
        y_test: Test labels (actions)
        y_margin_test: Test margins (for correlation)
        max_depth: Max tree depth
        min_samples_leaf: Min samples per leaf
        min_samples_split: Min samples to split
        random_state: Random seed

    Returns:
        (trained_tree, fidelity_metrics)
    """
    # Train tree
    surrogate = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        class_weight="balanced",
        random_state=random_state,
    )
    surrogate.fit(X_train, y_train)

    # Evaluate fidelity
    y_pred = surrogate.predict(X_test)
    y_pred_proba = surrogate.predict_proba(X_test)

    # Action agreement (global)
    action_agreement = float(accuracy_score(y_test, y_pred))

    # Action agreement on high-impact subset (if mask provided)
    action_agreement_high_impact: float | None = None
    if high_impact_mask is not None and high_impact_mask.sum() > 0:
        hi_mask = high_impact_mask[: len(y_test)]  # Ensure same length
        if hi_mask.sum() > 0:
            action_agreement_high_impact = float(accuracy_score(y_test[hi_mask], y_pred[hi_mask]))

    # Margin correlation (surrogate confidence vs teacher margin)
    # Use max probability as surrogate confidence
    surrogate_confidence = np.max(y_pred_proba, axis=1)
    margin_corr, _ = spearmanr(y_margin_test, surrogate_confidence)

    # Margin correlation on high-impact subset
    margin_corr_high_impact: float | None = None
    if high_impact_mask is not None and high_impact_mask.sum() > 0:
        hi_mask = high_impact_mask[: len(y_test)]
        if hi_mask.sum() > 5:  # Need at least 5 samples for correlation
            hi_margin = y_margin_test[hi_mask]
            hi_conf = surrogate_confidence[hi_mask]
            hi_corr, _ = spearmanr(hi_margin, hi_conf)
            margin_corr_high_impact = float(hi_corr) if not np.isnan(hi_corr) else None

    metrics = {
        "action_agreement_global": action_agreement,
        "margin_correlation": float(margin_corr) if not np.isnan(margin_corr) else 0.0,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "tree_depth": int(surrogate.get_depth()),
        "tree_n_leaves": int(surrogate.get_n_leaves()),
    }

    if action_agreement_high_impact is not None:
        metrics["action_agreement_high_impact"] = action_agreement_high_impact
    if margin_corr_high_impact is not None:
        metrics["margin_correlation_high_impact"] = margin_corr_high_impact

    return surrogate, metrics


def distill_policy(config: dict[str, Any]) -> dict[str, Any]:
    """Main distillation pipeline: TDQN -> Decision Tree.

    Args:
        config: Full config dict

    Returns:
        Dictionary with paths to generated artifacts
    """
    distill_cfg = config.get("distill", {})
    if not distill_cfg.get("enabled", True):
        logger.info("Distillation disabled in config")
        return {}

    # Setup
    seed = int(distill_cfg.get("seed", config.get("repro", {}).get("seed", 42)))
    set_seed(seed, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s, seed: %d", device, seed)

    # Resolve paths
    ckpt_path = distill_cfg.get("teacher_checkpoint", "artifacts/models/tdqn/Q_theta.ckpt")
    dataset_path = (
        config.get("mdp", {}).get("output", {}).get("path", "data/processed/D_offline.npz")
    )
    splits_path = str(
        Path(config.get("paths", {}).get("data_processed_dir", "data/processed"))
        / config.get("validation_split", {}).get("out_splits_json", "splits.json")
    )
    vocab_path = (
        config.get("encoding", {})
        .get("output", {})
        .get("vocab_activity_path", "data/interim/vocab_activity.json")
    )
    clean_log_path = config.get("data", {}).get("output_clean_path", "data/interim/clean.parquet")
    deltaq_path = (
        Path(config.get("paths", {}).get("artifacts_dir", "artifacts"))
        / config.get("xai", {}).get("out_dir", "xai")
        / config.get("xai", {})
        .get("outputs", {})
        .get("deltaq_explanations_json", "deltaQ_explanations.json")
    )

    # Resolve output directory
    # If out_dir is absolute or starts with "artifacts/", use it directly
    # Otherwise, join with artifacts_dir
    out_dir_str = distill_cfg.get("out_dir", "distill")
    if Path(out_dir_str).is_absolute() or out_dir_str.startswith("artifacts/"):
        out_dir = Path(out_dir_str)
    else:
        artifacts_base = Path(config.get("paths", {}).get("artifacts_dir", "artifacts"))
        out_dir = artifacts_base / out_dir_str
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading dataset from %s", dataset_path)
    dataset = load_npz(dataset_path)
    splits = load_json(splits_path)
    clean_df = load_parquet(clean_log_path)

    # Build distillation dataset
    n_total = int(distill_cfg.get("sample", {}).get("n_states", 2000))
    n_common = int(n_total * 0.7)
    n_high_impact = int(n_total * 0.3)

    logger.info(
        "Building distillation dataset: %d common + %d high-impact", n_common, n_high_impact
    )
    selected_indices, selection_metadata = build_distillation_dataset(
        dataset,
        splits,
        deltaq_path if deltaq_path.exists() else None,
        n_common,
        n_high_impact,
        seed,
        out_dir / "distill_selection.json",
    )

    # Track which indices are high-impact for later evaluation
    high_impact_indices = set()
    if deltaq_path and deltaq_path.exists():
        try:
            hi_idx_array = select_high_impact_states(
                deltaq_path, dataset, n_high_impact * 10
            )  # Get more candidates
            high_impact_indices = set(hi_idx_array.tolist())
        except Exception as e:
            logger.warning("Could not load high-impact indices for evaluation: %s", e)

    # Extract tabular features
    logger.info("Extracting tabular features for %d states...", len(selected_indices))
    X_distill, feature_names = extract_tabular_features(dataset, clean_df, selected_indices, config)

    # Load teacher model
    logger.info("Loading teacher Q-network from %s", ckpt_path)
    q_net = AgentFactory.load(ckpt_path, dataset_path, vocab_path, config, device)

    # Generate teacher labels
    logger.info("Generating teacher labels...")
    y_action, y_qstar, y_margin = generate_teacher_labels(q_net, dataset, selected_indices, device)

    # Train/test split
    train_indices, test_indices = train_test_split(
        np.arange(len(X_distill)),
        test_size=0.3,
        stratify=y_action,
        random_state=seed,
    )
    X_train = X_distill[train_indices]
    X_test = X_distill[test_indices]
    y_train = y_action[train_indices]
    y_test = y_action[test_indices]
    y_margin_test = y_margin[test_indices]

    # Build high-impact mask for test set
    test_selected_indices = selected_indices[test_indices]
    high_impact_mask_test = np.array(
        [idx in high_impact_indices for idx in test_selected_indices], dtype=bool
    )

    # Train surrogate
    surrogate_cfg = distill_cfg.get("surrogate", {})
    max_depth = int(surrogate_cfg.get("max_depth", 5))
    min_samples_leaf = int(surrogate_cfg.get("min_samples_leaf", 50))
    min_samples_split = int(surrogate_cfg.get("min_samples_split", 100))

    logger.info("Training surrogate (max_depth=%d)...", max_depth)
    surrogate, fidelity_metrics = train_surrogate(
        X_train,
        y_train,
        X_test,
        y_test,
        y_margin_test,
        high_impact_mask=high_impact_mask_test,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        random_state=seed,
    )

    logger.info("Fidelity metrics: %s", fidelity_metrics)

    # Save artifacts
    action_names = (
        config.get("mdp", {})
        .get("actions", {})
        .get("id2name", ["do_nothing", "contact_headquarters"])
    )

    # Save tree.pkl
    import pickle

    tree_pkl_path = out_dir / distill_cfg.get("outputs", {}).get("tree_pkl", "tree.pkl")
    ckpt_hash = fingerprint_data([ckpt_path], use_dvc=False)
    dataset_hash = fingerprint_data([dataset_path], use_dvc=False)

    with open(tree_pkl_path, "wb") as f:
        pickle.dump(
            {
                "model": surrogate,
                "feature_names": feature_names,
                "action_names": action_names,
                "fidelity_metrics": fidelity_metrics,
                "metadata": {
                    "ckpt_hash": ckpt_hash,
                    "dataset_hash": dataset_hash,
                    "n_train": len(X_train),
                    "n_test": len(X_test),
                    "timestamp": datetime.now().isoformat(),
                },
            },
            f,
        )

    logger.info("Saved tree to %s", tree_pkl_path)

    # Save fidelity metrics
    fidelity_path = out_dir / "fidelity_metrics.json"
    save_json(fidelity_metrics, fidelity_path)

    return {
        "tree_pkl": tree_pkl_path,
        "fidelity_metrics": fidelity_path,
        "selection": out_dir / "distill_selection.json",
    }
