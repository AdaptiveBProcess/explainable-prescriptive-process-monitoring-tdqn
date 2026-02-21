"""Main orchestrator for XAI: risk, deltaQ, and policy summary artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from xppm.rl.factory import AgentFactory
from xppm.rl.train_tdqn import load_dataset_with_splits
from xppm.utils.io import fingerprint_data, load_json, save_json, save_npz
from xppm.utils.logging import get_logger
from xppm.utils.seed import set_seed
from xppm.xai.attributions import compute_attributions
from xppm.xai.policy_summary import summarize_policy

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_q_network(
    ckpt_path: str | Path,
    npz_path: str | Path,
    vocab_path: str | Path,
    config: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    """Load the trained RL agent via :class:`~xppm.rl.factory.AgentFactory`."""
    return AgentFactory.load(ckpt_path, npz_path, vocab_path, config, device)


def _select_cases(
    case_ptrs: np.ndarray,
    rewards: np.ndarray,
    n_cases: int,
    strategy: str,
    seed: int,
    by: str | None = None,
) -> np.ndarray:
    """Select case IDs for explanation (reproducible).

    Returns:
        selected_case_ids: sorted array of unique case IDs
    """
    unique_ids = np.unique(case_ptrs)
    rng = np.random.default_rng(seed)

    n_cases = min(n_cases, len(unique_ids))

    if strategy == "random":
        selected = rng.choice(unique_ids, size=n_cases, replace=False)
    elif strategy == "stratified":
        # Compute per-case aggregate reward for stratification
        case_rewards = {}
        for cid in unique_ids:
            mask = case_ptrs == cid
            case_rewards[cid] = float(rewards[mask].sum())

        reward_vals = np.array([case_rewards[cid] for cid in unique_ids])
        # Bin into quartiles
        n_bins = min(4, len(unique_ids))
        bins = np.percentile(reward_vals, np.linspace(0, 100, n_bins + 1))
        bin_labels = np.digitize(reward_vals, bins[1:-1])

        selected_list: list[int] = []
        for b in range(n_bins):
            ids_in_bin = unique_ids[bin_labels == b]
            n_from_bin = max(1, int(n_cases * len(ids_in_bin) / len(unique_ids)))
            n_from_bin = min(n_from_bin, len(ids_in_bin))
            chosen = rng.choice(ids_in_bin, size=n_from_bin, replace=False)
            selected_list.extend(chosen.tolist())

        selected = np.array(selected_list[:n_cases])
    else:
        raise ValueError(f"Unknown selection strategy: {strategy}")

    return np.sort(selected)


def _select_transitions(
    case_ptrs: np.ndarray,
    t_ptrs: np.ndarray,
    selected_case_ids: np.ndarray,
    k_times: str | int,
) -> np.ndarray:
    """Select transition indices for selected cases.

    Returns:
        indices: sorted array of transition indices
    """
    selected_set = set(selected_case_ids.tolist())
    indices: list[int] = []

    if k_times == "all":
        for i, cid in enumerate(case_ptrs):
            if cid in selected_set:
                indices.append(i)
    elif k_times == "last":
        # For each case, pick the transition with max t
        case_last: dict[int, tuple[int, int]] = {}  # cid -> (max_t, idx)
        for i, cid in enumerate(case_ptrs):
            if cid in selected_set:
                t = int(t_ptrs[i])
                if cid not in case_last or t > case_last[cid][0]:
                    case_last[cid] = (t, i)
        indices = [v[1] for v in case_last.values()]
    else:
        # k_times is an integer: last k prefixes per case
        k = int(k_times)
        case_transitions: dict[int, list[tuple[int, int]]] = {}
        for i, cid in enumerate(case_ptrs):
            if cid in selected_set:
                t = int(t_ptrs[i])
                case_transitions.setdefault(int(cid), []).append((t, i))
        for cid, trans in case_transitions.items():
            trans.sort(key=lambda x: x[0], reverse=True)
            indices.extend(idx for _, idx in trans[:k])

    return np.sort(np.array(indices, dtype=np.int64))


def _resolve_contrast_action(
    config: dict[str, Any],
    action_names: list[str],
    valid_actions: np.ndarray | None = None,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Resolve the contrast action ID from config, with validation and fallback tracking.

    Args:
        config: xai config section
        action_names: list of action names
        valid_actions: (n_states, n_actions) action masks (optional, for validation)

    Returns:
        (contrast_action_id, contrast_valid_mask, fallback_used_mask)
        - contrast_valid_mask: (n_states,) bool, True if contrast action is valid
        - fallback_used_mask: (n_states,) bool, True if fallback was used
    """
    intervention_cfg = config.get("methods", {}).get("intervention", {}).get("contrast", {})

    # Priority 1: explicit action_id
    if "action_id" in intervention_cfg:
        contrast_id = int(intervention_cfg["action_id"])
    else:
        # Priority 2: name lookup
        baseline_name = intervention_cfg.get("baseline_action", "NOOP")
        noop_names = {"NOOP", "noop", "do_nothing", "no_op"}
        if baseline_name in noop_names:
            contrast_id = 0
        elif baseline_name in action_names:
            contrast_id = action_names.index(baseline_name)
        else:
            contrast_id = 0  # Fallback

    # Validate and track fallback usage
    if valid_actions is not None:
        contrast_valid = valid_actions[:, contrast_id] > 0
        fallback_used = ~contrast_valid

        # Note: We keep the original contrast_id even if invalid in some states.
        # The actual contrast used in compute_attributions will handle masking.
        # This function just tracks which states need fallback.
        return contrast_id, contrast_valid, fallback_used
    else:
        # No validation available, return dummy masks
        return contrast_id, np.array([True]), np.array([False])


def _build_metadata(
    config: dict[str, Any],
    config_hash: str | None,
    dataset_path: str,
    ckpt_path: str,
    seed: int,
    n_cases: int,
    n_selected: int,
    method: str,
    baseline: str,
    n_steps_ig: int,
    global_top_tokens_risk: dict[int, tuple[int, float]] | None = None,
    global_top_tokens_dq: dict[int, tuple[int, float]] | None = None,
    risk_completeness: dict[str, Any] | None = None,
    dq_completeness: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build metadata dict with hashes for reproducibility."""
    dataset_hash = fingerprint_data([dataset_path], use_dvc=False)
    metadata = {
        "config_hash": config_hash,
        "dataset_hash": dataset_hash,
        "ckpt_path": str(ckpt_path),
        "split": config.get("split", "test"),
        "seed": seed,
        "n_cases": n_cases,
        "n_transitions_selected": n_selected,
        "attribution_method": method,
        "baseline": baseline,
        "n_steps_ig": n_steps_ig,
    }
    if global_top_tokens_risk is not None:
        metadata["top_tokens_risk"] = [
            {"token_id": tid, "frequency": freq, "median_importance": med}
            for tid, (freq, med) in list(global_top_tokens_risk.items())[:10]
        ]
    if global_top_tokens_dq is not None:
        metadata["top_tokens_deltaq"] = [
            {"token_id": tid, "frequency": freq, "median_importance": med}
            for tid, (freq, med) in list(global_top_tokens_dq.items())[:10]
        ]
    # Add IG completeness stats
    if risk_completeness:
        metadata["ig_completeness_risk"] = risk_completeness
    if dq_completeness:
        metadata["ig_completeness_deltaq"] = dq_completeness
    return metadata


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def explain_policy(
    config: dict[str, Any],
    config_hash: str | None = None,
) -> dict[str, Path]:
    """Generate all XAI artifacts: risk, deltaQ, policy summary.

    Args:
        config: full config dict (raw from yaml)
        config_hash: optional config hash for metadata

    Returns:
        dict mapping artifact name to saved file path
    """
    xai_cfg = config.get("xai", {})

    # --- 1. Setup ---
    seed = int(xai_cfg.get("seed", config.get("repro", {}).get("seed", 42)))
    set_seed(seed, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s, seed: %d", device, seed)

    # Resolve paths
    ckpt_path = xai_cfg.get("checkpoint_path", "artifacts/checkpoints/Q_theta.ckpt")
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
    out_dir = Path(config.get("paths", {}).get("artifacts_dir", "artifacts")) / xai_cfg.get(
        "out_dir", "xai"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    action_names: list[str] = (
        config.get("mdp", {})
        .get("actions", {})
        .get("id2name", ["do_nothing", "contact_headquarters"])
    )
    outputs_cfg = xai_cfg.get("outputs", {})

    # --- 2. Load model and TEST data ---
    logger.info("Loading Q-network from %s", ckpt_path)
    q_net = _load_q_network(ckpt_path, dataset_path, vocab_path, config, device)
    vocab = load_json(vocab_path)
    id2token_raw = vocab.get("id2token", [])

    def _token_name(token_id: int) -> str:
        if isinstance(id2token_raw, list):
            if 0 <= token_id < len(id2token_raw):
                return str(id2token_raw[token_id])
            return f"UNK_{token_id}"
        return id2token_raw.get(str(token_id), f"UNK_{token_id}")

    logger.info("Loading TEST split data...")
    test_data = load_dataset_with_splits(dataset_path, splits_path, "test")
    s_test = test_data["s"]
    sm_test = test_data["s_mask"]
    r_test = test_data["r"]
    va_test = test_data["valid_actions"]
    cp_test = test_data["case_ptr"]
    tp_test = test_data.get("t_ptr", np.arange(len(s_test)))

    # --- 3. Select cases and transitions ---
    n_cases = int(xai_cfg.get("n_cases", 200))
    sel_cfg = xai_cfg.get("selection", {})
    strategy = sel_cfg.get("strategy", "random")
    k_times = sel_cfg.get("k_times_per_case", "last")

    selected_case_ids = _select_cases(
        cp_test,
        r_test,
        n_cases,
        strategy,
        seed,
        by=sel_cfg.get("by"),
    )
    selected_indices = _select_transitions(cp_test, tp_test, selected_case_ids, k_times)
    n_selected = len(selected_indices)

    # Pre-validate action masks
    va_selected = va_test[selected_indices]
    valid_count = va_selected.sum(axis=1)
    if (valid_count == 0).any():
        n_bad = int((valid_count == 0).sum())
        raise ValueError(
            f"{n_bad} selected transitions have NO valid actions. "
            "Check action mask construction."
        )
    logger.info(
        "Selected %d transitions from %d cases (strategy=%s, k=%s)",
        n_selected,
        len(selected_case_ids),
        strategy,
        k_times,
    )

    # Extract selected arrays
    s_sel = s_test[selected_indices]
    sm_sel = sm_test[selected_indices]
    va_sel = va_test[selected_indices]
    cp_sel = cp_test[selected_indices]
    tp_sel = tp_test[selected_indices]

    # --- 4. Risk explanations: V(s) attributions ---
    risk_cfg = xai_cfg.get("methods", {}).get("risk", {})
    attr_method = risk_cfg.get("attribution", "integrated_gradients")
    baseline = risk_cfg.get("baseline", "pad")
    n_steps_ig = int(risk_cfg.get("n_steps_ig", 32))
    top_k_risk = int(risk_cfg.get("top_k", 10))

    logger.info(
        "Computing RISK attributions (Q(s,a*), method=%s, baseline=%s)...",
        attr_method,
        baseline,
    )
    risk_results = compute_attributions(
        q_net=q_net,
        states=s_sel,
        state_masks=sm_sel,
        valid_actions=va_sel,
        config=xai_cfg,
        device=device,
        target="V",
    )

    # --- 5. DeltaQ explanations ---
    contrast_action_id, contrast_valid_mask, fallback_used_mask = _resolve_contrast_action(
        xai_cfg, action_names, va_sel
    )
    dq_cfg = xai_cfg.get("methods", {}).get("intervention", {}).get("delta_q", {})
    top_k_dq = int(dq_cfg.get("top_k", 10))

    logger.info(
        "Computing DELTAQ attributions (contrast_action=%s [id=%d], " "valid in %d/%d states)...",
        action_names[contrast_action_id],
        contrast_action_id,
        int(contrast_valid_mask.sum()),
        len(contrast_valid_mask),
    )
    dq_results = compute_attributions(
        q_net=q_net,
        states=s_sel,
        state_masks=sm_sel,
        valid_actions=va_sel,
        config=xai_cfg,
        device=device,
        target="deltaQ",
        contrast_action_id=contrast_action_id,
    )

    # --- 6. Policy summary (on ALL test transitions) ---
    logger.info("Generating policy summary on all %d test transitions...", len(s_test))
    summary = summarize_policy(
        q_net=q_net,
        states=s_test,
        state_masks=sm_test,
        valid_actions=va_test,
        case_ptrs=cp_test,
        t_ptrs=tp_test,
        action_names=action_names,
        config=xai_cfg,
        device=device,
    )

    # --- 7. Compute global top tokens (before building metadata) ---
    # Helper: compute top tokens with frequency-based ranking across all cases
    def _compute_top_tokens_robust(
        token_importance_all: np.ndarray,
        states_all: np.ndarray,
        top_k: int,
    ) -> dict[int, tuple[int, float]]:
        """Compute top tokens by frequency in top-k across all cases + median importance."""
        n_items, max_len = token_importance_all.shape
        token_scores: dict[int, tuple[list[float], int]] = {}  # token_id -> (scores, count)

        for i in range(n_items):
            token_imp = token_importance_all[i]
            top_indices = np.argsort(token_imp)[::-1][:top_k]
            for idx in top_indices:
                if token_imp[idx] > 0:
                    token_id = int(states_all[i, idx])
                    if token_id not in token_scores:
                        token_scores[token_id] = ([], 0)
                    token_scores[token_id][0].append(float(token_imp[idx]))
                    token_scores[token_id] = (
                        token_scores[token_id][0],
                        token_scores[token_id][1] + 1,
                    )

        # Rank by frequency first, then median importance
        token_ranking = []
        for token_id, (scores, freq) in token_scores.items():
            median_imp = float(np.median(scores))
            token_ranking.append((token_id, freq, median_imp))

        token_ranking.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return {tid: (freq, med) for tid, freq, med in token_ranking[:top_k]}

    global_top_tokens_risk = _compute_top_tokens_robust(
        risk_results["token_importance"], s_sel, top_k_risk
    )
    global_top_tokens_dq = _compute_top_tokens_robust(
        dq_results["token_importance"], s_sel, top_k_dq
    )

    # --- 8. Build metadata and save artifacts ---
    # Extract completeness stats from results
    risk_completeness = risk_results.get("ig_completeness", {})
    dq_completeness = dq_results.get("ig_completeness", {})

    metadata = _build_metadata(
        config=xai_cfg,
        config_hash=config_hash,
        dataset_path=str(dataset_path),
        ckpt_path=str(ckpt_path),
        seed=seed,
        n_cases=n_cases,
        n_selected=n_selected,
        method=attr_method,
        baseline=baseline,
        n_steps_ig=n_steps_ig,
        global_top_tokens_risk=global_top_tokens_risk,
        global_top_tokens_dq=global_top_tokens_dq,
        risk_completeness=risk_completeness,
        dq_completeness=dq_completeness,
    )

    output_paths: dict[str, Path] = {}

    # 7a. risk_explanations.json
    risk_items = []
    for i in range(n_selected):
        token_imp = risk_results["token_importance"][i]
        top_indices = np.argsort(token_imp)[::-1][:top_k_risk]
        q_star_val = float(risk_results["v_s"][i])  # V = Q(s,a*)
        risk_items.append(
            {
                "case_id": int(cp_sel[i]),
                "t": int(tp_sel[i]),
                "transition_idx": int(selected_indices[i]),  # Index in test split
                "a_star": int(risk_results["a_star"][i]),
                "a_star_name": action_names[int(risk_results["a_star"][i])],
                "V": q_star_val,
                "q_star": q_star_val,  # Explicit q_star for consistency with deltaQ
                "q_values": [float(x) for x in risk_results["q_values"][i]],
                "top_tokens": [
                    {
                        "position": int(idx),
                        "token_id": int(s_sel[i, idx]),
                        "token_name": _token_name(int(s_sel[i, idx])),
                        "importance": float(token_imp[idx]),
                        "global_frequency": global_top_tokens_risk.get(
                            int(s_sel[i, idx]), (0, 0.0)
                        )[0],
                        "global_median_importance": global_top_tokens_risk.get(
                            int(s_sel[i, idx]), (0, 0.0)
                        )[1],
                    }
                    for idx in top_indices
                    if token_imp[idx] > 0
                ],
            }
        )
    risk_path = out_dir / outputs_cfg.get("risk_explanations_json", "risk_explanations.json")
    save_json({"metadata": metadata, "items": risk_items}, risk_path)
    output_paths["risk"] = risk_path
    logger.info("Saved risk explanations to %s", risk_path)

    # 7b. deltaQ_explanations.json
    dq_items = []
    for i in range(n_selected):
        token_imp = dq_results["token_importance"][i]
        top_indices = np.argsort(token_imp)[::-1][:top_k_dq]
        dq_items.append(
            {
                "case_id": int(cp_sel[i]),
                "t": int(tp_sel[i]),
                "transition_idx": int(selected_indices[i]),  # Index in test split
                "a_star": int(dq_results["a_star"][i]),
                "a_star_name": action_names[int(dq_results["a_star"][i])],
                "a_contrast": contrast_action_id,
                "a_contrast_name": action_names[contrast_action_id],
                "contrast_valid": bool(contrast_valid_mask[i]),
                "contrast_fallback_used": bool(fallback_used_mask[i]),
                "q_star": float(dq_results["q_star"][i]),
                "q_contrast": float(dq_results["q_contrast"][i]),
                "delta_q": float(dq_results["delta_q"][i]),
                "q_values": [float(x) for x in dq_results["q_values"][i]],
                "top_drivers": [
                    {
                        "position": int(idx),
                        "token_id": int(s_sel[i, idx]),
                        "token_name": _token_name(int(s_sel[i, idx])),
                        "importance": float(token_imp[idx]),
                        "global_frequency": global_top_tokens_dq.get(int(s_sel[i, idx]), (0, 0.0))[
                            0
                        ],
                        "global_median_importance": global_top_tokens_dq.get(
                            int(s_sel[i, idx]), (0, 0.0)
                        )[1],
                    }
                    for idx in top_indices
                    if token_imp[idx] > 0
                ],
            }
        )
    dq_path = out_dir / outputs_cfg.get("deltaq_explanations_json", "deltaQ_explanations.json")
    save_json({"metadata": metadata, "items": dq_items}, dq_path)
    output_paths["deltaQ"] = dq_path
    logger.info("Saved deltaQ explanations to %s", dq_path)

    # 7c. ig_grad_attributions.npz
    attr_path = out_dir / outputs_cfg.get("attributions_npy", "ig_grad_attributions.npz")
    save_npz(
        attr_path,
        risk_attr=risk_results["attributions_emb"],
        deltaq_attr=dq_results["attributions_emb"],
        token_importance_risk=risk_results["token_importance"],
        token_importance_deltaq=dq_results["token_importance"],
    )
    output_paths["attributions"] = attr_path
    logger.info("Saved raw attributions to %s", attr_path)

    # 7d. policy_summary.json
    summary_path = out_dir / outputs_cfg.get("policy_summary_json", "policy_summary.json")
    save_json({**summary, "metadata": metadata}, summary_path)
    output_paths["policy_summary"] = summary_path
    logger.info("Saved policy summary to %s", summary_path)

    # 7e. explanations_selection.json
    selection_info: dict[str, Any] = {
        "seed": seed,
        "split": xai_cfg.get("split", "test"),
        "strategy": strategy,
        "n_cases": n_cases,
        "k_times_per_case": str(k_times),
        "n_transitions_selected": n_selected,
        "selected_case_ids": selected_case_ids.tolist(),
        "selected_transitions": [
            {
                "case_id": int(cp_sel[i]),
                "t": int(tp_sel[i]),
                "transition_idx": int(selected_indices[i]),
            }
            for i in range(n_selected)
        ],
    }
    sel_path = out_dir / outputs_cfg.get("selection_json", "explanations_selection.json")
    save_json(selection_info, sel_path)
    output_paths["selection"] = sel_path
    logger.info("Saved selection indices to %s", sel_path)

    logger.info("All XAI artifacts saved to %s", out_dir)
    return output_paths
