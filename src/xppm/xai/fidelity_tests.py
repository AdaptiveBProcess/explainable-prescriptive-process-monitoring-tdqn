"""Fidelity tests for XAI explanations: Q-drop, action-flip, rank-consistency.

Implements three fidelity tests as per plan in 3-2-setup.md:
1. Q-drop: Measures if removing top-k important tokens causes larger Q-value drops
   than random removal
2. Action-flip: Measures if removing top-k tokens causes action changes faster
   than random removal
3. Rank-consistency: Spearman/Kendall correlation between Q-based rankings
   and OPE proxy rankings
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau, spearmanr

from xppm.rl.factory import load_q_network
from xppm.rl.models.masking import apply_action_mask
from xppm.rl.train_tdqn import TransformerQNetwork, load_dataset_with_splits
from xppm.utils.io import fingerprint_data, get_git_commit, load_json
from xppm.utils.logging import ensure_dir, get_logger
from xppm.utils.seed import set_seed

logger = get_logger(__name__)


def _load_q_network(
    ckpt_path: str | Path,
    npz_path: str | Path,
    vocab_path: str | Path,
    config: dict[str, Any],
    device: torch.device,
) -> TransformerQNetwork:
    """Load TransformerQNetwork from checkpoint. Delegates to the canonical loader."""
    return load_q_network(ckpt_path, npz_path, vocab_path, config, device)


def _debug_perturbation(
    item_idx: int,
    states_orig: np.ndarray,
    states_pert: np.ndarray,
    state_masks: np.ndarray,
    top_positions: list[int],
    pad_id: int,
    vocab: dict[str, Any] | None = None,
) -> None:
    """Debug helper to verify perturbation is working correctly."""
    logger.info("=" * 60)
    logger.info(f"DEBUG Perturbation - Item {item_idx}")
    logger.info("=" * 60)

    # Check positions
    logger.info(f"Top-k positions to remove: {top_positions}")
    logger.info(f"PAD token ID: {pad_id}")

    # Check original vs perturbed
    seq_orig = states_orig[item_idx]
    seq_pert = states_pert[item_idx]
    mask = state_masks[item_idx]

    n_nonpad = int(mask.sum())
    logger.info(f"Non-pad length: {n_nonpad}")

    # Check if positions are within bounds
    valid_positions = [p for p in top_positions if 0 <= p < len(seq_orig)]
    logger.info(f"Valid positions (within bounds): {len(valid_positions)}/{len(top_positions)}")

    # Check tokens at those positions
    if len(valid_positions) > 0:
        tokens_orig_at_pos = [int(seq_orig[p]) for p in valid_positions[:5]]
        tokens_pert_at_pos = [int(seq_pert[p]) for p in valid_positions[:5]]
        logger.info(f"Original tokens at top positions (first 5): {tokens_orig_at_pos}")
        logger.info(f"Perturbed tokens at top positions (first 5): {tokens_pert_at_pos}")
        logger.info(f"Expected PAD: {pad_id}")

        # Check if they changed
        changed = [
            tokens_orig_at_pos[i] != tokens_pert_at_pos[i] for i in range(len(tokens_orig_at_pos))
        ]
        logger.info(f"Tokens changed: {sum(changed)}/{len(changed)}")

        if vocab and "id2token" in vocab:
            id2token = vocab["id2token"]
            logger.info(
                f"Original token names: {[id2token.get(t, f'UNK_{t}') for t in tokens_orig_at_pos]}"
            )
            perturbed_names = [id2token.get(t, f"UNK_{t}") for t in tokens_pert_at_pos]
            logger.info(f"Perturbed token names: {perturbed_names}")

    # Check mask changes
    mask_orig = mask.copy()
    mask_pert = (
        state_masks[item_idx].copy() if len(state_masks.shape) > 1 else state_masks[item_idx]
    )
    mask_changed = not np.array_equal(mask_orig, mask_pert)
    logger.info(f"Mask changed: {mask_changed}")

    logger.info("=" * 60)


def _perturb_states_mask_topk(
    states: np.ndarray,
    state_masks: np.ndarray,
    top_k_indices: list[list[int]],
    pad_id: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Perturb states by masking top-k important tokens to PAD.

    Args:
        states: (N, max_len) token IDs
        state_masks: (N, max_len) 1=real, 0=pad
        top_k_indices: List of N lists, each containing k token positions to mask
        pad_id: PAD token ID (typically 0)

    Returns:
        perturbed_states: (N, max_len) with top-k tokens replaced by pad_id
        perturbed_masks: (N, max_len) updated masks (same as input, tokens become pad)
    """
    perturbed_states = states.copy()
    perturbed_masks = state_masks.copy()
    max_len = states.shape[1]

    for i, positions in enumerate(top_k_indices):
        valid_pos = [p for p in positions if 0 <= p < max_len]
        if valid_pos:
            perturbed_states[i, valid_pos] = pad_id
            perturbed_masks[i, valid_pos] = 0

    return perturbed_states, perturbed_masks


def _validate_perturbation(
    q_net: TransformerQNetwork,
    states: np.ndarray,
    state_masks: np.ndarray,
    device: torch.device,
) -> bool:
    """Validate that perturbation doesn't break forward pass.

    Args:
        q_net: Q-network
        states: (N, max_len) token IDs
        state_masks: (N, max_len) masks
        device: torch device

    Returns:
        True if forward pass succeeds, False otherwise
    """
    try:
        # Test with a single perturbed case (50% tokens masked)
        test_states = states[:1].copy()
        test_masks = state_masks[:1].copy()
        n_nonpad = int(test_masks[0].sum())
        k = max(1, n_nonpad // 2)

        # Mask k random tokens
        nonpad_positions = np.where(test_masks[0] > 0)[0]
        if len(nonpad_positions) > 0:
            mask_positions = np.random.choice(
                nonpad_positions, size=min(k, len(nonpad_positions)), replace=False
            )
            for pos in mask_positions:
                test_states[0, pos] = 0
                test_masks[0, pos] = 0

        s_t = torch.from_numpy(test_states).long().to(device)
        m_t = torch.from_numpy(test_masks).float().to(device)

        with torch.no_grad():
            _ = q_net(s_t, m_t)

        return True
    except Exception as e:
        logger.error("Perturbation validation failed: %s", e)
        return False


def _compute_q_values(
    q_net: TransformerQNetwork,
    states: np.ndarray,
    state_masks: np.ndarray,
    valid_actions: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Q-values, V(s), and a* for states.

    Returns:
        q_values: (N, n_actions) Q(s,a) for all actions
        v_s: (N,) V(s) = max_a Q_masked(s,a)
        a_star: (N,) argmax action
    """
    n = states.shape[0]
    q_list = []
    v_list = []
    a_list = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        s_b = torch.from_numpy(states[start:end]).long().to(device)
        m_b = torch.from_numpy(state_masks[start:end]).float().to(device)
        va_b = torch.from_numpy(valid_actions[start:end]).float().to(device)

        with torch.no_grad():
            q_vals = q_net(s_b, m_b)
            q_masked = apply_action_mask(q_vals, va_b)
            v_s, _ = torch.max(q_masked, dim=-1)
            a_star = q_masked.argmax(dim=-1)

        q_list.append(q_vals.cpu().numpy())
        v_list.append(v_s.cpu().numpy())
        a_list.append(a_star.cpu().numpy())

    return (
        np.concatenate(q_list, axis=0),
        np.concatenate(v_list, axis=0),
        np.concatenate(a_list, axis=0),
    )


def _test_q_drop(
    q_net: TransformerQNetwork,
    states: np.ndarray,
    state_masks: np.ndarray,
    valid_actions: np.ndarray,
    explanations: dict[str, Any],
    p_remove_list: list[float],
    n_random: int,
    seed: int,
    device: torch.device,
    target: str = "q_star",
    debug: bool = False,
    vocab: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Test 1: Q-drop - measure drop in Q when removing top-k vs random tokens.

    Args:
        q_net: Q-network
        states: (N, max_len) token IDs
        state_masks: (N, max_len) masks
        valid_actions: (N, n_actions) action masks
        explanations: dict with 'items' list containing 'top_tokens' per transition
        p_remove_list: List of removal percentages [0.1, 0.2, 0.3, 0.5]
        n_random: Number of random repetitions
        seed: Random seed
        target: "q_star" or "delta_q"

    Returns:
        List of dicts with metrics for each p_remove
    """
    logger.info("Running Q-drop test (target=%s)...", target)
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_items = len(explanations.get("items", []))
    if n_items == 0:
        raise ValueError("No explanation items found")

    # Get original Q-values
    q_vals_orig, v_s_orig, a_star_orig = _compute_q_values(
        q_net, states, state_masks, valid_actions, device
    )

    # Extract target values and top tokens from explanations
    # Note: states and state_masks should already be aligned with explanations["items"]
    target_orig = []
    top_tokens_list = []
    bottom_tokens_list = []
    action_pairs = []  # (a_star, a_contrast) per item, needed to recompute delta_q under masking

    for idx, item in enumerate(explanations["items"]):
        if idx >= len(states):
            logger.warning("More explanation items than states, stopping at %d", len(states))
            break

        if target == "q_star":
            target_val = item.get("q_star") or item.get("V")
        elif target == "delta_q":
            target_val = item.get("delta_q")
        else:
            raise ValueError(f"Unknown target: {target}")

        if target_val is None:
            logger.warning("Missing target value for item %d, skipping", idx)
            continue

        if target == "delta_q":
            a_star_i = item.get("a_star")
            a_con_i = item.get("a_contrast")
            if a_star_i is None or a_con_i is None:
                logger.warning("Missing a_star/a_contrast for item %d, skipping", idx)
                continue
            action_pairs.append((int(a_star_i), int(a_con_i)))

        target_orig.append(target_val)
        # For delta_q, use top_drivers/bottom_drivers; for q_star, use top_tokens/bottom_tokens
        if target == "delta_q":
            top_tokens = item.get("top_drivers", [])
            bottom_tokens = item.get("bottom_drivers", [])
        else:
            top_tokens = item.get("top_tokens", [])
            bottom_tokens = item.get("bottom_tokens", [])
        max_len = states.shape[1]
        top_positions = [
            t.get("position")
            for t in top_tokens
            if isinstance(t, dict)
            and "position" in t
            and t.get("position") is not None
            and 0 <= t.get("position") < max_len
            and state_masks[idx][t.get("position")] > 0
        ]
        bottom_positions = [
            t.get("position")
            for t in bottom_tokens
            if isinstance(t, dict)
            and "position" in t
            and t.get("position") is not None
            and 0 <= t.get("position") < max_len
            and state_masks[idx][t.get("position")] > 0
        ]
        top_tokens_list.append(top_positions)
        bottom_tokens_list.append(bottom_positions)

    target_orig = np.array(target_orig)
    n_valid = len(target_orig)
    action_pairs_arr = np.array(action_pairs) if target == "delta_q" else None

    def _target_from(q_vals: np.ndarray, v_s: np.ndarray) -> np.ndarray:
        """Perturbed-state target: V(s) for q_star; Q(s,a*)-Q(s,a_contrast) for delta_q,
        holding the original action pair fixed so the margin is recomputed, not proxied by V."""
        if target == "q_star":
            return v_s
        rows = np.arange(q_vals.shape[0])
        return (
            q_vals[rows, action_pairs_arr[: q_vals.shape[0], 0]]
            - q_vals[rows, action_pairs_arr[: q_vals.shape[0], 1]]
        )

    results = []

    for p_remove in p_remove_list:
        logger.info("  Testing p_remove=%.2f...", p_remove)

        # Compute k (number of tokens to remove)
        k_list = []
        top_k_indices_list = []
        for i in range(n_valid):
            n_nonpad = int(state_masks[i].sum())
            k = max(1, int(np.ceil(p_remove * n_nonpad)))
            k_list.append(k)

            # Get top-k positions from explanations
            # top_tokens_list[i] already contains valid positions (filtered in extraction)
            # Just take the first k
            top_positions = (
                top_tokens_list[i][:k] if len(top_tokens_list[i]) >= k else top_tokens_list[i]
            )
            top_k_indices_list.append(top_positions)

        # Top-k removal
        states_topk, masks_topk = _perturb_states_mask_topk(
            states[:n_valid], state_masks[:n_valid], top_k_indices_list
        )

        # Debug: verify perturbation for first item
        if debug and len(top_k_indices_list) > 0:
            _debug_perturbation(
                0,
                states[:n_valid],
                states_topk,
                masks_topk,
                top_k_indices_list[0],
                pad_id=0,
                vocab=vocab,
            )

        q_topk, v_s_topk, _ = _compute_q_values(
            q_net, states_topk, masks_topk, valid_actions[:n_valid], device
        )
        target_topk = _target_from(q_topk, v_s_topk)

        # Debug: check Q-values
        if debug:
            logger.info(f"DEBUG Q-drop (p_remove={p_remove}):")
            logger.info(f"  target_orig[0:3]: {target_orig[:3]}")
            logger.info(f"  target_topk[0:3]: {target_topk[:3]}")
            logger.info(f"  drop_topk[0:3]: {target_orig[:3] - target_topk[:3]}")

        drop_topk = target_orig - target_topk
        drop_topk_norm = drop_topk / (
            np.abs(target_orig) + 1e-6
        )  # 1e-6 for better numerical stability

        # Random removal (average over n_random repetitions)
        # Deterministic: seed + item_idx + rep for reproducibility
        drop_rand_list = []
        for r in range(n_random):
            rand_indices_list = []
            for i in range(n_valid):
                # Deterministic seed per item and repetition
                item_seed = seed + i * 1000 + r
                np.random.seed(item_seed)
                n_nonpad = int(state_masks[i].sum())
                k = k_list[i]
                nonpad_positions = np.where(state_masks[i] > 0)[0]
                if len(nonpad_positions) >= k:
                    rand_positions = np.random.choice(
                        nonpad_positions, size=k, replace=False
                    ).tolist()
                else:
                    rand_positions = nonpad_positions.tolist()
                rand_indices_list.append(rand_positions)

            states_rand, masks_rand = _perturb_states_mask_topk(
                states[:n_valid], state_masks[:n_valid], rand_indices_list
            )
            q_rand, v_s_rand, _ = _compute_q_values(
                q_net, states_rand, masks_rand, valid_actions[:n_valid], device
            )
            target_rand = _target_from(q_rand, v_s_rand)

            drop_rand = target_orig - target_rand
            drop_rand_list.append(drop_rand)

        drop_rand_array = np.array(drop_rand_list)  # (n_random, n_valid)
        drop_rand_mean = drop_rand_array.mean(axis=0)
        drop_rand_std = drop_rand_array.std(axis=0)
        drop_rand_norm = drop_rand_mean / (np.abs(target_orig) + 1e-6)

        # Anti-guided removal (least important tokens)
        has_bottom = any(len(b) > 0 for b in bottom_tokens_list)
        if has_bottom:
            bottom_k_indices_list = []
            for i in range(n_valid):
                bottom_positions = (
                    bottom_tokens_list[i][: k_list[i]]
                    if len(bottom_tokens_list[i]) >= k_list[i]
                    else bottom_tokens_list[i]
                )
                bottom_k_indices_list.append(bottom_positions)

            states_anti, masks_anti = _perturb_states_mask_topk(
                states[:n_valid], state_masks[:n_valid], bottom_k_indices_list
            )
            q_anti, v_s_anti, _ = _compute_q_values(
                q_net, states_anti, masks_anti, valid_actions[:n_valid], device
            )
            target_anti = _target_from(q_anti, v_s_anti)
            drop_anti = target_orig - target_anti
            gap_anti = drop_anti - drop_rand_mean  # should be < 0 for faithful attributions
        else:
            drop_anti = None
            gap_anti = None

        # Gap
        gap = drop_topk - drop_rand_mean
        gap_norm = drop_topk_norm - drop_rand_norm

        results.append(
            {
                "test": "q_drop",
                "split": "test",
                "target": target,
                "p_remove": p_remove,
                "metric": "drop_topk",
                "value": float(drop_topk.mean()),
                "n_items": n_valid,
                "seed": seed,
                "baseline_type": "pad",
            }
        )
        results.append(
            {
                "test": "q_drop",
                "split": "test",
                "target": target,
                "p_remove": p_remove,
                "metric": "drop_topk_norm",
                "value": float(drop_topk_norm.mean()),
                "n_items": n_valid,
                "seed": seed,
                "baseline_type": "pad",
            }
        )
        results.append(
            {
                "test": "q_drop",
                "split": "test",
                "target": target,
                "p_remove": p_remove,
                "metric": "drop_rand_mean",
                "value": float(drop_rand_mean.mean()),
                "n_items": n_valid,
                "seed": seed,
                "baseline_type": "pad",
            }
        )
        results.append(
            {
                "test": "q_drop",
                "split": "test",
                "target": target,
                "p_remove": p_remove,
                "metric": "drop_rand_std",
                "value": float(drop_rand_std.mean()),
                "n_items": n_valid,
                "seed": seed,
                "baseline_type": "pad",
            }
        )
        results.append(
            {
                "test": "q_drop",
                "split": "test",
                "target": target,
                "p_remove": p_remove,
                "metric": "gap",
                "value": float(gap.mean()),
                "n_items": n_valid,
                "seed": seed,
                "baseline_type": "pad",
            }
        )
        results.append(
            {
                "test": "q_drop",
                "split": "test",
                "target": target,
                "p_remove": p_remove,
                "metric": "gap_norm",
                "value": float(gap_norm.mean()),
                "n_items": n_valid,
                "seed": seed,
                "baseline_type": "pad",
            }
        )
        if drop_anti is not None:
            results.append(
                {
                    "test": "q_drop",
                    "split": "test",
                    "target": target,
                    "p_remove": p_remove,
                    "metric": "drop_anti",
                    "value": float(drop_anti.mean()),
                    "n_items": n_valid,
                    "seed": seed,
                    "baseline_type": "pad",
                }
            )
            results.append(
                {
                    "test": "q_drop",
                    "split": "test",
                    "target": target,
                    "p_remove": p_remove,
                    "metric": "gap_anti",
                    "value": float(gap_anti.mean()),
                    "n_items": n_valid,
                    "seed": seed,
                    "baseline_type": "pad",
                }
            )

    return results


def _test_action_flip(
    q_net: TransformerQNetwork,
    states: np.ndarray,
    state_masks: np.ndarray,
    valid_actions: np.ndarray,
    explanations: dict[str, Any],
    p_remove_list: list[float],
    n_random: int,
    seed: int,
    device: torch.device,
    debug: bool = False,
    vocab: dict[str, Any] | None = None,
    attr_label: str = "-",
) -> list[dict[str, Any]]:
    """Test 2: Action-flip - measure action changes when removing top-k vs random tokens.

    Args:
        q_net: Q-network
        states: (N, max_len) token IDs
        state_masks: (N, max_len) masks
        valid_actions: (N, n_actions) action masks
        explanations: dict with 'items' list containing 'top_tokens' per transition
        p_remove_list: List of removal percentages
        n_random: Number of random repetitions
        seed: Random seed

    Returns:
        List of dicts with metrics for each p_remove
    """
    logger.info("Running action-flip test...")
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_items = len(explanations.get("items", []))
    if n_items == 0:
        raise ValueError("No explanation items found")

    # Get original actions and Q-values
    q_vals_orig, v_s_orig, a_star_orig = _compute_q_values(
        q_net, states, state_masks, valid_actions, device
    )

    # Debug: check action space
    if debug:
        logger.info("DEBUG Action-flip - Original state:")
        logger.info(f"  Q-values[0]: {q_vals_orig[0]}")
        logger.info(f"  V(s)[0]: {v_s_orig[0]}")
        logger.info(f"  a_star[0]: {a_star_orig[0]}")
        logger.info(f"  valid_actions[0]: {valid_actions[0]}")
        logger.info(f"  Num valid actions[0]: {valid_actions[0].sum()}")

    # Extract top and bottom tokens from explanations (filter PAD positions)
    # Note: states and state_masks should already be aligned with explanations["items"]
    top_tokens_list = []
    bottom_tokens_list = []
    for idx, item in enumerate(explanations["items"]):
        if idx >= len(states):
            logger.warning("More explanation items than states, stopping at %d", len(states))
            break

        # risk explanations carry top_tokens/bottom_tokens; deltaQ explanations carry
        # top_drivers/bottom_drivers — accept either so the flip test can target phi^{Delta Q}
        top_tokens = item.get("top_tokens") or item.get("top_drivers", [])
        bottom_tokens = item.get("bottom_tokens") or item.get("bottom_drivers", [])
        max_len = states.shape[1]
        top_positions = [
            t.get("position")
            for t in top_tokens
            if isinstance(t, dict)
            and "position" in t
            and t.get("position") is not None
            and 0 <= t.get("position") < max_len
            and state_masks[idx][t.get("position")] > 0
        ]
        bottom_positions = [
            t.get("position")
            for t in bottom_tokens
            if isinstance(t, dict)
            and "position" in t
            and t.get("position") is not None
            and 0 <= t.get("position") < max_len
            and state_masks[idx][t.get("position")] > 0
        ]
        top_tokens_list.append(top_positions)
        bottom_tokens_list.append(bottom_positions)

    n_valid = len(top_tokens_list)

    # Identify items where flip is possible (≥2 valid actions)
    flip_possible = valid_actions[:n_valid].sum(axis=1) >= 2
    n_flip_possible = int(flip_possible.sum())
    flip_possible_rate = float(n_flip_possible / n_valid) if n_valid > 0 else 0.0

    logger.info(
        "Items with ≥2 valid actions (flip-possible): %d/%d (%.1f%%)",
        n_flip_possible,
        n_valid,
        100 * flip_possible_rate,
    )

    results = []

    for p_remove in p_remove_list:
        logger.info("  Testing p_remove=%.2f...", p_remove)

        # Compute k and top-k indices
        k_list = []
        top_k_indices_list = []
        for i in range(n_valid):
            n_nonpad = int(state_masks[i].sum())
            k = max(1, int(np.ceil(p_remove * n_nonpad)))
            k_list.append(k)
            top_positions = (
                top_tokens_list[i][:k] if len(top_tokens_list[i]) >= k else top_tokens_list[i]
            )
            top_k_indices_list.append(top_positions)

        # Top-k removal
        states_topk, masks_topk = _perturb_states_mask_topk(
            states[:n_valid], state_masks[:n_valid], top_k_indices_list
        )

        # Debug: verify perturbation for first item
        if debug and len(top_k_indices_list) > 0:
            _debug_perturbation(
                0,
                states[:n_valid],
                states_topk,
                masks_topk,
                top_k_indices_list[0],
                pad_id=0,
                vocab=vocab,
            )

        q_vals_topk, v_s_topk, a_star_topk = _compute_q_values(
            q_net, states_topk, masks_topk, valid_actions[:n_valid], device
        )

        # Debug: check Q-values and actions
        if debug:
            logger.info(f"DEBUG Action-flip (p_remove={p_remove}):")
            logger.info(f"  Q-values orig[0]: {q_vals_orig[0]}")
            logger.info(f"  Q-values pert[0]: {q_vals_topk[0]}")
            logger.info(f"  a_star orig[0]: {a_star_orig[0]}")
            logger.info(f"  a_star pert[0]: {a_star_topk[0]}")
            logger.info(f"  Flip[0]: {a_star_orig[0] != a_star_topk[0]}")

        flip_topk = (a_star_topk != a_star_orig[:n_valid]).astype(float)
        flip_topk_pct = float(flip_topk.mean())

        # Filter to only flip-possible cases
        flip_topk_filtered = flip_topk[flip_possible]
        flip_topk_pct_filtered = float(flip_topk_filtered.mean()) if n_flip_possible > 0 else 0.0

        # Random removal (deterministic: seed + item_idx + rep)
        flip_rand_list = []
        for r in range(n_random):
            rand_indices_list = []
            for i in range(n_valid):
                item_seed = seed + i * 1000 + r
                np.random.seed(item_seed)
                n_nonpad = int(state_masks[i].sum())
                k = k_list[i]
                nonpad_positions = np.where(state_masks[i] > 0)[0]
                if len(nonpad_positions) >= k:
                    rand_positions = np.random.choice(
                        nonpad_positions, size=k, replace=False
                    ).tolist()
                else:
                    rand_positions = nonpad_positions.tolist()
                rand_indices_list.append(rand_positions)

            states_rand, masks_rand = _perturb_states_mask_topk(
                states[:n_valid], state_masks[:n_valid], rand_indices_list
            )
            _, _, a_star_rand = _compute_q_values(
                q_net, states_rand, masks_rand, valid_actions[:n_valid], device
            )

            flip_rand = (a_star_rand != a_star_orig[:n_valid]).astype(float)
            flip_rand_list.append(flip_rand)

        flip_rand_array = np.array(flip_rand_list)  # (n_random, n_valid)
        flip_rand_mean = float(flip_rand_array.mean())
        flip_rand_std = float(flip_rand_array.std())

        # Filter random flips to only flip-possible cases
        flip_rand_filtered = flip_rand_array[:, flip_possible]  # (n_random, n_flip_possible)
        flip_rand_mean_filtered = float(flip_rand_filtered.mean()) if n_flip_possible > 0 else 0.0

        # Anti-guided flip (least important tokens)
        has_bottom = any(len(b) > 0 for b in bottom_tokens_list)
        if has_bottom:
            bottom_k_indices_list = []
            for i in range(n_valid):
                bottom_positions = (
                    bottom_tokens_list[i][: k_list[i]]
                    if len(bottom_tokens_list[i]) >= k_list[i]
                    else bottom_tokens_list[i]
                )
                bottom_k_indices_list.append(bottom_positions)

            states_anti, masks_anti = _perturb_states_mask_topk(
                states[:n_valid], state_masks[:n_valid], bottom_k_indices_list
            )
            _, _, a_star_anti = _compute_q_values(
                q_net, states_anti, masks_anti, valid_actions[:n_valid], device
            )
            flip_anti = (a_star_anti != a_star_orig[:n_valid]).astype(float)
            flip_anti_pct = float(flip_anti.mean())
            flip_anti_filtered = flip_anti[flip_possible]
            flip_anti_pct_filtered = (
                float(flip_anti_filtered.mean()) if n_flip_possible > 0 else 0.0
            )
        else:
            flip_anti_pct = None
            flip_anti_pct_filtered = None

        # Gap (overall and filtered)
        flip_gap = flip_topk_pct - flip_rand_mean
        flip_gap_filtered = flip_topk_pct_filtered - flip_rand_mean_filtered

        results.append(
            {
                "test": "action_flip",
                "split": "test",
                "target": attr_label,
                "p_remove": p_remove,
                "metric": "flip_topk",
                "value": flip_topk_pct,
                "n_items": n_valid,
                "seed": seed,
                "baseline_type": "pad",
            }
        )
        results.append(
            {
                "test": "action_flip",
                "split": "test",
                "target": attr_label,
                "p_remove": p_remove,
                "metric": "flip_rand_mean",
                "value": flip_rand_mean,
                "n_items": n_valid,
                "seed": seed,
                "baseline_type": "pad",
            }
        )
        results.append(
            {
                "test": "action_flip",
                "split": "test",
                "target": attr_label,
                "p_remove": p_remove,
                "metric": "flip_rand_std",
                "value": flip_rand_std,
                "n_items": n_valid,
                "seed": seed,
                "baseline_type": "pad",
            }
        )
        results.append(
            {
                "test": "action_flip",
                "split": "test",
                "target": attr_label,
                "p_remove": p_remove,
                "metric": "flip_gap",
                "value": flip_gap,
                "n_items": n_valid,
                "seed": seed,
                "baseline_type": "pad",
            }
        )

        # Add flip-possible metrics (only report once per p_remove, use first iteration)
        if p_remove == p_remove_list[0]:
            results.append(
                {
                    "test": "action_flip",
                    "split": "test",
                    "target": "-",
                    "p_remove": -1,  # Use -1 to indicate this is a global metric
                    "metric": "flip_possible_rate",
                    "value": flip_possible_rate,
                    "n_items": n_valid,
                    "seed": seed,
                    "baseline_type": "pad",
                }
            )

        # Add filtered metrics (only on flip-possible cases)
        results.append(
            {
                "test": "action_flip",
                "split": "test",
                "target": attr_label,
                "p_remove": p_remove,
                "metric": "flip_topk_on_possible",
                "value": flip_topk_pct_filtered,
                "n_items": n_flip_possible,
                "seed": seed,
                "baseline_type": "pad",
            }
        )
        results.append(
            {
                "test": "action_flip",
                "split": "test",
                "target": attr_label,
                "p_remove": p_remove,
                "metric": "flip_rand_mean_on_possible",
                "value": flip_rand_mean_filtered,
                "n_items": n_flip_possible,
                "seed": seed,
                "baseline_type": "pad",
            }
        )
        results.append(
            {
                "test": "action_flip",
                "split": "test",
                "target": attr_label,
                "p_remove": p_remove,
                "metric": "flip_gap_on_possible",
                "value": flip_gap_filtered,
                "n_items": n_flip_possible,
                "seed": seed,
                "baseline_type": "pad",
            }
        )
        if flip_anti_pct is not None:
            results.append(
                {
                    "test": "action_flip",
                    "split": "test",
                    "target": "-",
                    "p_remove": p_remove,
                    "metric": "flip_anti",
                    "value": flip_anti_pct,
                    "n_items": n_valid,
                    "seed": seed,
                    "baseline_type": "pad",
                }
            )
            results.append(
                {
                    "test": "action_flip",
                    "split": "test",
                    "target": "-",
                    "p_remove": p_remove,
                    "metric": "flip_anti_on_possible",
                    "value": flip_anti_pct_filtered,
                    "n_items": n_flip_possible,
                    "seed": seed,
                    "baseline_type": "pad",
                }
            )

    return results


def _test_rank_consistency(
    explanations: dict[str, Any],
    policy_summary: dict[str, Any] | None,
    ope_path: str | Path | None,
    metrics: list[str],
    seed: int,
) -> list[dict[str, Any]]:
    """Test 3: Rank-consistency - Spearman/Kendall correlation between Q and OPE rankings.

    Uses Option B: ranking by cluster (from policy_summary) with proxy OPE scores.
    Uses clusters directly from policy_summary.json (not filtered by selected items).

    Args:
        explanations: risk_explanations.json or deltaQ_explanations.json
            (not used, kept for API compatibility)
        policy_summary: policy_summary.json with cluster assignments
        ope_path: Path to OPE results (optional, for proxy scores)
        metrics: List of metrics ["spearman", "kendall"]
        seed: Random seed

    Returns:
        List of dicts with correlation metrics
    """
    logger.info("Running rank-consistency test...")

    results = []

    # Option B: Ranking by cluster (using clusters directly from policy_summary)
    if policy_summary is None:
        logger.warning("No policy_summary provided, skipping rank-consistency")
        return results

    clusters = policy_summary.get("clusters", [])
    if len(clusters) < 2:
        logger.warning("Need at least 2 clusters for rank-consistency, found %d", len(clusters))
        return results

    logger.info("Using %d clusters from policy_summary.json", len(clusters))

    # Build rankings directly from cluster-level metrics
    # Use mean_v as Q-score and std_v (or mean_policy_margin) as proxy OPE score
    # NOTE: std_v represents value variability within cluster, which is independent of mean_v
    # and can serve as a proxy for cluster heterogeneity (higher std = more diverse states)
    score_Q = []
    score_OPE_proxy = []
    cluster_ids = []

    for cluster in clusters:
        cluster_id = cluster.get("cluster_id")
        mean_v = cluster.get("mean_v")
        std_v = cluster.get("std_v")
        mean_margin = cluster.get("mean_policy_margin")

        if mean_v is None:
            logger.warning("Cluster %d missing mean_v, skipping", cluster_id)
            continue

        cluster_ids.append(cluster_id)
        score_Q.append(float(mean_v))

        # Priority: mean_policy_margin > std_v > mean_v (fallback)
        # mean_policy_margin: policy confidence (Q(a*) - Q(a2))
        # std_v: value variability (independent of mean_v, represents cluster heterogeneity)
        if mean_margin is not None:
            ope_score = float(mean_margin)
        elif std_v is not None:
            ope_score = float(std_v)
        else:
            logger.warning(
                "Cluster %d missing both mean_policy_margin and std_v, using mean_v as fallback",
                cluster_id,
            )
            ope_score = float(mean_v)
        score_OPE_proxy.append(ope_score)

    if len(score_Q) < 2:
        logger.warning("Need at least 2 clusters with valid scores, found %d", len(score_Q))
        return results

    # Convert to arrays for correlation
    score_Q_array = np.array(score_Q)
    score_OPE_array = np.array(score_OPE_proxy)

    # Compute correlations
    for metric_name in metrics:
        if metric_name == "spearman":
            corr, p_value = spearmanr(score_Q_array, score_OPE_array)
            corr_value = float(corr) if not np.isnan(corr) else 0.0
        elif metric_name == "kendall":
            corr, p_value = kendalltau(score_Q_array, score_OPE_array)
            corr_value = float(corr) if not np.isnan(corr) else 0.0
        else:
            logger.warning("Unknown metric: %s, skipping", metric_name)
            continue

        # Determine OPE proxy description based on what was actually used
        # Check which proxy was used (should be consistent across clusters)
        first_cluster = clusters[0] if clusters else {}
        if first_cluster.get("mean_policy_margin") is not None:
            ope_proxy_desc = "PROXY: mean_policy_margin (Q(a*)-Q(a2))"
        elif first_cluster.get("std_v") is not None:
            ope_proxy_desc = "PROXY: std_v (value variability)"
        else:
            ope_proxy_desc = "PROXY: mean_v (fallback - NOT RECOMMENDED)"

        results.append(
            {
                "test": "rank_consistency",
                "split": "test",
                "level": "cluster",
                "target": "-",
                "p_remove": None,
                "metric": metric_name,
                "value": corr_value,
                "n_items": len(score_Q),
                "n_units": len(score_Q),
                "seed": seed,
                "baseline_type": "-",
                "score_Q_used": "mean_v",
                "score_OPE_used": ope_proxy_desc,
            }
        )

    return results


def run_fidelity_tests(config: dict[str, Any], config_obj: Any = None) -> None:
    """Main entry point for fidelity tests.

    Args:
        config: Full config dict from yaml
    """
    fidelity_cfg = config.get("fidelity", {})
    if not fidelity_cfg.get("enabled", False):
        logger.info("Fidelity tests disabled in config, skipping")
        return

    # Setup
    seed = int(fidelity_cfg.get("seed", config.get("repro", {}).get("seed", 42)))
    set_seed(seed, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s, seed: %d", device, seed)

    # Resolve paths
    paths_cfg = config.get("paths", {})
    artifacts_dir = Path(paths_cfg.get("artifacts_dir", "artifacts"))
    xai_dir = artifacts_dir / config.get("xai", {}).get("out_dir", "xai")

    # Try final/ subdirectory first, then base xai_dir
    # Check if files exist in final/, otherwise use base xai_dir
    xai_final_dir = xai_dir / "final"
    risk_path_final = xai_final_dir / "risk_explanations.json"
    risk_path_base = xai_dir / "risk_explanations.json"

    if risk_path_final.exists():
        logger.info("Using XAI outputs from %s", xai_final_dir)
    elif risk_path_base.exists():
        xai_final_dir = xai_dir
        logger.info(
            "Using XAI outputs from %s (final/ subdirectory exists but files not found there)",
            xai_dir,
        )
    else:
        # Keep final_dir but will raise error below
        logger.warning("Neither %s nor %s found, will check both", risk_path_final, risk_path_base)

    ckpt_path = config.get("xai", {}).get("checkpoint_path", "artifacts/checkpoints/Q_theta.ckpt")
    dataset_path = (
        config.get("mdp", {}).get("output", {}).get("path", "data/processed/D_offline.npz")
    )
    splits_path = str(
        Path(paths_cfg.get("data_processed_dir", "data/processed"))
        / config.get("validation_split", {}).get("out_splits_json", "splits.json")
    )
    vocab_path = (
        config.get("encoding", {})
        .get("output", {})
        .get("vocab_activity_path", "data/interim/vocab_activity.json")
    )

    # Load XAI outputs - try both locations
    risk_explanations_path = xai_final_dir / "risk_explanations.json"
    if not risk_explanations_path.exists():
        risk_explanations_path = xai_dir / "risk_explanations.json"

    deltaq_explanations_path = xai_final_dir / "deltaQ_explanations.json"
    if not deltaq_explanations_path.exists():
        deltaq_explanations_path = xai_dir / "deltaQ_explanations.json"

    policy_summary_path = xai_final_dir / "policy_summary.json"
    if not policy_summary_path.exists():
        policy_summary_path = xai_dir / "policy_summary.json"

    selection_path = xai_final_dir / "explanations_selection.json"
    if not selection_path.exists():
        selection_path = xai_dir / "explanations_selection.json"

    if not risk_explanations_path.exists():
        raise FileNotFoundError(
            f"risk_explanations.json not found. Checked: {xai_final_dir} and {xai_dir}"
        )

    logger.info("Loading XAI files from: %s", risk_explanations_path.parent)

    logger.info("Loading XAI outputs from %s", xai_final_dir)
    risk_explanations = load_json(risk_explanations_path)
    deltaq_explanations = (
        load_json(deltaq_explanations_path) if deltaq_explanations_path.exists() else None
    )
    policy_summary = load_json(policy_summary_path) if policy_summary_path.exists() else None

    # Load model and test data
    logger.info("Loading Q-network from %s", ckpt_path)
    q_net = _load_q_network(ckpt_path, dataset_path, vocab_path, config, device)

    logger.info("Loading TEST split data...")
    test_data = load_dataset_with_splits(dataset_path, splits_path, "test")
    s_test = test_data["s"]
    sm_test = test_data["s_mask"]
    va_test = test_data["valid_actions"]

    # Get transition indices - prefer explanations_selection.json if available
    # (selection_path already resolved above)
    if selection_path.exists():
        logger.info("Using transition indices from explanations_selection.json")
        selection_info = load_json(selection_path)
        transition_indices = [
            t.get("transition_idx") for t in selection_info.get("selected_transitions", [])
        ]
        if len(transition_indices) == 0:
            # Fallback to risk_explanations
            transition_indices = [
                item.get("transition_idx") for item in risk_explanations.get("items", [])
            ]
    else:
        logger.info(
            "explanations_selection.json not found, using transition_idx from risk_explanations"
        )
        transition_indices = [
            item.get("transition_idx") for item in risk_explanations.get("items", [])
        ]

    if len(transition_indices) == 0:
        raise ValueError(
            "No transition indices found in explanations_selection.json or risk_explanations"
        )

    logger.info("Using %d transition indices for fidelity tests", len(transition_indices))

    # Filter test data to selected transitions
    max_idx = max(transition_indices)
    if max_idx >= len(s_test):
        raise ValueError(f"Transition index {max_idx} out of bounds (test size: {len(s_test)})")

    s_selected = s_test[transition_indices]
    sm_selected = sm_test[transition_indices]
    va_selected = va_test[transition_indices]

    # Validate perturbation
    logger.info("Validating perturbation...")
    if not _validate_perturbation(q_net, s_selected, sm_selected, device):
        raise RuntimeError("Perturbation validation failed - forward pass broken")

    # Test configuration
    tests_cfg = fidelity_cfg.get("tests", {})
    p_remove_list = fidelity_cfg.get("p_remove", [0.1, 0.2, 0.3, 0.5])
    n_random = fidelity_cfg.get("n_random", 20)
    n_items = fidelity_cfg.get("n_items", None)  # Limit items for smoke test
    debug_mode = fidelity_cfg.get("debug", False)  # Enable debug output

    # Load vocab for debug
    vocab = None
    if debug_mode:
        # vocab_path is already a full path or relative to data_interim_dir
        vocab_path_full = Path(vocab_path)
        if not vocab_path_full.is_absolute():
            vocab_path_full = Path(paths_cfg.get("data_interim_dir", "data/interim")) / vocab_path
        if vocab_path_full.exists():
            vocab = load_json(vocab_path_full)
        else:
            logger.warning(
                "Vocab file not found at %s, debug will not show token names", vocab_path_full
            )

    if n_items is not None:
        n_items = min(n_items, len(transition_indices))
        s_selected = s_selected[:n_items]
        sm_selected = sm_selected[:n_items]
        va_selected = va_selected[:n_items]
        risk_explanations["items"] = risk_explanations["items"][:n_items]
        logger.info("Limited to %d items for smoke test", n_items)

    # Run tests
    all_results = []

    # Test 1: Q-drop
    if tests_cfg.get("q_drop", {}).get("enabled", True):
        logger.info("=" * 60)
        logger.info("TEST 1: Q-drop")
        logger.info("=" * 60)

        # Q-drop for q_star
        q_drop_results = _test_q_drop(
            q_net,
            s_selected,
            sm_selected,
            va_selected,
            risk_explanations,
            p_remove_list,
            n_random,
            seed,
            device,
            "q_star",
            debug=debug_mode,
            vocab=vocab,
        )
        all_results.extend(q_drop_results)

        # Q-drop for delta_q (if available)
        if deltaq_explanations is not None:
            deltaq_results = _test_q_drop(
                q_net,
                s_selected,
                sm_selected,
                va_selected,
                deltaq_explanations,
                p_remove_list,
                n_random,
                seed,
                device,
                "delta_q",
                debug=debug_mode,
                vocab=vocab,
            )
            all_results.extend(deltaq_results)

    # Test 2: Action-flip
    if tests_cfg.get("action_flip", {}).get("enabled", True):
        logger.info("=" * 60)
        logger.info("TEST 2: Action-flip")
        logger.info("=" * 60)

        action_flip_results = _test_action_flip(
            q_net,
            s_selected,
            sm_selected,
            va_selected,
            risk_explanations,
            p_remove_list,
            n_random,
            seed,
            device,
            debug=debug_mode,
            vocab=vocab,
            attr_label="phi_V",
        )
        all_results.extend(action_flip_results)

        # Also run the flip test masking the intervention attribution's drivers: this is
        # the variant that actually tests phi^{Delta Q} (previously only phi_V was tested
        # here while the paper attributed the result to the intervention explanation).
        if deltaq_explanations is not None:
            action_flip_dq = _test_action_flip(
                q_net,
                s_selected,
                sm_selected,
                va_selected,
                deltaq_explanations,
                p_remove_list,
                n_random,
                seed,
                device,
                debug=debug_mode,
                vocab=vocab,
                attr_label="phi_dq",
            )
            all_results.extend(action_flip_dq)

    # Test 3: Rank-consistency
    if tests_cfg.get("rank_consistency", {}).get("enabled", True):
        logger.info("=" * 60)
        logger.info("TEST 3: Rank-consistency")
        logger.info("=" * 60)

        ope_path = None  # Could load from artifacts/ope/ope_dr.json if needed
        rank_metrics = tests_cfg.get("rank_consistency", {}).get("metrics", ["spearman", "kendall"])

        rank_results = _test_rank_consistency(
            risk_explanations, policy_summary, ope_path, rank_metrics, seed
        )
        all_results.extend(rank_results)

    # Save results
    df = pd.DataFrame(all_results)

    # Add metadata
    try:
        ckpt_hash = (
            fingerprint_data([ckpt_path], use_dvc=False) if Path(ckpt_path).exists() else "unknown"
        )
    except Exception:
        ckpt_hash = "unknown"

    if config_obj is not None and hasattr(config_obj, "config_hash"):
        config_hash = config_obj.config_hash
    else:
        config_hash = "unknown"

    git_commit_info = get_git_commit()
    git_commit = (
        git_commit_info.get("commit", "unknown") if isinstance(git_commit_info, dict) else "unknown"
    )

    df["ckpt_hash"] = ckpt_hash
    df["config_hash"] = config_hash
    df["git_commit"] = git_commit

    # Save CSV
    output_csv = fidelity_cfg.get("out_csv", "artifacts/fidelity/fidelity.csv")
    output_path = Path(output_csv)
    ensure_dir(output_path.parent)
    df.to_csv(output_path, index=False)

    logger.info("=" * 60)
    logger.info("Fidelity tests completed!")
    logger.info("Results saved to: %s", output_path)
    logger.info("Total rows: %d", len(df))
    logger.info("=" * 60)
