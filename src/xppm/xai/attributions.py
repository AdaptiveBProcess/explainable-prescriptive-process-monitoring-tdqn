"""Integrated Gradients attributions for TDQN Q-network explanations."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch

from xppm.rl.models.masking import apply_action_mask
from xppm.rl.train_tdqn import TransformerQNetwork
from xppm.utils.logging import get_logger

logger = get_logger(__name__)


def _forward_from_embeddings(
    q_net: TransformerQNetwork,
    embeddings: torch.Tensor,
    state_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Forward pass from embedding space (bypassing q_net.embedding).

    Args:
        q_net: Q-network
        embeddings: (batch, max_len, d_model)
        state_mask: (batch, max_len) 1=real, 0=pad

    Returns:
        q_values: (batch, n_actions)
    """
    encoded = q_net.encoder(embeddings)

    if state_mask is not None:
        lengths = state_mask.sum(dim=1).long() - 1
        lengths = torch.clamp(lengths, min=0, max=q_net.max_len - 1)
        batch_idx = torch.arange(encoded.size(0), device=encoded.device)
        state_repr = encoded[batch_idx, lengths]
    else:
        state_repr = encoded[:, -1]

    state_repr = q_net.state_proj(state_repr)
    state_repr = torch.relu(state_repr)
    return q_net.q_head(state_repr)


def _build_baseline(
    states: torch.Tensor,
    method: str,
    pad_id: int,
    q_net: TransformerQNetwork,
) -> torch.Tensor:
    """Build baseline embedding tensor for IG.

    Args:
        states: (batch, max_len) token IDs
        method: "pad" or "zero_emb"
        pad_id: PAD token ID (typically 0)
        q_net: the Q-network (to access embedding layer)

    Returns:
        baseline_emb: (batch, max_len, d_model)
    """
    if method == "pad":
        pad_tokens = torch.full_like(states, pad_id)
        with torch.no_grad():
            baseline_emb = q_net.embedding(pad_tokens)
        return baseline_emb
    elif method == "zero_emb":
        return torch.zeros(
            states.size(0),
            states.size(1),
            q_net.d_model,
            device=states.device,
            dtype=torch.float,
        )
    else:
        raise ValueError(f"Unknown baseline method: {method}")


def integrated_gradients_embedding(
    q_net: TransformerQNetwork,
    states: torch.Tensor,
    state_mask: torch.Tensor,
    target_fn: Callable[[torch.Tensor], torch.Tensor],
    baseline_emb: torch.Tensor,
    n_steps: int = 32,
) -> torch.Tensor:
    """Compute IG attributions in embedding space.

    Args:
        q_net: Q-network (eval mode, but gradients enabled for embeddings)
        states: (batch, max_len) token IDs
        state_mask: (batch, max_len)
        target_fn: (batch, n_actions) -> (batch,) scalar target
        baseline_emb: (batch, max_len, d_model)
        n_steps: number of interpolation steps

    Returns:
        attributions: (batch, max_len, d_model) IG attributions
    """
    states_clamped = torch.clamp(states, min=0, max=q_net.vocab_size - 1)
    with torch.no_grad():
        input_emb = q_net.embedding(states_clamped)

    diff = input_emb - baseline_emb
    grad_sum = torch.zeros_like(input_emb)

    # Trapezoidal rule: average endpoints and midpoints for better accuracy
    alphas = torch.linspace(0.0, 1.0, n_steps + 1, device=states.device)

    for i in range(n_steps):
        # Use both endpoints for trapezoidal rule
        alpha_start = alphas[i]
        alpha_end = alphas[i + 1]
        alpha_mid = (alpha_start + alpha_end) / 2.0

        # Compute gradient at midpoint (main contribution)
        interp_mid = baseline_emb + alpha_mid * diff
        interp_mid = interp_mid.detach().requires_grad_(True)
        q_mid = _forward_from_embeddings(q_net, interp_mid, state_mask)
        targets_mid = target_fn(q_mid)
        scalar_mid = targets_mid.sum()
        scalar_mid.backward()
        grad_sum = grad_sum + interp_mid.grad.detach()
        interp_mid.grad = None

    # Average gradients (trapezoidal rule approximation)
    avg_grad = grad_sum / n_steps
    ig_attr = diff.detach() * avg_grad

    # Completeness check (relative error)
    with torch.no_grad():
        q_input = _forward_from_embeddings(q_net, input_emb, state_mask)
        q_base = _forward_from_embeddings(q_net, baseline_emb, state_mask)
        f_input = target_fn(q_input)
        f_base = target_fn(q_base)
        expected_diff = f_input - f_base
        ig_sum = ig_attr.sum(dim=(1, 2))
        abs_err = (ig_sum - expected_diff).abs()
        abs_err_mean = abs_err.mean().item()
        expected_diff_mean = expected_diff.mean().item()
        # Use the actual Q-value scale for the denominator so rel_err is
        # meaningful. Clamp to 1e-3 to avoid division by zero on trivial states
        # where f(x) ≈ f(baseline), but do NOT force a minimum of 1.0 which
        # would hide large errors when Q-values are small.
        denom = expected_diff.abs().clamp(min=1e-3)
        rel_err = (abs_err / denom).mean().item()
        log_fn = logger.warning if abs_err_mean > 0.5 else logger.info
        log_fn(
            "IG completeness: abs_err=%.4f, rel_err=%.4f, E[f(x)-f(base)]=%.4f%s",
            abs_err_mean,
            rel_err,
            expected_diff_mean,
            " — HIGH ABSOLUTE ERROR: attributions may be inaccurate" if abs_err_mean > 0.5 else "",
        )

    # Return attributions and completeness stats
    completeness_stats = {
        "abs_err_mean": float(abs_err_mean),
        "rel_err_mean": float(rel_err),
        "expected_diff_mean": float(expected_diff_mean),
        "abs_err_per_sample": abs_err.cpu().numpy().tolist(),
        "rel_err_per_sample": (abs_err / denom).cpu().numpy().tolist(),
    }

    return ig_attr, completeness_stats


def aggregate_token_importance(
    attr_emb: np.ndarray,
    state_masks: np.ndarray,
) -> np.ndarray:
    """Aggregate embedding-level attributions to token-level importance.

    token_importance[t] = sum(|attr_emb[t, :]|), zero for PAD.

    Args:
        attr_emb: (n_items, max_len, d_model)
        state_masks: (n_items, max_len)

    Returns:
        (n_items, max_len) token importance scores
    """
    token_imp = np.abs(attr_emb).sum(axis=2)  # (n_items, max_len)
    token_imp = token_imp * state_masks  # zero out PAD positions
    return token_imp


def compute_attributions(
    q_net: TransformerQNetwork,
    states: np.ndarray,
    state_masks: np.ndarray,
    valid_actions: np.ndarray,
    config: dict[str, Any],
    device: torch.device,
    target: str = "V",
    contrast_action_id: int | None = None,
    batch_size: int = 64,
) -> dict[str, np.ndarray]:
    """Compute attributions for a set of states.

    Args:
        q_net: loaded Q-network (eval mode)
        states: (n_items, max_len) token IDs
        state_masks: (n_items, max_len) binary masks
        valid_actions: (n_items, n_actions) action masks
        config: xai config section
        device: torch device
        target: "V" (risk) or "deltaQ" (intervention)
        contrast_action_id: action ID for contrast (required if target="deltaQ")
        batch_size: sub-batch size for processing

    Returns:
        dict with attributions_emb, token_importance, q_values, v_s, a_star,
        and if deltaQ: delta_q, q_star, q_contrast
    """
    risk_cfg = config.get("methods", {}).get("risk", {})
    baseline_method = risk_cfg.get("baseline", "pad")
    n_steps = int(risk_cfg.get("n_steps_ig", 32))
    pad_id = 0  # PAD token is always ID 0

    n_items = states.shape[0]
    all_attr = []
    all_q = []
    all_v = []
    all_a_star = []
    all_dq = [] if target == "deltaQ" else None
    all_q_star = [] if target == "deltaQ" else None
    all_q_contrast = [] if target == "deltaQ" else None
    # Accumulate completeness stats across batches
    completeness_stats_list: list[dict[str, Any]] = []

    for start in range(0, n_items, batch_size):
        end = min(start + batch_size, n_items)
        s_b = torch.from_numpy(states[start:end]).long().to(device)
        sm_b = torch.from_numpy(state_masks[start:end]).float().to(device)
        va_b = torch.from_numpy(valid_actions[start:end]).float().to(device)

        baseline_emb = _build_baseline(s_b, baseline_method, pad_id, q_net)

        # Fix actions from original input to avoid discontinuities during IG path
        # This improves IG completeness by keeping the target function smooth
        with torch.no_grad():
            q_input = q_net(s_b, sm_b)
            q_masked_input = apply_action_mask(q_input, va_b)
            a_star_fixed = q_masked_input.argmax(dim=-1)  # (batch,)

        if target == "V":
            # Use Q(s, a_star_fixed) - fixed action from original input
            # This avoids discontinuities from recalculating a* at each IG step
            def target_fn(q: torch.Tensor) -> torch.Tensor:
                # Use fixed a_star from original input, not recalculated
                q_star = q.gather(1, a_star_fixed.unsqueeze(1)).squeeze(1)
                return q_star
        elif target == "deltaQ":
            assert contrast_action_id is not None
            cid = contrast_action_id

            # For deltaQ, also fix contrast action (use provided ID if valid, else fallback)
            # Check if contrast is valid for all states in batch
            contrast_valid_batch = va_b[:, cid] > 0
            if not contrast_valid_batch.all():
                # Fallback: use first valid action for states where contrast is invalid
                a_contrast_fixed = torch.zeros_like(a_star_fixed)
                for i in range(len(a_contrast_fixed)):
                    if not contrast_valid_batch[i]:
                        valid_mask = va_b[i] > 0
                        if valid_mask.any():
                            a_contrast_fixed[i] = torch.nonzero(valid_mask)[0, 0]
                        else:
                            a_contrast_fixed[i] = a_star_fixed[i]  # Fallback to a_star
                    else:
                        a_contrast_fixed[i] = cid
            else:
                a_contrast_fixed = torch.full_like(a_star_fixed, cid)

            def target_fn(q: torch.Tensor) -> torch.Tensor:
                # Use fixed actions from original input
                q_star = q.gather(1, a_star_fixed.unsqueeze(1)).squeeze(1)
                q_contrast = q.gather(1, a_contrast_fixed.unsqueeze(1)).squeeze(1)
                return q_star - q_contrast
        else:
            raise ValueError(f"Unknown target: {target}")

        attr, completeness_batch = integrated_gradients_embedding(
            q_net,
            s_b,
            sm_b,
            target_fn,
            baseline_emb,
            n_steps=n_steps,
        )
        all_attr.append(attr.cpu().numpy())
        completeness_stats_list.append(completeness_batch)

        # Compute Q-values for this batch (for output metadata)
        with torch.no_grad():
            q_vals = q_net(s_b, sm_b)
            q_masked = apply_action_mask(q_vals, va_b)
            v_s, _ = torch.max(q_masked, dim=-1)
            a_star = q_masked.argmax(dim=-1)

        all_q.append(q_vals.cpu().numpy())
        all_v.append(v_s.cpu().numpy())
        all_a_star.append(a_star.cpu().numpy())

        if target == "deltaQ":
            q_star_vals = q_vals.gather(1, a_star.unsqueeze(1)).squeeze(1)
            q_contrast_vals = q_vals[:, contrast_action_id]
            dq = q_star_vals - q_contrast_vals
            all_dq.append(dq.cpu().numpy())
            all_q_star.append(q_star_vals.cpu().numpy())
            all_q_contrast.append(q_contrast_vals.cpu().numpy())

        logger.info(
            "Attributions [%s] batch %d-%d / %d done",
            target,
            start,
            end,
            n_items,
        )

    attr_emb = np.concatenate(all_attr, axis=0)
    token_imp = aggregate_token_importance(attr_emb, state_masks)

    result: dict[str, np.ndarray] = {
        "attributions_emb": attr_emb,
        "token_importance": token_imp,
        "q_values": np.concatenate(all_q, axis=0),
        "v_s": np.concatenate(all_v, axis=0),
        "a_star": np.concatenate(all_a_star, axis=0),
    }

    if target == "deltaQ":
        result["delta_q"] = np.concatenate(all_dq, axis=0)
        result["q_star"] = np.concatenate(all_q_star, axis=0)
        result["q_contrast"] = np.concatenate(all_q_contrast, axis=0)

    # Aggregate completeness stats across all batches
    if completeness_stats_list:
        all_abs_err = np.concatenate([s["abs_err_per_sample"] for s in completeness_stats_list])
        all_rel_err = np.concatenate([s["rel_err_per_sample"] for s in completeness_stats_list])
        result["ig_completeness"] = {
            "mean_abs_err": float(np.mean(all_abs_err)),
            "mean_rel_err": float(np.mean(all_rel_err)),
            "median_rel_err": float(np.median(all_rel_err)),
            "min_rel_err": float(np.min(all_rel_err)),
            "max_rel_err": float(np.max(all_rel_err)),
            "std_rel_err": float(np.std(all_rel_err)),
            "expected_diff_mean": float(
                np.mean([s["expected_diff_mean"] for s in completeness_stats_list])
            ),
        }

    return result
