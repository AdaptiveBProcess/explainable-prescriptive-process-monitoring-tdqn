"""Policy summary via k-means clustering on encoder embeddings."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.cluster import KMeans

from xppm.rl.models.masking import apply_action_mask
from xppm.rl.train_tdqn import TransformerQNetwork
from xppm.utils.logging import get_logger

logger = get_logger(__name__)


def extract_encoder_embeddings(
    q_net: TransformerQNetwork,
    states: np.ndarray,
    state_masks: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """Extract pooled encoder embeddings (state_proj + ReLU, before q_head).

    Args:
        q_net: loaded Q-network (eval mode)
        states: (N, max_len) token IDs
        state_masks: (N, max_len) masks
        device: torch device
        batch_size: processing batch size

    Returns:
        embeddings: (N, d_model) pooled encoder representations
    """
    n = states.shape[0]
    embeddings_list = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        s_b = torch.from_numpy(states[start:end]).long().to(device)
        m_b = torch.from_numpy(state_masks[start:end]).float().to(device)

        with torch.no_grad():
            s_clamped = torch.clamp(s_b, min=0, max=q_net.vocab_size - 1)
            x = q_net.embedding(s_clamped)
            encoded = q_net.encoder(x)

            # Pool last non-padded token
            lengths = m_b.sum(dim=1).long() - 1
            lengths = torch.clamp(lengths, min=0, max=q_net.max_len - 1)
            batch_idx = torch.arange(encoded.size(0), device=device)
            state_repr = encoded[batch_idx, lengths]

            # state_proj + ReLU (same as TransformerQNetwork.forward)
            state_repr = q_net.state_proj(state_repr)
            state_repr = torch.relu(state_repr)

        embeddings_list.append(state_repr.cpu().numpy())

    return np.concatenate(embeddings_list, axis=0)


def _compute_q_and_actions(
    q_net: TransformerQNetwork,
    states: np.ndarray,
    state_masks: np.ndarray,
    valid_actions: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
    contrast_action_id: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Compute V(s), a*, deltaQ, and policy margin for all transitions.

    Returns:
        v_s: (N,) V(s) = max_a Q_masked(s,a)
        a_star: (N,) argmax action
        delta_q: (N,) Q(a*) - Q(contrast) if contrast_action_id provided, else None
        policy_margin: (N,) Q(a*) - Q(a2) where a2 is second-best action
    """
    n = states.shape[0]
    v_list = []
    a_list = []
    dq_list = [] if contrast_action_id is not None else None
    margin_list = []

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

            # Policy margin: Q(a*) - Q(a2) where a2 is second-best
            # Sort Q-values (descending) and take difference between top 2
            q_sorted, _ = torch.sort(q_masked, dim=-1, descending=True)
            # Handle cases with only 1 valid action: margin = 0 (no alternative)
            n_valid_per_state = va_b.sum(dim=1)
            margin = torch.where(
                n_valid_per_state > 1,
                q_sorted[:, 0] - q_sorted[:, 1],  # Q(a*) - Q(a2)
                torch.zeros_like(q_sorted[:, 0]),  # No alternative, margin=0
            )

            # DeltaQ if contrast action provided
            if contrast_action_id is not None:
                q_star = q_vals.gather(1, a_star.unsqueeze(1)).squeeze(1)
                q_contrast = q_vals[:, contrast_action_id]
                delta_q = q_star - q_contrast
                dq_list.append(delta_q.cpu().numpy())

        v_list.append(v_s.cpu().numpy())
        a_list.append(a_star.cpu().numpy())
        margin_list.append(margin.cpu().numpy())

    delta_q_arr = np.concatenate(dq_list) if dq_list is not None else None
    return (
        np.concatenate(v_list),
        np.concatenate(a_list),
        delta_q_arr,
        np.concatenate(margin_list),
    )


def summarize_policy(
    q_net: TransformerQNetwork,
    states: np.ndarray,
    state_masks: np.ndarray,
    valid_actions: np.ndarray,
    case_ptrs: np.ndarray,
    t_ptrs: np.ndarray,
    action_names: list[str],
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Generate policy summary via k-means clustering on encoder embeddings.

    Args:
        q_net: loaded Q-network
        states, state_masks, valid_actions: TEST split arrays
        case_ptrs, t_ptrs: case and time identifiers
        action_names: list of action names by ID
        config: xai config section
        device: torch device

    Returns:
        Policy summary dict with cluster information
    """
    cluster_cfg = config.get("clustering", {})
    k = int(cluster_cfg.get("k", 8))
    cluster_seed = int(cluster_cfg.get("seed", 123))
    n_prototypes = int(cluster_cfg.get("n_prototypes", 3))
    n_actions = valid_actions.shape[1]

    logger.info("Extracting encoder embeddings for %d transitions...", states.shape[0])
    embeddings = extract_encoder_embeddings(q_net, states, state_masks, device)

    # Get contrast action ID from config if available
    contrast_action_id = (
        config.get("methods", {}).get("intervention", {}).get("contrast", {}).get("action_id", 0)
    )  # Default to 0 (do_nothing)

    logger.info("Computing Q-values, actions, deltaQ, and policy margins...")
    v_s, a_star, delta_q, policy_margin = _compute_q_and_actions(
        q_net,
        states,
        state_masks,
        valid_actions,
        device,
        contrast_action_id=contrast_action_id,
    )

    logger.info("Running K-Means (k=%d, seed=%d)...", k, cluster_seed)
    kmeans = KMeans(n_clusters=k, random_state=cluster_seed, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    clusters = []
    for cid in range(k):
        mask = labels == cid
        n_in_cluster = int(mask.sum())
        if n_in_cluster == 0:
            continue

        # Action distribution
        actions_in_cluster = a_star[mask]
        action_counts = np.bincount(actions_in_cluster, minlength=n_actions)
        action_dist = {
            action_names[i]: float(action_counts[i] / n_in_cluster) for i in range(n_actions)
        }

        # V(s) stats
        v_in_cluster = v_s[mask]

        # DeltaQ stats (if available)
        delta_q_in_cluster = delta_q[mask] if delta_q is not None else None
        mean_delta_q = float(delta_q_in_cluster.mean()) if delta_q_in_cluster is not None else None

        # Policy margin stats (confidence measure)
        margin_in_cluster = policy_margin[mask]
        mean_margin = float(margin_in_cluster.mean())
        # Policy entropy approximation: lower margin = higher uncertainty
        # We report margin as "confidence" (higher = more confident)

        # Prototypes: closest to centroid
        cluster_embeddings = embeddings[mask]
        centroid = kmeans.cluster_centers_[cid]
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        proto_local_indices = np.argsort(distances)[:n_prototypes]
        # Map local indices back to global indices
        global_indices = np.nonzero(mask)[0]
        proto_global = global_indices[proto_local_indices]

        prototypes = []
        for gi in proto_global:
            prototypes.append(
                {
                    "case_id": int(case_ptrs[gi]),
                    "t": int(t_ptrs[gi]),
                    "v": float(v_s[gi]),
                    "a_star": action_names[int(a_star[gi])],
                }
            )

        clusters.append(
            {
                "cluster_id": int(cid),
                "n": n_in_cluster,
                "action_distribution": action_dist,
                "mean_v": float(v_in_cluster.mean()),
                "std_v": float(v_in_cluster.std()),
                "mean_delta_q": mean_delta_q,  # Average gain from recommended vs contrast
                "mean_policy_margin": mean_margin,  # Average confidence (Q* - Q2)
                "prototypes": prototypes,
            }
        )

    logger.info("Policy summary: %d clusters created", len(clusters))

    return {
        "method": "kmeans",
        "k": k,
        "seed": cluster_seed,
        "n_transitions": int(states.shape[0]),
        "clusters": clusters,
    }
