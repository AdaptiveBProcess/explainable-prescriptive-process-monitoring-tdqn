from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from xppm.ope.baselines import (
    compute_heuristic_policy_probs,
    compute_noop_policy_probs,
    evaluate_baseline_policy,
)
from xppm.ope.behavior_model import BehaviorPolicy
from xppm.rl.models.masking import apply_action_mask
from xppm.rl.train_tdqn import TransformerQNetwork
from xppm.utils.io import load_json, load_npz


class _OpeDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Dataset wrapper for OPE (test split transitions)."""

    def __init__(
        self,
        s: np.ndarray,
        s_mask: np.ndarray,
        a: np.ndarray,
        valid_actions: np.ndarray,
        indices: np.ndarray,
    ) -> None:
        self.s = s
        self.s_mask = s_mask
        self.a = a
        self.valid_actions = valid_actions
        self.indices = indices

    def __len__(self) -> int:  # type: ignore[override]
        return int(self.indices.shape[0])

    def __getitem__(  # type: ignore[override]
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        i = int(self.indices[idx])
        return (
            torch.from_numpy(self.s[i]).long(),
            torch.from_numpy(self.s_mask[i]).float(),
            torch.tensor(int(self.a[i]), dtype=torch.long),
            torch.from_numpy(self.valid_actions[i]).float(),
        )


def _load_q_network(
    ckpt_path: str | Path,
    npz_path: str | Path,
    vocab_path: str | Path,
    config: dict[str, Any],
    device: torch.device,
) -> TransformerQNetwork:
    """Load TransformerQNetwork with the architecture used in training."""
    data = load_npz(npz_path)
    training_cfg = config.get("training", {})
    transformer_cfg = training_cfg.get("transformer", {})

    max_len = int(transformer_cfg.get("max_len", data["s"].shape[1]))
    d_model = int(transformer_cfg.get("d_model", 128))
    n_heads = int(transformer_cfg.get("n_heads", 4))
    n_layers = int(transformer_cfg.get("n_layers", 3))
    dropout = float(transformer_cfg.get("dropout", 0.1))

    vocab = load_json(vocab_path)
    token2id = vocab.get("token2id", {})
    vocab_size = int(len(token2id)) if token2id else int(data["s"].max() + 1)
    n_actions = int(data["valid_actions"].shape[1])

    q_net = TransformerQNetwork(
        vocab_size=vocab_size,
        max_len=max_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
        n_actions=n_actions,
    ).to(device)
    raw_ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = raw_ckpt.get("model_state_dict", raw_ckpt)
    q_net.load_state_dict(state_dict, strict=False)
    q_net.eval()
    return q_net


def _compute_q_values(
    q_net: TransformerQNetwork,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Q(s,a) and V(s) for all transitions in the loader."""
    q_sa_list: list[np.ndarray] = []
    v_s_list: list[np.ndarray] = []

    with torch.no_grad():
        for s_b, s_mask_b, a_b, valid_b in loader:
            s_b = s_b.to(device)
            s_mask_b = s_mask_b.to(device)
            a_b = a_b.to(device)
            valid_b = valid_b.to(device)

            q_values = q_net(s_b, s_mask_b)  # (batch, n_actions)
            # Apply action mask for V(s)
            q_masked = apply_action_mask(q_values, valid_b)
            v_s, _ = torch.max(q_masked, dim=-1)

            # Q(s,a_logged)
            q_sa = q_values.gather(1, a_b.view(-1, 1)).squeeze(1)

            q_sa_list.append(q_sa.detach().cpu().numpy())
            v_s_list.append(v_s.detach().cpu().numpy())

    return np.concatenate(q_sa_list, axis=0), np.concatenate(v_s_list, axis=0)


def _compute_policy_probs_from_q(
    q_values: np.ndarray,
    valid_actions: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Compute π_e(a|s) as softmax over Q with action masking."""
    # Mask invalid actions
    masked_q = np.where(valid_actions > 0, q_values, -1e9)
    # Temperature softmax
    logits = masked_q / max(tau, 1e-6)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / np.clip(exp_logits.sum(axis=1, keepdims=True), 1e-12, None)
    return probs


def doubly_robust_estimate(
    ckpt_path: str | Path,
    dataset_path: str | Path,
    splits_path: str | Path,
    vocab_path: str | Path,
    config: dict[str, Any],
    behavior: BehaviorPolicy,
    rho_cap: float = 20.0,
    n_bootstrap: int = 200,
) -> dict[str, Any]:
    """Doubly Robust Off-Policy Evaluation (step-wise DR + WIS) on TEST split.

    This implementation follows the practical guidelines in 2-2-setup.md:
    - Uses DR on held-out TEST split only.
    - Uses behavior policy π_b estimated from TRAIN (and validated on VAL).
    - Uses π_e derived from Q_theta via softmax over Q with action mask.
    - Applies propensity clipping and IS weight truncation.
    - Computes bootstrap confidence intervals.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_npz(dataset_path)
    splits = load_json(splits_path)

    # Build TEST mask from case assignments
    case_ids = data["case_ptr"]
    test_cases = set(int(c) for c in splits["cases"]["test"])
    test_mask = np.isin(case_ids, list(test_cases))
    idx_test = np.nonzero(test_mask)[0]

    if idx_test.size == 0:
        raise ValueError("No test transitions found in splits.json (test_mask is empty).")

    # Extract TEST transitions
    s = data["s"][test_mask]
    s_mask = data["s_mask"][test_mask]
    a = data["a"][test_mask]
    r = data["r"][test_mask]
    valid_actions = data["valid_actions"][test_mask]
    case_ids = data["case_ptr"][test_mask]  # Keep case_ids for bootstrap by cases

    n_test = int(s.shape[0])
    n_actions = int(valid_actions.shape[1])

    # Build loader for Q evaluation
    batch_size = int(config.get("behavior_model", {}).get("eval_batch_size", 1024))
    ds = _OpeDataset(s, s_mask, a, valid_actions, np.arange(n_test, dtype=int))
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    # Load Q network and compute Q(s,a) and V(s)
    q_net = _load_q_network(
        ckpt_path=ckpt_path,
        npz_path=dataset_path,
        vocab_path=vocab_path,
        config=config,
        device=device,
    )
    q_sa, v_s = _compute_q_values(q_net, loader, device=device)

    # Build π_e(a|s) from Q (softmax policy with temperature tau)
    tau = float(config.get("ope", {}).get("pi_e_temperature", 1.0))
    # To compute π_e we need Q(s,·) for all actions; recompute in np for test
    # We can reuse the same loader but now collect full Q
    q_all_list: list[np.ndarray] = []
    with torch.no_grad():
        for s_b, s_mask_b, _, valid_b in loader:
            s_b = s_b.to(device)
            s_mask_b = s_mask_b.to(device)
            valid_b = valid_b.to(device)
            q_values = q_net(s_b, s_mask_b)
            q_all_list.append(q_values.detach().cpu().numpy())
    q_all = np.concatenate(q_all_list, axis=0)  # (n_test, n_actions)

    pi_e = _compute_policy_probs_from_q(q_all, valid_actions=valid_actions, tau=tau)

    # Behavior policy π_b(a|s) for TEST transitions
    if behavior.probs.shape[0] != data["s"].shape[0]:
        raise ValueError(
            "BehaviorPolicy.probs must have shape (N, n_actions) matching D_offline.npz."
        )
    if behavior.n_actions != n_actions:
        raise ValueError("BehaviorPolicy.n_actions does not match dataset n_actions.")

    pi_b = behavior.probs[idx_test, :]  # (n_test, n_actions)

    # Extract π_e and π_b at logged actions
    pi_e_logged = pi_e[np.arange(n_test), a]
    pi_b_logged = pi_b[np.arange(n_test), a]

    # Propensity clipping and IS weights
    pi_b_clipped = np.clip(pi_b_logged, 1e-6, None)
    rho = pi_e_logged / pi_b_clipped
    rho_trunc = np.clip(rho, 0.0, rho_cap)

    # Step-wise DR estimator:
    #   DR_t = ρ_t * (r_t - Q(s_t, a_t)) + V(s_t)
    dr_step = rho_trunc * (r - q_sa) + v_s
    dr_mean = float(dr_step.mean())

    # Weighted IS (self-normalized)
    wis_num = float(np.sum(rho_trunc * r))
    wis_den = float(np.sum(rho_trunc)) if float(np.sum(rho_trunc)) > 0 else 1.0
    wis = wis_num / wis_den

    # Diagnostics: rho distribution and ESS
    rho_percentiles = np.percentile(rho_trunc, [50, 75, 90, 95, 99]).tolist()
    ess = (np.sum(rho_trunc) ** 2) / np.sum(rho_trunc**2) if np.sum(rho_trunc**2) > 0 else 0.0
    ess_fraction = ess / float(n_test)

    # Bootstrap CIs by cases (not by transitions) to account for correlation within cases
    # Get unique case_ids in TEST (case_ids is already filtered by test_mask)
    case_ids_test = case_ids  # Already filtered in line 177
    unique_case_ids = np.unique(case_ids_test)
    n_unique_cases = len(unique_case_ids)

    # Build mapping: case_id -> list of transition indices (relative to test set)
    case_to_indices: dict[int, np.ndarray] = {}
    for case_id in unique_case_ids:
        case_to_indices[case_id] = np.nonzero(case_ids_test == case_id)[0]

    rng = np.random.default_rng(seed=int(config.get("repro", {}).get("seed", 42)))
    dr_samples: list[float] = []
    wis_samples: list[float] = []
    for _ in range(n_bootstrap):
        # Sample cases with replacement
        boot_case_ids = rng.choice(unique_case_ids, size=n_unique_cases, replace=True)
        # Collect all transition indices for resampled cases
        boot_indices_list: list[int] = []
        for case_id in boot_case_ids:
            boot_indices_list.extend(case_to_indices[case_id].tolist())
        idx_boot = np.array(boot_indices_list, dtype=np.int64)

        if len(idx_boot) == 0:
            continue  # Skip empty bootstrap

        dr_samples.append(float(dr_step[idx_boot].mean()))
        w_boot = rho_trunc[idx_boot]
        r_boot = r[idx_boot]
        wis_num_b = float(np.sum(w_boot * r_boot))
        wis_den_b = float(np.sum(w_boot)) if float(np.sum(w_boot)) > 0 else 1.0
        wis_samples.append(wis_num_b / wis_den_b)

    dr_ci = np.percentile(np.array(dr_samples), [2.5, 97.5]).tolist()
    wis_ci = np.percentile(np.array(wis_samples), [2.5, 97.5]).tolist()

    # Behavior empirical return on TEST (no OPE, solo promedio observado)
    behavior_empirical_return = float(r.mean())

    # Evaluate baselines (no-op and heuristic)
    baselines_results: dict[str, Any] = {}

    # Get action names and noop action ID from config
    actions_cfg = config.get("mdp", {}).get("actions", {})
    action_names = actions_cfg.get("id2name", ["NOOP"])
    noop_action_id = 0  # Default
    if "noop_action" in actions_cfg:
        noop_action_name = actions_cfg["noop_action"]
        if noop_action_name in action_names:
            noop_action_id = action_names.index(noop_action_name)

    # Compute prefix lengths for heuristic
    prefix_lengths = s_mask.sum(axis=1)

    # No-op baseline
    pi_noop = compute_noop_policy_probs(valid_actions, noop_action_id, n_actions)
    noop_results = evaluate_baseline_policy(
        pi_baseline=pi_noop,
        pi_b=pi_b,
        a_logged=a,
        r=r,
        q_sa=q_sa,
        v_s=v_s,
        valid_actions=valid_actions,
        gamma=float(config.get("training", {}).get("tdqn", {}).get("gamma", 0.99)),
        rho_cap=rho_cap,
    )

    # Bootstrap CI for no-op
    noop_dr_samples: list[float] = []
    noop_wis_samples: list[float] = []
    for _ in range(n_bootstrap):
        boot_case_ids = rng.choice(unique_case_ids, size=n_unique_cases, replace=True)
        boot_indices_list: list[int] = []
        for case_id in boot_case_ids:
            boot_indices_list.extend(case_to_indices[case_id].tolist())
        idx_boot = np.array(boot_indices_list, dtype=np.int64)
        if len(idx_boot) == 0:
            continue

        pi_noop_boot = pi_noop[idx_boot]
        pi_b_boot = pi_b[idx_boot]
        a_boot = a[idx_boot]
        r_boot = r[idx_boot]
        q_sa_boot = q_sa[idx_boot]
        v_s_boot = v_s[idx_boot]
        valid_boot = valid_actions[idx_boot]

        noop_boot_results = evaluate_baseline_policy(
            pi_baseline=pi_noop_boot,
            pi_b=pi_b_boot,
            a_logged=a_boot,
            r=r_boot,
            q_sa=q_sa_boot,
            v_s=v_s_boot,
            valid_actions=valid_boot,
            gamma=float(config.get("training", {}).get("tdqn", {}).get("gamma", 0.99)),
            rho_cap=rho_cap,
        )
        noop_dr_samples.append(noop_boot_results["dr_mean"])
        noop_wis_samples.append(noop_boot_results["wis_mean"])

    noop_dr_ci = (
        np.percentile(np.array(noop_dr_samples), [2.5, 97.5]).tolist()
        if noop_dr_samples
        else [0.0, 0.0]
    )
    noop_wis_ci = (
        np.percentile(np.array(noop_wis_samples), [2.5, 97.5]).tolist()
        if noop_wis_samples
        else [0.0, 0.0]
    )

    baselines_results["noop"] = {
        "dr_mean": noop_results["dr_mean"],
        "dr_ci95": [float(noop_dr_ci[0]), float(noop_dr_ci[1])],
        "wis_mean": noop_results["wis_mean"],
        "wis_ci95": [float(noop_wis_ci[0]), float(noop_wis_ci[1])],
        "ess_fraction": noop_results["ess_fraction"],
    }

    # Heuristic baseline (if enabled in config)
    if config.get("ope", {}).get("evaluate_heuristic", False):
        pi_heuristic = compute_heuristic_policy_probs(
            valid_actions, prefix_lengths, n_actions, config
        )
        heuristic_results = evaluate_baseline_policy(
            pi_baseline=pi_heuristic,
            pi_b=pi_b,
            a_logged=a,
            r=r,
            q_sa=q_sa,
            v_s=v_s,
            valid_actions=valid_actions,
            gamma=float(config.get("training", {}).get("tdqn", {}).get("gamma", 0.99)),
            rho_cap=rho_cap,
        )

        # Bootstrap CI for heuristic
        heuristic_dr_samples: list[float] = []
        heuristic_wis_samples: list[float] = []
        for _ in range(n_bootstrap):
            boot_case_ids = rng.choice(unique_case_ids, size=n_unique_cases, replace=True)
            boot_indices_list = []
            for case_id in boot_case_ids:
                boot_indices_list.extend(case_to_indices[case_id].tolist())
            idx_boot = np.array(boot_indices_list, dtype=np.int64)
            if len(idx_boot) == 0:
                continue

            pi_heuristic_boot = pi_heuristic[idx_boot]
            pi_b_boot = pi_b[idx_boot]
            a_boot = a[idx_boot]
            r_boot = r[idx_boot]
            q_sa_boot = q_sa[idx_boot]
            v_s_boot = v_s[idx_boot]
            valid_boot = valid_actions[idx_boot]

            heuristic_boot_results = evaluate_baseline_policy(
                pi_baseline=pi_heuristic_boot,
                pi_b=pi_b_boot,
                a_logged=a_boot,
                r=r_boot,
                q_sa=q_sa_boot,
                v_s=v_s_boot,
                valid_actions=valid_boot,
                gamma=float(config.get("training", {}).get("tdqn", {}).get("gamma", 0.99)),
                rho_cap=rho_cap,
            )
            heuristic_dr_samples.append(heuristic_boot_results["dr_mean"])
            heuristic_wis_samples.append(heuristic_boot_results["wis_mean"])

        heuristic_dr_ci = (
            np.percentile(np.array(heuristic_dr_samples), [2.5, 97.5]).tolist()
            if heuristic_dr_samples
            else [0.0, 0.0]
        )
        heuristic_wis_ci = (
            np.percentile(np.array(heuristic_wis_samples), [2.5, 97.5]).tolist()
            if heuristic_wis_samples
            else [0.0, 0.0]
        )

        baselines_results["heuristic"] = {
            "dr_mean": heuristic_results["dr_mean"],
            "dr_ci95": [float(heuristic_dr_ci[0]), float(heuristic_dr_ci[1])],
            "wis_mean": heuristic_results["wis_mean"],
            "wis_ci95": [float(heuristic_wis_ci[0]), float(heuristic_wis_ci[1])],
            "ess_fraction": heuristic_results["ess_fraction"],
        }

    diagnostics: dict[str, Any] = {
        "rho_percentiles": [float(x) for x in rho_percentiles],
        "rho_cap": float(rho_cap),
        "pi_e_temperature": float(tau),
        "ess": float(ess),
        "ess_fraction": float(ess_fraction),
        "n_test_transitions": int(n_test),
        "n_test_cases": int(n_unique_cases),
        "q_values_stats": {
            "q_sa_mean": float(q_sa.mean()),
            "q_sa_std": float(q_sa.std()),
            "v_s_mean": float(v_s.mean()),
            "v_s_std": float(v_s.std()),
        },
        "behavior_model": {
            "val_nll": float(behavior.metrics.get("val_nll", 0.0)),
            "val_acc": float(behavior.metrics.get("val_acc", 0.0)),
            "val_mean_entropy": float(behavior.metrics.get("val_mean_entropy", 0.0)),
        },
    }

    results: dict[str, Any] = {
        "tdqn_dr_mean": float(dr_mean),
        "tdqn_dr_ci95": [float(dr_ci[0]), float(dr_ci[1])],
        "tdqn_wis_mean": float(wis),
        "tdqn_wis_ci95": [float(wis_ci[0]), float(wis_ci[1])],
        "behavior_empirical_return_mean": behavior_empirical_return,
        "baselines": baselines_results,
    }

    return {
        "results": results,
        "diagnostics": diagnostics,
        "n_bootstrap": int(n_bootstrap),
    }
