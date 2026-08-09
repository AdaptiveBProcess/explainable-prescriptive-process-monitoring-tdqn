from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from xppm.ope.baselines import (
    compute_always_intervene_policy_probs,
    compute_heuristic_policy_probs,
    compute_noop_policy_probs,
)
from xppm.ope.behavior_model import BehaviorPolicy
from xppm.ope.weights import (
    dr_sequential,
    effective_sample_size,
    step_weights,
    trajectory_weights,
    wis_trajectory,
)
from xppm.rl.factory import AgentFactory
from xppm.rl.models.masking import apply_action_mask
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
) -> nn.Module:
    """Load the trained RL agent via :class:`~xppm.rl.factory.AgentFactory`."""
    return AgentFactory.load(ckpt_path, npz_path, vocab_path, config, device)


def _compute_q_values(
    q_net: nn.Module,
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
    from scipy.special import softmax

    masked_q = np.where(valid_actions > 0, q_values, -1e9)
    return softmax(masked_q / max(tau, 1e-6), axis=1)


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
    """Trajectory-aware off-policy evaluation (WIS + sequential DR) on the TEST split.

    Every arm -- the learned policy and *each* baseline -- goes through the same
    estimator and the same support gate, so no arm can be dropped after seeing
    its value.  The gate is ``ope.min_ess_fraction``, declared in the config and
    applied symmetrically; arms below it are reported with ``on_support: false``
    rather than omitted.

    Weights are cumulative ratios ``rho_{1:t}`` restarted at each case boundary.
    The single-step (contextual-bandit) numbers the earlier version reported are
    still computed and returned under ``*_stepwise`` keys for comparison.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_npz(dataset_path)
    splits = load_json(splits_path)

    case_ids_all = data["case_ptr"]
    test_cases = set(int(c) for c in splits["cases"]["test"])
    test_mask = np.isin(case_ids_all, list(test_cases))
    idx_test = np.nonzero(test_mask)[0]
    if idx_test.size == 0:
        raise ValueError("No test transitions found in splits.json (test_mask is empty).")

    s = data["s"][test_mask]
    s_mask = data["s_mask"][test_mask]
    a = data["a"][test_mask]
    r = data["r"][test_mask]
    valid_actions = data["valid_actions"][test_mask]
    case_ids = data["case_ptr"][test_mask]
    if "t_ptr" in data:
        t_ptr = np.asarray(data["t_ptr"])[test_mask]
    else:  # fall back to storage order within each case
        t_ptr = np.zeros(case_ids.shape[0], dtype=np.int64)
        for c in np.unique(case_ids):
            m = np.nonzero(case_ids == c)[0]
            t_ptr[m] = np.arange(m.shape[0])

    n_test = int(s.shape[0])
    n_actions = int(valid_actions.shape[1])

    batch_size = int(config.get("behavior_model", {}).get("eval_batch_size", 1024))
    ds = _OpeDataset(s, s_mask, a, valid_actions, np.arange(n_test, dtype=int))
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    q_net = _load_q_network(
        ckpt_path=ckpt_path,
        npz_path=dataset_path,
        vocab_path=vocab_path,
        config=config,
        device=device,
    )
    q_sa, v_s = _compute_q_values(q_net, loader, device=device)

    tau = float(config.get("ope", {}).get("pi_e_temperature", 1.0))
    q_all_list: list[np.ndarray] = []
    with torch.no_grad():
        for s_b, s_mask_b, _, valid_b in loader:
            q_all_list.append(q_net(s_b.to(device), s_mask_b.to(device)).detach().cpu().numpy())
    q_all = np.concatenate(q_all_list, axis=0)
    pi_e = _compute_policy_probs_from_q(q_all, valid_actions=valid_actions, tau=tau)

    if behavior.probs.shape[0] != data["s"].shape[0]:
        raise ValueError(
            "BehaviorPolicy.probs must have shape (N, n_actions) matching D_offline.npz."
        )
    if behavior.n_actions != n_actions:
        raise ValueError("BehaviorPolicy.n_actions does not match dataset n_actions.")
    pi_b = behavior.probs[idx_test, :]

    rows = np.arange(n_test)
    pi_b_logged = pi_b[rows, a]
    gamma = float(config.get("ope", {}).get("gamma", 1.0))

    # --- support gate, declared before any value is looked at ---------------
    min_ess = float(config.get("ope", {}).get("min_ess_fraction", 0.10))
    _cc = config.get("ope", {}).get("cum_rho_cap", None)
    cum_cap = float(_cc) if _cc else None

    unique_case_ids = np.unique(case_ids)
    n_unique_cases = int(unique_case_ids.shape[0])

    def _arm(pi_arm: np.ndarray) -> dict[str, Any]:
        """Score one policy with both estimators and the support diagnostics."""
        rho = step_weights(pi_arm[rows, a], pi_b_logged, rho_cap)
        wis = wis_trajectory(rho, r, case_ids, t_ptr, cum_cap)
        dr = dr_sequential(rho, r, q_sa, v_s, case_ids, t_ptr, cum_cap, gamma=gamma)
        _, w_case, returns = trajectory_weights(rho, case_ids, t_ptr, cum_cap, r=r)
        ess_step, ess_step_frac = effective_sample_size(rho)
        return {
            "wis_mean": wis["value"],
            "dr_mean": dr["value"],
            "ess": wis["ess"],
            "ess_fraction": wis["ess_fraction"],
            "on_support": bool(wis["ess_fraction"] >= min_ess),
            "stepwise": {
                "wis_per_decision": float(np.sum(rho * r) / max(float(np.sum(rho)), 1e-12)),
                "dr_per_decision": float((rho * (r - q_sa) + v_s).mean()),
                "ess_fraction": ess_step_frac,
            },
            "_w_case": w_case,
            "_returns": returns,
        }

    actions_cfg = config.get("mdp", {}).get("actions", {})
    action_names = actions_cfg.get("id2name", ["NOOP"])
    noop_action_id = 0
    if "noop_action" in actions_cfg:
        noop_action_name = actions_cfg["noop_action"]
        if noop_action_name in action_names:
            noop_action_id = action_names.index(noop_action_name)

    arms: dict[str, np.ndarray] = {
        "policy": pi_e,
        "noop": compute_noop_policy_probs(valid_actions, noop_action_id, n_actions),
        "always_intervene": compute_always_intervene_policy_probs(
            valid_actions, noop_action_id, n_actions
        ),
    }
    if config.get("ope", {}).get("evaluate_heuristic", False):
        arms["heuristic"] = compute_heuristic_policy_probs(
            valid_actions, s_mask.sum(axis=1), n_actions, config
        )

    scored = {name: _arm(pi) for name, pi in arms.items()}

    # --- case-level bootstrap: per-arm intervals and paired differences -----
    rng = np.random.default_rng(seed=int(config.get("repro", {}).get("seed", 42)))
    boot_case_matrix = rng.integers(0, n_unique_cases, size=(n_bootstrap, n_unique_cases))

    def _wis_from_case_weights(w: np.ndarray, ret: np.ndarray, sel: np.ndarray) -> float:
        den = float(np.sum(w[sel]))
        return float(np.sum(w[sel] * ret[sel]) / den) if den > 0 else 0.0

    per_arm_samples: dict[str, list[float]] = {k: [] for k in scored}
    paired_samples: dict[str, list[float]] = {k: [] for k in scored if k != "policy"}
    for b in range(n_bootstrap):
        sel = boot_case_matrix[b]
        vals = {
            name: _wis_from_case_weights(sc["_w_case"], sc["_returns"], sel)
            for name, sc in scored.items()
        }
        for name, v in vals.items():
            per_arm_samples[name].append(v)
        for name in paired_samples:
            paired_samples[name].append(vals["policy"] - vals[name])

    def _ci(xs: list[float]) -> list[float]:
        arr = np.asarray(xs)
        return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]

    def _summarize(xs: list[float]) -> dict[str, Any]:
        arr = np.asarray(xs)
        return {
            "mean": float(arr.mean()),
            "ci95": _ci(xs),
            "p_gt_0": float((arr > 0).mean()),
        }

    baselines_results: dict[str, Any] = {}
    for name, sc in scored.items():
        if name == "policy":
            continue
        baselines_results[name] = {
            "dr_mean": sc["dr_mean"],
            "wis_mean": sc["wis_mean"],
            "wis_ci95": _ci(per_arm_samples[name]),
            "ess_fraction": sc["ess_fraction"],
            "on_support": sc["on_support"],
            "stepwise": sc["stepwise"],
        }

    paired_diff: dict[str, Any] = {
        f"vs_{name}": {
            **_summarize(paired_samples[name]),
            "baseline_on_support": scored[name]["on_support"],
        }
        for name in paired_samples
    }
    paired_diff["note"] = (
        "case-level paired bootstrap on trajectory WIS; the same resample scores "
        "policy and baseline. Differences against an off-support baseline "
        "(on_support=false) are reported but not interpretable."
    )

    pol = scored["policy"]
    rho_policy = step_weights(pi_e[rows, a], pi_b_logged, rho_cap)
    diagnostics: dict[str, Any] = {
        "estimator": "trajectory (per-decision cumulative IS), reported per case",
        "gamma": gamma,
        "rho_step_cap": float(rho_cap),
        "cum_rho_cap": cum_cap,
        "pi_e_temperature": float(tau),
        "min_ess_fraction": min_ess,
        "support_gate": (
            "declared in config as ope.min_ess_fraction and applied identically to "
            "every arm; arms below it are reported with on_support=false, never dropped"
        ),
        "rho_step_percentiles": [float(x) for x in np.percentile(rho_policy, [50, 75, 90, 95, 99])],
        "ess": float(pol["ess"]),
        "ess_fraction": float(pol["ess_fraction"]),
        "on_support": pol["on_support"],
        "n_test_transitions": int(n_test),
        "n_test_cases": n_unique_cases,
        "arms_on_support": {k: v["on_support"] for k, v in scored.items()},
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

    # empirical per-case return of the logged policy (no OPE)
    _, _, logged_returns = trajectory_weights(np.ones(n_test), case_ids, t_ptr, None, r=r)
    assert logged_returns is not None

    results: dict[str, Any] = {
        "tdqn_dr_mean": pol["dr_mean"],
        "tdqn_wis_mean": pol["wis_mean"],
        "tdqn_wis_ci95": _ci(per_arm_samples["policy"]),
        "tdqn_on_support": pol["on_support"],
        "tdqn_stepwise": pol["stepwise"],
        "behavior_empirical_return_mean": float(logged_returns.mean()),
        "baselines": baselines_results,
    }

    return {
        "results": results,
        "diagnostics": diagnostics,
        "paired_diff": paired_diff,
        "n_bootstrap": int(n_bootstrap),
    }
