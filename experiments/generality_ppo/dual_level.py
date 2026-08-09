"""Dual-level attribution (PL-xPsPM) instantiated on SB3 PPO agents.

Ports the paper's two explanation targets from the TDQN sequence encoder to a
feature-vector PPO policy:

- **Risk** (*why is this case at risk?*): Integrated Gradients on the critic
  V(s). Same semantics as the paper's phi^V (completeness w.r.t. the value).
- **Margin** (*why act now rather than wait?*): Integrated Gradients on the
  actor's logit margin log pi(intervene|s) - log pi(wait|s). This is a
  *declared semantics change* versus the paper's Delta-Q: a policy-preference
  margin, not an expected-return difference (the paper's Def. 2 states the
  questions independently of how the targets are computed).

The fidelity tests are the paper's deletion tests ported to feature space:
masking replaces a feature with a *reference value* (the mean over the
background states, the analog of the PAD baseline), and guided / random /
anti-guided |displacement| of the target are compared. The sign-flip rate of
the margin is reported alongside, as in Def. 4's refinement. The evaluability
criterion drops the masking-granularity condition (meaningless for a handful
of named features) and keeps the sample and target-above-null conditions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

# --------------------------------------------------------------------------
# Differentiable heads over an SB3 ActorCriticPolicy
# --------------------------------------------------------------------------


def _latents(policy, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(latent_pi, latent_vf) via the policy's own extractors, differentiable."""
    features = policy.extract_features(obs)
    if isinstance(features, tuple):
        pi_features, vf_features = features
        latent_pi = policy.mlp_extractor.forward_actor(pi_features)
        latent_vf = policy.mlp_extractor.forward_critic(vf_features)
    else:
        latent_pi, latent_vf = policy.mlp_extractor(features)
    return latent_pi, latent_vf


class CriticHead(nn.Module):
    """V(s) as a differentiable function of the observation."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        _, latent_vf = _latents(self.policy, obs)
        return self.policy.value_net(latent_vf).squeeze(-1)


class MarginHead(nn.Module):
    """logit(intervene) - logit(wait) as a differentiable function of obs."""

    def __init__(self, policy, intervene_action: int = 1):
        super().__init__()
        self.policy = policy
        self.intervene_action = intervene_action

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        latent_pi, _ = _latents(self.policy, obs)
        logits = self.policy.action_net(latent_pi)
        a = self.intervene_action
        wait = 1 - a if logits.shape[-1] == 2 else 0
        return logits[..., a] - logits[..., wait]


# --------------------------------------------------------------------------
# Integrated Gradients (feature space, straight-line path from reference)
# --------------------------------------------------------------------------


def _device(head: nn.Module) -> torch.device:
    return next(head.policy.parameters()).device


def integrated_gradients(
    head: nn.Module,
    states: np.ndarray,
    reference: np.ndarray,
    n_steps: int = 128,
) -> np.ndarray:
    """IG attributions phi with sum_j phi_j = f(x) - f(reference), per state."""
    dev = _device(head)
    x = torch.from_numpy(states.astype(np.float32)).to(dev)
    ref = torch.from_numpy(reference.astype(np.float32)).to(dev).expand_as(x)
    alphas = torch.linspace(0.0, 1.0, n_steps, device=dev).view(-1, 1, 1)
    path = ref.unsqueeze(0) + alphas * (x - ref).unsqueeze(0)  # (S, N, D)
    path = path.reshape(-1, x.shape[1]).requires_grad_(True)
    out = head(path)
    grads = torch.autograd.grad(out.sum(), path)[0]
    grads = grads.reshape(n_steps, x.shape[0], x.shape[1]).mean(dim=0)
    phi = (x - ref) * grads
    return phi.detach().cpu().numpy()


# --------------------------------------------------------------------------
# Feature-deletion fidelity test
# --------------------------------------------------------------------------


@dataclass
class DeletionResult:
    k: int
    abs_guided: float
    abs_random: float
    abs_anti: float
    gap: float
    gap_se: float
    flip_guided: float | None = None
    flip_random: float | None = None
    extras: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = {
            "k": self.k,
            "abs_guided": self.abs_guided,
            "abs_random": self.abs_random,
            "abs_anti": self.abs_anti,
            "gap": self.gap,
            "gap_se": self.gap_se,
        }
        if self.flip_guided is not None:
            d["flip_guided"] = self.flip_guided
            d["flip_random"] = self.flip_random
        d.update(self.extras)
        return d


def _mask(states: np.ndarray, idx: np.ndarray, reference: np.ndarray) -> np.ndarray:
    out = states.copy()
    rows = np.arange(len(states))[:, None]
    out[rows, idx] = reference[idx]
    return out


def deletion_test(
    head: nn.Module,
    states: np.ndarray,
    phi: np.ndarray,
    reference: np.ndarray,
    k: int,
    n_random: int = 20,
    seed: int = 123,
    track_sign_flips: bool = False,
) -> DeletionResult:
    """Guided vs random vs anti-guided top-k feature deletion on |target|.

    Guided deletes the k features with largest |phi|; anti the k smallest;
    random averages n_random draws. Displacement is |f(x_masked) - f(x)|
    per state; the gap is guided-minus-random with the SE of the paired
    per-state differences.
    """
    rng = np.random.default_rng(seed)
    dev = _device(head)
    with torch.no_grad():
        f0 = head(torch.from_numpy(states.astype(np.float32)).to(dev)).cpu().numpy()

    def displacement(masked: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            fm = head(torch.from_numpy(masked.astype(np.float32)).to(dev)).cpu().numpy()
        return fm

    order = np.argsort(-np.abs(phi), axis=1)
    guided_idx, anti_idx = order[:, :k], order[:, -k:]
    f_guided = displacement(_mask(states, guided_idx, reference))
    f_anti = displacement(_mask(states, anti_idx, reference))

    d = states.shape[1]
    rand_disp = np.zeros((n_random, len(states)))
    rand_flip = np.zeros(n_random)
    for r in range(n_random):
        ridx = np.stack([rng.choice(d, size=k, replace=False) for _ in range(len(states))])
        fr = displacement(_mask(states, ridx, reference))
        rand_disp[r] = np.abs(fr - f0)
        if track_sign_flips:
            rand_flip[r] = float((np.sign(fr) != np.sign(f0)).mean())

    disp_guided = np.abs(f_guided - f0)
    disp_rand = rand_disp.mean(axis=0)
    disp_anti = np.abs(f_anti - f0)
    paired = disp_guided - disp_rand
    res = DeletionResult(
        k=k,
        abs_guided=float(disp_guided.mean()),
        abs_random=float(disp_rand.mean()),
        abs_anti=float(disp_anti.mean()),
        gap=float(paired.mean()),
        gap_se=float(paired.std(ddof=1) / np.sqrt(len(paired))),
    )
    if track_sign_flips:
        res.flip_guided = float((np.sign(f_guided) != np.sign(f0)).mean())
        res.flip_random = float(rand_flip.mean())
    return res


# --------------------------------------------------------------------------
# Full dual-level study on one agent
# --------------------------------------------------------------------------


def dual_level_study(
    model,
    states: np.ndarray,
    feature_names: list[str],
    intervene_action: int = 1,
    ks: tuple[int, ...] = (1, 2),
    n_margin_min: int = 30,
    null_factor: float = 3.0,
    seed: int = 123,
    margin_pool: np.ndarray | None = None,
    max_margin_states: int = 500,
) -> dict:
    """Run both attributions under both tests (the paper's 2x2 cross matrix).

    Risk states: all provided states. Margin states: those where the policy's
    argmax action is ``intervene_action`` (the analog of the paper's explained
    intervention cases), scanned over ``margin_pool`` when given (policies that
    intervene rarely need the full prefix pool) and capped at
    ``max_margin_states``. Evaluability requires at least ``n_margin_min`` of
    them and a mean |margin| at least ``null_factor`` times the random
    displacement at the first k.
    """
    policy = model.policy
    critic, margin = CriticHead(policy), MarginHead(policy, intervene_action)
    reference = states.mean(axis=0)

    scan = states if margin_pool is None else margin_pool
    dev = _device(margin)
    with torch.no_grad():
        m0 = margin(torch.from_numpy(scan.astype(np.float32)).to(dev)).cpu().numpy()
    margin_states = scan[m0 > 0]
    if len(margin_states) > max_margin_states:
        idx = np.random.default_rng(seed).choice(
            len(margin_states), size=max_margin_states, replace=False
        )
        margin_states = margin_states[idx]

    out: dict = {
        "n_states": len(states),
        "n_margin_states": int(len(margin_states)),
        "feature_names": feature_names,
        "reference": reference.tolist(),
        "intervene_rate": float((m0 > 0).mean()),
    }

    phi_v = integrated_gradients(critic, states, reference)
    out["phi_v_mean_abs"] = np.abs(phi_v).mean(axis=0).tolist()
    out["value_test"] = {
        "phi_v": {
            str(k): deletion_test(critic, states, phi_v, reference, k, seed=seed).as_dict()
            for k in ks
        }
    }

    margin_ok = len(margin_states) >= n_margin_min
    out["margin_evaluable_sample"] = bool(margin_ok)
    if margin_ok:
        phi_dq = integrated_gradients(margin, margin_states, reference)
        out["phi_dq_mean_abs"] = np.abs(phi_dq).mean(axis=0).tolist()
        # target-above-null condition (Def. 4 (iii) analog)
        probe = deletion_test(
            margin, margin_states, phi_dq, reference, ks[0], seed=seed, track_sign_flips=True
        )
        with torch.no_grad():
            m_sel = margin(torch.from_numpy(margin_states.astype(np.float32)).to(dev)).cpu().numpy()
        mean_abs_margin = float(np.abs(m_sel).mean())
        out["mean_abs_margin"] = mean_abs_margin
        out["margin_above_null"] = bool(mean_abs_margin >= null_factor * probe.abs_random)
        out["margin_test"] = {
            "phi_dq": {
                str(k): deletion_test(
                    margin, margin_states, phi_dq, reference, k, seed=seed, track_sign_flips=True
                ).as_dict()
                for k in ks
            },
            # cross tests: each ranking on the other target
            "phi_v_on_margin": {
                str(k): deletion_test(
                    margin,
                    margin_states,
                    integrated_gradients(critic, margin_states, reference),
                    reference,
                    k,
                    seed=seed,
                    track_sign_flips=True,
                ).as_dict()
                for k in ks
            },
        }
        out["value_test"]["phi_dq_on_value"] = {
            str(k): deletion_test(critic, margin_states, phi_dq, reference, k, seed=seed).as_dict()
            for k in ks
        }
    return out
