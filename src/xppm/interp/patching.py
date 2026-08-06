"""Activation patching / causal tracing (Paper 2, Phase 2).

For each twin pair (risk, healthy), activations from the healthy run are
transplanted into the risk run one hook point x position at a time; where the
transplant recovers the healthy intervention margin, that is where the
decision causally lives. Both activations come from real logged prefixes, so
the intervention is in-distribution by construction — unlike the input-masking
fidelity tests of paper 1 (OOD critique; BPI 2020-RFP "not evaluable").

The headline comparison of the paper: per-position causal effect from this
module vs. the IG attributions phi^V / phi^DeltaQ from ``xppm.xai``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch

from xppm.interp.hooked_model import HookedTDQN

# Default actions follow configs/config.yaml: 0 = do_nothing, 1 = contact_headquarters.
DEFAULT_A0 = 0
DEFAULT_A_STAR = 1


def default_trace_hooks(n_layers: int) -> list[str]:
    """Sequence-shaped hook points swept by causal tracing, in forward order."""
    names = ["embed", "proj"]
    for i in range(n_layers):
        names += [f"attn_{i}", f"mlp_{i}", f"resid_{i}"]
    return names


def delta_q(q: torch.Tensor, a_star: int = DEFAULT_A_STAR, a0: int = DEFAULT_A0) -> float:
    """Intervention margin DeltaQ = Q(a*) - Q(a0) for a single-state batch."""
    return float((q[0, a_star] - q[0, a0]).item())


def _to_batch(state: np.ndarray, pad_id: int, device: torch.device | str):
    s = torch.as_tensor(np.asarray(state), dtype=torch.long, device=device).unsqueeze(0)
    m = (s != pad_id).float()
    return s, m


def causal_trace_pair(
    hooked: HookedTDQN,
    risk_state: np.ndarray,
    healthy_state: np.ndarray,
    *,
    pad_id: int = 0,
    hook_names: Sequence[str] | None = None,
    positions: Sequence[int] | None = None,
    a_star: int = DEFAULT_A_STAR,
    a0: int = DEFAULT_A0,
    device: torch.device | str = "cpu",
) -> pd.DataFrame:
    """Single-pair causal trace over hook points x positions.

    For every (hook, position), the healthy activation at that position is
    patched into the risk forward pass and the recovery of the intervention
    margin is measured::

        recovery = (dq_patched - dq_risk) / (dq_healthy - dq_risk)

    Args:
        hooked: instrumented model.
        risk_state: (L,) padded token ids of the risk prefix.
        healthy_state: (L,) padded token ids of the healthy twin.
        pad_id: PAD token id (used to derive the state mask).
        hook_names: hook points to sweep (default:
            :func:`default_trace_hooks`).
        positions: sequence positions to sweep (default: all L positions).
        a_star: index of the best non-null action.
        a0: index of the do-nothing action.
        device: torch device.

    Returns:
        Long DataFrame: ``hook``, ``position``, ``dq_patched``, ``recovery``,
        plus constant columns ``dq_risk`` and ``dq_healthy``.
    """
    names = list(hook_names) if hook_names is not None else default_trace_hooks(hooked.n_layers)
    risk_s, risk_m = _to_batch(risk_state, pad_id, device)
    healthy_s, healthy_m = _to_batch(healthy_state, pad_id, device)

    with torch.no_grad():
        q_healthy, clean_cache = hooked.run_with_cache(healthy_s, healthy_m, names=names)
        dq_healthy = delta_q(q_healthy, a_star, a0)
        dq_risk = delta_q(hooked(risk_s, risk_m), a_star, a0)

    seq_len = risk_s.size(1)
    pos_sweep = list(positions) if positions is not None else list(range(seq_len))
    denom = dq_healthy - dq_risk

    rows = []
    for name in names:
        clean = clean_cache[name]
        for pos in pos_sweep:

            def edit(activation: torch.Tensor, _clean=clean, _pos=pos) -> torch.Tensor:
                patched = activation.clone()
                patched[:, _pos, :] = _clean[:, _pos, :]
                return patched

            with torch.no_grad():
                q = hooked.run_with_hooks(risk_s, risk_m, edits={name: edit})
            dq_patched = delta_q(q, a_star, a0)
            rows.append(
                {
                    "hook": name,
                    "position": pos,
                    "dq_patched": dq_patched,
                    "recovery": (dq_patched - dq_risk) / denom if denom != 0.0 else np.nan,
                }
            )

    out = pd.DataFrame(rows)
    out["dq_risk"] = dq_risk
    out["dq_healthy"] = dq_healthy
    return out


def aggregate_traces(traces: pd.DataFrame) -> pd.DataFrame:
    """Mean/std recovery per (hook, position) across pairs (expects a ``pair_id`` column)."""
    grouped = traces.groupby(["hook", "position"])["recovery"]
    out = grouped.agg(recovery_mean="mean", recovery_std="std", n="count").reset_index()
    return out.sort_values(["hook", "position"]).reset_index(drop=True)


def patch_heads(*_args, **_kwargs):
    """Per-head patching — Phase 3 (circuit analysis), not yet implemented.

    ``nn.MultiheadAttention`` only exposes the head-concatenated output after
    ``out_proj``; isolating one head requires decomposing the attention block
    (recompute per-head outputs from ``pattern_{i}`` and the value projection,
    then remix through ``out_proj`` slices). Planned deliverable of the
    circuit phase; see plan §Fase 3 and the IOI / path-patching reference
    repos.
    """
    raise NotImplementedError("Per-head patching arrives with Phase 3 (circuit analysis).")
