"""Instrumented wrapper around the trained TDQN policy (Paper 2, Phase 0).

White-box access to :class:`~xppm.rl.train_tdqn.TransformerQNetwork` internals
via native PyTorch forward hooks. No port to an external MI framework: the
wrapped module IS the deployed checkpoint, so numerical fidelity is guaranteed
by construction (verified in ``tests/unit/test_interp_equivalence.py``).

Architecture facts that shape this module (see ``rl/models/transformer.py``):

- No positional encoding: the encoder is permutation-equivariant; token
  content and the pooling position are the only order signals.
- Post-LN ``nn.TransformerEncoderLayer`` blocks, bidirectional attention, and
  no padding mask: PAD positions participate in attention.
- Pooling at the last non-padded position, then ``state_proj`` + ReLU and a
  linear Q-head.

The PyTorch MultiheadAttention fastpath is disabled at construction time: the
fastpath bypasses ``self_attn.forward``, which would make hooks silently miss
attention activations and patterns.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch import nn

from xppm.interp.cache import ActivationCache, hook_names

# An edit takes the original activation and returns a replacement (same shape).
EditFn = Callable[[torch.Tensor], torch.Tensor]

# Hook points whose captured tensor does not feed back into the computation.
_READ_ONLY = ("pattern", "q")


def _disable_mha_fastpath() -> None:
    """Force the slow (hookable) path of ``nn.MultiheadAttention`` globally."""
    try:
        torch.backends.mha.set_fastpath_enabled(False)
    except AttributeError:  # torch < 2.1
        pass


def _force_attention_weights(mha: nn.MultiheadAttention) -> None:
    """Make *mha* always return per-head attention weights.

    ``nn.TransformerEncoderLayer`` calls its attention with
    ``need_weights=False``; this wrapper overrides that so ``pattern_{i}``
    hook points can capture (B, H, L, L) weights. Idempotent.
    """
    if getattr(mha, "_xppm_forced_weights", False):
        return
    orig_forward = mha.forward

    def forward(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **kwargs: Any):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False
        return orig_forward(query, key, value, **kwargs)

    mha.forward = forward  # type: ignore[method-assign]
    mha._xppm_forced_weights = True  # type: ignore[attr-defined]


class HookedTDQN(nn.Module):
    """Hook-based instrumentation of a ``TransformerQNetwork``.

    Usage::

        hooked = HookedTDQN(q_net)                      # or .from_checkpoint(...)
        q, cache = hooked.run_with_cache(states, mask)  # capture activations
        q_patched = hooked.run_with_hooks(              # activation patching
            states, mask, edits={"resid_1": my_edit_fn}
        )

    Hook-point names and shapes are documented in :mod:`xppm.interp.cache`.
    """

    def __init__(self, q_net: nn.Module) -> None:
        super().__init__()
        for attr in ("embedding", "encoder", "state_proj", "q_head"):
            if not hasattr(q_net, attr):
                raise TypeError(
                    f"Expected a TransformerQNetwork-like module with '{attr}'; "
                    f"got {type(q_net).__name__}"
                )
        _disable_mha_fastpath()
        self.q_net = q_net.eval()
        for layer in self._layers:
            _force_attention_weights(layer.self_attn)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str | Path,
        npz_path: str | Path,
        vocab_path: str | Path,
        config: dict[str, Any],
        device: torch.device,
    ) -> "HookedTDQN":
        """Load a checkpoint via :class:`~xppm.rl.factory.AgentFactory` and wrap it."""
        from xppm.rl.factory import AgentFactory

        q_net = AgentFactory.load(ckpt_path, npz_path, vocab_path, config, device)
        return cls(q_net)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def _layers(self) -> nn.ModuleList:
        return self.q_net.encoder.encoder.layers

    @property
    def n_layers(self) -> int:
        return len(self._layers)

    @property
    def n_heads(self) -> int:
        return self._layers[0].self_attn.num_heads

    def hook_names(self) -> list[str]:
        return hook_names(self.n_layers)

    def editable_hook_names(self) -> list[str]:
        return [n for n in self.hook_names() if not n.startswith(_READ_ONLY)]

    def _hook_points(self) -> dict[str, tuple[nn.Module, str]]:
        """Map hook name -> (module, capture kind)."""
        q_net = self.q_net
        points: dict[str, tuple[nn.Module, str]] = {
            "embed": (q_net.embedding, "output"),
            "proj": (q_net.encoder.input_proj, "output"),
            "pooled": (q_net.state_proj, "input"),
            "state_repr": (q_net.state_proj, "output"),
            "q": (q_net.q_head, "output"),
        }
        for i, layer in enumerate(self._layers):
            points[f"attn_{i}"] = (layer.self_attn, "attn")
            points[f"pattern_{i}"] = (layer.self_attn, "pattern")
            points[f"mlp_{i}"] = (layer.linear2, "output")
            points[f"resid_{i}"] = (layer, "output")
        return points

    # ------------------------------------------------------------------
    # Forward variants
    # ------------------------------------------------------------------

    def forward(self, states: torch.Tensor, state_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Plain forward, identical to the wrapped ``TransformerQNetwork``."""
        return self.q_net(states, state_mask)

    def run_with_cache(
        self,
        states: torch.Tensor,
        state_mask: torch.Tensor | None = None,
        names: Iterable[str] | None = None,
    ) -> tuple[torch.Tensor, ActivationCache]:
        """Forward pass capturing activations.

        Args:
            states: (B, L) token ids.
            state_mask: (B, L) 1 for real tokens, 0 for PAD.
            names: subset of hook points to capture (default: all).

        Returns:
            (Q-values, :class:`ActivationCache`).
        """
        wanted = set(names) if names is not None else set(self.hook_names())
        self._validate_names(wanted, self.hook_names())
        store: dict[str, torch.Tensor] = {}
        with self._capture(store, wanted):
            q = self.q_net(states, state_mask)
        return q, ActivationCache(store)

    def run_with_hooks(
        self,
        states: torch.Tensor,
        state_mask: torch.Tensor | None = None,
        edits: dict[str, EditFn] | None = None,
    ) -> torch.Tensor:
        """Forward pass with activation edits (the patching primitive).

        Each edit receives the original activation tensor at its hook point
        and must return a tensor of the same shape, which replaces it for the
        rest of the forward pass. ``pattern_{i}`` and ``q`` are read-only:
        the returned attention weights do not feed the computation, so
        editing them would silently do nothing.
        """
        edits = edits or {}
        self._validate_names(edits, self.editable_hook_names())
        with self._apply_edits(edits):
            return self.q_net(states, state_mask)

    # ------------------------------------------------------------------
    # Hook plumbing
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_names(names: Iterable[str], allowed: Iterable[str]) -> None:
        allowed_set = set(allowed)
        unknown = sorted(set(names) - allowed_set)
        if unknown:
            raise ValueError(
                f"Unknown or non-editable hook points {unknown}. " f"Allowed: {sorted(allowed_set)}"
            )

    @contextmanager
    def _capture(self, store: dict[str, torch.Tensor], wanted: set[str]) -> Iterator[None]:
        handles = []

        def save_output(name: str):
            def hook(_m: nn.Module, _args: Any, output: torch.Tensor) -> None:
                store[name] = output.detach().clone()

            return hook

        def save_attn(out_name: str, pattern_name: str):
            def hook(_m: nn.Module, _args: Any, output: tuple) -> None:
                attn_out, weights = output
                if out_name in wanted:
                    store[out_name] = attn_out.detach().clone()
                if pattern_name in wanted and weights is not None:
                    store[pattern_name] = weights.detach().clone()

            return hook

        def save_input(name: str):
            def hook(_m: nn.Module, args: tuple) -> None:
                store[name] = args[0].detach().clone()

            return hook

        seen_attn_modules: set[int] = set()
        try:
            for name, (module, kind) in self._hook_points().items():
                if kind == "output" and name in wanted:
                    handles.append(module.register_forward_hook(save_output(name)))
                elif kind == "input" and name in wanted:
                    handles.append(module.register_forward_pre_hook(save_input(name)))
                elif kind in ("attn", "pattern"):
                    # One hook per attention module serves both attn_{i} and pattern_{i}.
                    i = name.split("_")[1]
                    if id(module) in seen_attn_modules:
                        continue
                    if f"attn_{i}" in wanted or f"pattern_{i}" in wanted:
                        handles.append(
                            module.register_forward_hook(save_attn(f"attn_{i}", f"pattern_{i}"))
                        )
                        seen_attn_modules.add(id(module))
            yield
        finally:
            for h in handles:
                h.remove()

    @contextmanager
    def _apply_edits(self, edits: dict[str, EditFn]) -> Iterator[None]:
        handles = []
        points = self._hook_points()

        def edit_output(fn: EditFn):
            def hook(_m: nn.Module, _args: Any, output: torch.Tensor) -> torch.Tensor:
                return fn(output)

            return hook

        def edit_attn(fn: EditFn):
            def hook(_m: nn.Module, _args: Any, output: tuple) -> tuple:
                attn_out, weights = output
                return fn(attn_out), weights

            return hook

        def edit_input(fn: EditFn):
            def hook(_m: nn.Module, args: tuple) -> tuple:
                return (fn(args[0]), *args[1:])

            return hook

        try:
            for name, fn in edits.items():
                module, kind = points[name]
                if kind == "output":
                    handles.append(module.register_forward_hook(edit_output(fn)))
                elif kind == "attn":
                    handles.append(module.register_forward_hook(edit_attn(fn)))
                elif kind == "input":
                    handles.append(module.register_forward_pre_hook(edit_input(fn)))
            yield
        finally:
            for h in handles:
                h.remove()


def compute_state_values(
    model: nn.Module,
    states: np.ndarray,
    state_mask: np.ndarray,
    valid_actions: np.ndarray | None = None,
    batch_size: int = 512,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """Compute V(s) = max_a Q(s, a) over (optionally masked) actions.

    Mirrors the risk target of the paper (Eq. V): invalid actions, when a
    ``valid_actions`` binary mask is given, are excluded from the max.

    Args:
        model: ``TransformerQNetwork`` or :class:`HookedTDQN`.
        states: (N, L) int token ids.
        state_mask: (N, L) binary, 1 for real tokens.
        valid_actions: optional (N, A) binary mask.
        batch_size: forward batch size.
        device: torch device.

    Returns:
        (N,) float32 array of state values.
    """
    model = model.to(device).eval()
    values = np.empty(len(states), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            end = start + batch_size
            s = torch.as_tensor(states[start:end], dtype=torch.long, device=device)
            m = torch.as_tensor(state_mask[start:end], dtype=torch.float32, device=device)
            q = model(s, m)
            if valid_actions is not None:
                mask = torch.as_tensor(valid_actions[start:end], dtype=torch.bool, device=device)
                q = q.masked_fill(~mask, float("-inf"))
            values[start:end] = q.max(dim=1).values.float().cpu().numpy()
    return values
