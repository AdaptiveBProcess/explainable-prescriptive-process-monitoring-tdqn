"""Mechanistic interpretability of the TDQN policy (Paper 2).

Phase 0: ``hooked_model`` (instrumentation), ``cache`` (activation cache),
``pairs`` (counterfactual twin pairs). Phase 1: ``probing``. Phase 2:
``patching``. Phase 4: ``sae``. See
``5-escribir-paper/bpm/When_One_Step_Is_Not_Enough/plan-de-accion-mi-pspm.md``.
"""

from xppm.interp.cache import ActivationCache, gather_last_position
from xppm.interp.hooked_model import HookedTDQN, compute_state_values
from xppm.interp.pairs import build_twin_pairs

__all__ = [
    "ActivationCache",
    "HookedTDQN",
    "build_twin_pairs",
    "compute_state_values",
    "gather_last_position",
]
