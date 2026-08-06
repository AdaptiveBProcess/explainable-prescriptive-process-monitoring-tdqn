"""
xppm — Explainable Prescriptive Process Monitoring.

Quickstart
----------
::

    from xppm import (
        EventLogSchema, LifecycleConfig, OutcomeConfig,
        preprocess_event_log,
        encode_prefixes,
        build_mdp_dataset,
        validate_and_split_dataset,
        TDQNConfig, train_tdqn,
        doubly_robust_estimate,
        explain_policy,
        distill_policy,
    )

    # 1. Describe your event log columns
    schema = EventLogSchema(
        case_id="case:concept:name",
        activity="concept:name",
        timestamp="time:timestamp",
        lifecycle=LifecycleConfig(enabled=True, keep=["complete"]),
        outcome=OutcomeConfig(
            mode="from_activity",
            positive_activities=["A_Approved"],
        ),
    )

    # 2. Preprocess
    stats = preprocess_event_log("BPI2017.xes", "clean.parquet", schema=schema)

    # 3. Encode prefixes → train TDQN → explain
    encode_prefixes("clean.parquet", "prefixes.npz", config=cfg.raw)
    build_mdp_dataset("prefixes.npz", "clean.parquet", "D_offline.npz", config=cfg.raw)
    train_tdqn(tdqn_config)
    explain_policy(cfg.raw)
"""

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------

from xppm.data.build_mdp import build_mdp_dataset
from xppm.data.encode_prefixes import encode_prefixes
from xppm.data.preprocess import preprocess_event_log
from xppm.data.schemas import (
    EventLogSchema,
    LifecycleConfig,
    MDPConfig,
    OutcomeConfig,
)
from xppm.data.validate_split import validate_and_split_dataset

# ---------------------------------------------------------------------------
# Policy distillation
# ---------------------------------------------------------------------------
from xppm.distill.distill_policy import distill_policy

# ---------------------------------------------------------------------------
# Off-Policy Evaluation
# ---------------------------------------------------------------------------
from xppm.ope.behavior_model import BehaviorPolicy, fit_behavior_policy_tdqn_encoder
from xppm.ope.doubly_robust import doubly_robust_estimate
from xppm.rl.factory import AgentFactory

# ---------------------------------------------------------------------------
# RL layer
# ---------------------------------------------------------------------------
from xppm.rl.train_tdqn import TDQNConfig, TDQNTrainer, TransformerQNetwork, train_tdqn

# ---------------------------------------------------------------------------
# Config utility
# ---------------------------------------------------------------------------
from xppm.utils.config import Config

# ---------------------------------------------------------------------------
# Explainability (XAI)
# ---------------------------------------------------------------------------
from xppm.xai.attributions import (
    aggregate_token_importance,
    compute_attributions,
    integrated_gradients_embedding,
)
from xppm.xai.explain_policy import explain_policy

# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    # Versioning
    "__version__",
    # Data — schemas
    "EventLogSchema",
    "LifecycleConfig",
    "OutcomeConfig",
    "MDPConfig",
    # Data — pipeline functions
    "preprocess_event_log",
    "encode_prefixes",
    "build_mdp_dataset",
    "validate_and_split_dataset",
    # RL
    "TDQNConfig",
    "TDQNTrainer",
    "TransformerQNetwork",
    "train_tdqn",
    "AgentFactory",
    # OPE
    "BehaviorPolicy",
    "fit_behavior_policy_tdqn_encoder",
    "doubly_robust_estimate",
    # XAI
    "compute_attributions",
    "integrated_gradients_embedding",
    "aggregate_token_importance",
    "explain_policy",
    # Distillation
    "distill_policy",
    # Config
    "Config",
]
