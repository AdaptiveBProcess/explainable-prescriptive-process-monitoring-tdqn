from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Helpers (kept for backward compatibility)
# ---------------------------------------------------------------------------


def load_json(path: str | Path) -> Any:
    import json

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_event_log(df, schema_path: str | Path) -> None:
    """Validate event log DataFrame against JSON schema (checks required columns)."""
    schema = load_json(schema_path)
    required = schema.get("required", [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in event log: {missing}")


# ---------------------------------------------------------------------------
# EventLogSchema
# ---------------------------------------------------------------------------


class LifecycleConfig(BaseModel):
    """Lifecycle transition filter for XES event logs.

    XES logs record both 'start' and 'complete' events per activity.
    Enable this to keep only a subset (e.g., only 'complete' events).
    """

    enabled: bool = False
    column: str = "lifecycle:transition"
    keep: list[str] = Field(default_factory=lambda: ["complete", "COMPLETE"])


class OutcomeConfig(BaseModel):
    """How to derive the binary outcome (reward) column.

    mode='from_activity': outcome=1 if any activity in ``positive_activities``
        appears in the case trace; 0 otherwise.
    mode='from_column': the column ``column`` already exists in the log and
        will be used directly as the outcome.
    """

    mode: Literal["from_activity", "from_column"] = "from_activity"
    positive_activities: list[str] = Field(default_factory=list)
    column: str = Field("outcome", description="Source column (from_column mode) or output name.")
    output_col: str = Field("outcome", description="Name of the resulting outcome column.")


class EventLogSchema(BaseModel):
    """Column mapping and input configuration for any event log.

    Maps domain-specific column names to the standard names used throughout
    xppm (``case_id``, ``activity``, ``timestamp``). At minimum you only need
    to override the fields that differ from those defaults.

    Examples::

        # BPI 2017 (XES format)
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
        stats = preprocess_event_log("BPI2017.xes", "clean.parquet", schema=schema)

        # SimBank (pickle format — columns already use standard names)
        schema = EventLogSchema(
            case_id="case_nr",
            outcome=OutcomeConfig(mode="from_column", column="outcome"),
        )
    """

    # Core column mapping: value = original name in the log
    case_id: str = Field("case_id", description="Column that identifies a process case.")
    activity: str = Field("activity", description="Column with the activity/event type name.")
    timestamp: str = Field("timestamp", description="Column with the event timestamp.")

    # Optional columns
    resource: str | None = Field(None, description="Column for the resource/user (optional).")

    # Timezone of raw timestamps (None = assume naive/UTC)
    timezone: str | None = Field(
        None,
        description="Timezone of raw timestamps, e.g. 'Europe/Amsterdam'. None = naive/UTC.",
    )

    # Explicit column selection after normalization (None = keep all)
    select_cols: list[str] | None = Field(
        None,
        description=(
            "Keep only these columns after normalization. "
            "Useful for XES logs with hundreds of case attributes. "
            "The outcome column is always appended automatically."
        ),
    )

    # Lifecycle transition filter (common in XES)
    lifecycle: LifecycleConfig = Field(default_factory=LifecycleConfig)

    # Outcome / reward engineering
    outcome: OutcomeConfig = Field(default_factory=OutcomeConfig)

    @property
    def column_mapping(self) -> dict[str, str]:
        """Return ``{standard_name: original_name}`` dict for normalize_schema()."""
        mapping: dict[str, str] = {
            "case_id": self.case_id,
            "activity": self.activity,
            "timestamp": self.timestamp,
        }
        if self.resource is not None:
            mapping["resource"] = self.resource
        return mapping

    def to_preprocess_config(self) -> dict[str, Any]:
        """Build the ``config`` dict expected by ``preprocess_event_log()``.

        This is the bridge between the typed schema and the existing dict-based
        pipeline so both the old and new APIs work at the same time.
        """
        cfg: dict[str, Any] = {
            "schema": self.column_mapping,
            "time": {
                "timezone": self.timezone,
                "output_timezone": "UTC",
                "sort": True,
                "drop_duplicates": True,
            },
            "preprocess": {},
        }

        if self.lifecycle.enabled:
            cfg["preprocess"]["lifecycle_filter"] = {
                "enabled": True,
                "column": self.lifecycle.column,
                "keep": self.lifecycle.keep,
            }

        if self.outcome.mode == "from_activity" and self.outcome.positive_activities:
            cfg["outcome_engineering"] = {
                "enabled": True,
                "positive_activities": self.outcome.positive_activities,
                "outcome_col": self.outcome.output_col,
            }

        if self.select_cols is not None:
            cfg["preprocess"]["select_cols"] = self.select_cols

        return cfg


# ---------------------------------------------------------------------------
# MDPConfig
# ---------------------------------------------------------------------------


class MDPConfig(BaseModel):
    """MDP construction configuration.

    Defines the action space and reward structure used to build the offline
    RL dataset (``D_offline.npz``) from a preprocessed event log.

    Example::

        mdp = MDPConfig(
            actions=["do_nothing", "contact_hq"],
            noop_action="do_nothing",
            behavior_trigger_activity="contact_headquarters",
            reward_column="outcome",
        )
    """

    # Action space: index in this list = action ID used throughout the pipeline
    actions: list[str] = Field(
        default_factory=lambda: ["do_nothing", "intervene"],
        description="Action names. The list index is the integer action ID.",
    )
    noop_action: str = Field(
        "do_nothing",
        description="Name of the no-op action. Must appear in ``actions``.",
    )

    # Activity in the log that signals an intervention was taken historically
    behavior_trigger_activity: str | None = Field(
        None,
        description=(
            "Activity name in the log that corresponds to taking the intervention action. "
            "Used to reconstruct the behavioral policy from the log."
        ),
    )

    # Reward
    reward_column: str = Field(
        "outcome",
        description="Column in the clean log with the terminal reward / outcome value.",
    )
    intermediate_reward: float = Field(
        0.0,
        description="Reward assigned to non-terminal (intermediate) transitions.",
    )

    def to_mdp_config(self) -> dict[str, Any]:
        """Build the ``config['mdp']`` section expected by ``build_mdp_dataset()``.

        The returned dict can be used as ``config["mdp"]`` when constructing
        the full pipeline config.
        """
        noop_idx = self.actions.index(self.noop_action) if self.noop_action in self.actions else 0
        return {
            "actions": {
                "id2name": self.actions,
                "noop_action": self.actions[noop_idx],
            },
            "behavior_trigger_activity": self.behavior_trigger_activity,
            "reward": {
                "type": "terminal_profit_delayed",
                "terminal_column": self.reward_column,
                "intermediate_reward": self.intermediate_reward,
            },
        }
