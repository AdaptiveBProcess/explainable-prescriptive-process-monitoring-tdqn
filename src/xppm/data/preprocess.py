from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from xppm.utils.io import save_parquet
from xppm.utils.logging import get_logger

logger = get_logger(__name__)


def load_event_log(path: str | Path, fmt: str = "auto") -> pd.DataFrame:
    """Load event log from CSV, XES, or pickle file.

    Args:
        path: Path to event log file
        fmt: Format ('csv', 'xes', 'pickle', or 'auto' to detect from extension)

    Returns:
        DataFrame with event log data
    """
    path = Path(path)
    if fmt == "auto":
        # Handle double extensions like .xes.gz
        suffixes = "".join(path.suffixes).lower()
        if path.suffix == ".csv":
            fmt = "csv"
        elif path.suffix == ".xes" or suffixes.endswith(".xes.gz"):
            fmt = "xes"
        elif path.suffix in (".pkl", ".pickle"):
            fmt = "pickle"
        else:
            raise ValueError(f"Cannot auto-detect format for {path.suffix}")

    if fmt == "csv":
        df = pd.read_csv(path)
    elif fmt == "pickle":
        # SimBank format: can be DataFrame or list of dicts
        with open(path, "rb") as f:
            data = pickle.load(f)
        # If already DataFrame, use it; otherwise convert list of dicts
        if isinstance(data, pd.DataFrame):
            df = data
        else:
            df = pd.DataFrame(data)
    elif fmt == "xes":
        try:
            import pm4py

            event_log = pm4py.read_xes(str(path))
            df = pm4py.convert_to_dataframe(event_log)
        except ImportError:
            raise ImportError("pm4py is required for XES files. Install with: pip install pm4py")
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    logger.info("Loaded event log: %d rows, %d columns", len(df), len(df.columns))
    return df


def normalize_schema(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Normalize column names to standard schema.

    Args:
        df: Input DataFrame
        mapping: Dict mapping standard names to original column names
                 e.g., {'case_id': 'case_nr', 'activity': 'activity', 'timestamp': 'timestamp'}
                 Original columns are renamed to standard names

    Returns:
        DataFrame with normalized column names
    """
    df = df.copy()
    # mapping is {standard_name: original_name}
    # We need to rename original_name -> standard_name
    rename_dict = {v: k for k, v in mapping.items() if v in df.columns}
    df = df.rename(columns=rename_dict)
    logger.debug("Renamed columns: %s", rename_dict)
    return df


def normalize_timestamps(
    df: pd.DataFrame,
    tz_in: str | None = None,
    tz_out: str | None = "UTC",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Normalize timestamps to consistent timezone.

    Args:
        df: Input DataFrame
        tz_in: Input timezone (e.g., 'America/Bogota') or None if already timezone-aware
        tz_out: Output timezone ('UTC' or None for naive)
        timestamp_col: Name of timestamp column

    Returns:
        DataFrame with normalized timestamps
    """
    if timestamp_col not in df.columns:
        return df

    df = df.copy()

    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")

    # Localize if needed
    if tz_in and df[timestamp_col].dt.tz is None:
        df[timestamp_col] = df[timestamp_col].dt.tz_localize(tz_in)

    # Convert to output timezone
    if tz_out and df[timestamp_col].dt.tz is not None:
        df[timestamp_col] = df[timestamp_col].dt.tz_convert(tz_out)
    elif tz_out is None and df[timestamp_col].dt.tz is not None:
        # Convert to naive (assume UTC)
        df[timestamp_col] = df[timestamp_col].dt.tz_localize(None)

    logger.debug("Normalized timestamps: tz_in=%s, tz_out=%s", tz_in, tz_out)
    return df


def filter_lifecycle_transitions(
    df: pd.DataFrame,
    lifecycle_col: str = "lifecycle:transition",
    keep: list[str] | None = None,
) -> pd.DataFrame:
    """Filter events to specific lifecycle transition types.

    Useful for XES logs (e.g., BPI challenge) that record both 'start' and
    'complete' lifecycle events per activity. Keeping only 'complete' gives
    one event per activity execution.

    Args:
        df: Input DataFrame
        lifecycle_col: Name of the lifecycle column (default: 'lifecycle:transition')
        keep: List of lifecycle values to keep (default: ['complete', 'COMPLETE'])

    Returns:
        Filtered DataFrame
    """
    if lifecycle_col not in df.columns:
        logger.debug("Lifecycle column '%s' not found, skipping filter", lifecycle_col)
        return df

    if keep is None:
        keep = ["complete", "COMPLETE"]

    keep_lower = {k.lower() for k in keep}
    before = len(df)
    df = df[df[lifecycle_col].str.lower().isin(keep_lower)].copy()
    dropped = before - len(df)
    logger.info(
        "Lifecycle filter: kept %s → dropped %d events, %d remaining", keep, dropped, len(df)
    )
    return df


def compute_outcome_from_activities(
    df: pd.DataFrame,
    positive_activities: list[str],
    case_id_col: str = "case_id",
    activity_col: str = "activity",
    outcome_col: str = "outcome",
) -> pd.DataFrame:
    """Compute binary outcome column from activity presence in trace.

    For each case, outcome=1.0 if any activity in positive_activities appears
    in the case trace; outcome=0.0 otherwise. The value is broadcast to all
    events of the case so downstream MDP builder can read it from the last event.

    Args:
        df: Input DataFrame (with case_id_col and activity_col)
        positive_activities: List of activities that signal a positive outcome
        case_id_col: Name of case ID column
        activity_col: Name of activity column
        outcome_col: Name of output column to create

    Returns:
        DataFrame with new outcome_col column
    """
    if not positive_activities:
        logger.warning("No positive_activities specified; %s will be 0 for all cases", outcome_col)
        df = df.copy()
        df[outcome_col] = 0.0
        return df

    positive_set = set(positive_activities)
    df = df.copy()

    # For each case, check if any positive activity appears
    case_has_positive = df.groupby(case_id_col)[activity_col].transform(
        lambda acts: 1.0 if any(a in positive_set for a in acts) else 0.0
    )
    df[outcome_col] = case_has_positive

    n_cases = df[case_id_col].nunique()
    n_positive = int(df.groupby(case_id_col)[outcome_col].first().sum())
    logger.info(
        "Outcome engineering: %d/%d cases positive (%.1f%%) → column '%s'",
        n_positive,
        n_cases,
        100.0 * n_positive / n_cases if n_cases > 0 else 0.0,
        outcome_col,
    )
    return df


def validate_min_schema(df: pd.DataFrame, required_cols: list[str] | None = None) -> None:
    """Validate that DataFrame has minimum required columns.

    Args:
        df: DataFrame to validate
        required_cols: List of required column names (default: ['case_id', 'activity', 'timestamp'])

    Raises:
        ValueError: If required columns are missing or contain nulls
    """
    if required_cols is None:
        required_cols = ["case_id", "activity", "timestamp"]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Check for nulls in required columns
    null_counts = df[required_cols].isnull().sum()
    if null_counts.any():
        null_cols = null_counts[null_counts > 0].to_dict()
        raise ValueError(f"Null values in required columns: {null_cols}")

    # Check at least 1 case and 1 event
    if len(df) == 0:
        raise ValueError("Event log is empty")
    if "case_id" in df.columns and df["case_id"].nunique() == 0:
        raise ValueError("No cases found in event log")

    logger.debug("Schema validation passed: %d rows, %d cases", len(df), df["case_id"].nunique())


def compute_ingest_stats(df: pd.DataFrame) -> dict:
    """Compute statistics about the ingested event log.

    Args:
        df: Event log DataFrame

    Returns:
        Dictionary with statistics
    """
    stats = {
        "n_events": len(df),
        "n_cases": df["case_id"].nunique() if "case_id" in df.columns else 0,
    }

    # Missing values percentage
    if "case_id" in df.columns:
        stats["missing_case_id_pct"] = (df["case_id"].isnull().sum() / len(df)) * 100
    if "activity" in df.columns:
        stats["missing_activity_pct"] = (df["activity"].isnull().sum() / len(df)) * 100
    if "timestamp" in df.columns:
        stats["missing_timestamp_pct"] = (df["timestamp"].isnull().sum() / len(df)) * 100

    # Date range
    if "timestamp" in df.columns and pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        stats["date_min"] = df["timestamp"].min().isoformat()
        stats["date_max"] = df["timestamp"].max().isoformat()
    else:
        stats["date_min"] = None
        stats["date_max"] = None

    # Case length distribution
    if "case_id" in df.columns:
        case_lengths = df.groupby("case_id").size()
        stats["case_length_min"] = int(case_lengths.min())
        stats["case_length_mean"] = float(case_lengths.mean())
        stats["case_length_p50"] = float(case_lengths.median())
        stats["case_length_p95"] = float(case_lengths.quantile(0.95))
        stats["case_length_max"] = int(case_lengths.max())

    # Top activities
    if "activity" in df.columns:
        top_activities = df["activity"].value_counts().head(10)
        stats["top_activities"] = top_activities.to_dict()

    return stats


def save_clean_log(df: pd.DataFrame, path: str | Path) -> None:
    """Save cleaned event log to parquet.

    Args:
        df: Cleaned DataFrame
        path: Output path
    """
    save_parquet(df, path)
    logger.info("Saved clean log to %s (%d rows)", path, len(df))


def preprocess_event_log(
    input_path: str | Path,
    output_path: str | Path,
    config: dict | None = None,
) -> dict:
    """Preprocess event log: load, normalize, validate, and save.

    Args:
        input_path: Path to input event log
        output_path: Path to output clean parquet
        config: Optional config dict with schema mapping, timezone, etc.

    Returns:
        Dictionary with ingest statistics
    """
    config = config or {}

    # Load
    fmt = config.get("format", "auto")
    df = load_event_log(input_path, fmt=fmt)

    # Normalize schema
    schema_mapping = config.get("schema", {})
    if schema_mapping:
        df = normalize_schema(df, schema_mapping)

    # Normalize timestamps — always use the standard column name ("timestamp")
    # because normalize_schema() has already renamed the original column.
    time_config = config.get("time", {})
    df = normalize_timestamps(
        df,
        tz_in=time_config.get("timezone"),
        tz_out=time_config.get("output_timezone", "UTC"),
        timestamp_col="timestamp",
    )

    # Filter lifecycle transitions (e.g., keep only 'complete' events in BPI XES logs)
    preprocess_filters = config.get("preprocess", {})
    lifecycle_cfg = preprocess_filters.get("lifecycle_filter", {})
    if lifecycle_cfg.get("enabled", False):
        lifecycle_col = lifecycle_cfg.get("column", "lifecycle:transition")
        keep = lifecycle_cfg.get("keep", ["complete", "COMPLETE"])
        df = filter_lifecycle_transitions(df, lifecycle_col, keep)

    # Compute binary outcome column from activity presence in trace
    outcome_eng_cfg = config.get("outcome_engineering", {})
    if outcome_eng_cfg.get("enabled", False):
        positive_activities = outcome_eng_cfg.get("positive_activities", [])
        outcome_col = outcome_eng_cfg.get("outcome_col", "outcome")
        df = compute_outcome_from_activities(df, positive_activities, outcome_col=outcome_col)

    # Encode case IDs as integer codes when they are non-integer (e.g., BPI string IDs)
    # This is required by the downstream encode_prefixes / build_mdp pipeline.
    if "case_id" in df.columns and not pd.api.types.is_integer_dtype(df["case_id"]):
        unique_cases = sorted(df["case_id"].unique(), key=str)
        case_id_map = {cid: i for i, cid in enumerate(unique_cases)}
        df["case_id"] = df["case_id"].map(case_id_map)
        logger.info(
            "Encoded %d string case IDs to integer codes 0–%d",
            len(case_id_map),
            len(case_id_map) - 1,
        )

    # Select only specified columns (useful for XES logs with hundreds of case attributes)
    select_cols = preprocess_filters.get("select_cols", None)
    if select_cols is not None:
        # Always include outcome column if engineering produced it
        cols_to_keep = list(select_cols)
        if outcome_eng_cfg.get("enabled", False):
            oc = outcome_eng_cfg.get("outcome_col", "outcome")
            if oc not in cols_to_keep:
                cols_to_keep.append(oc)
        existing = [c for c in cols_to_keep if c in df.columns]
        dropped_cols = len(df.columns) - len(existing)
        if dropped_cols > 0:
            logger.info("select_cols: keeping %d columns, dropping %d", len(existing), dropped_cols)
        df = df[existing]

    # Sort by case and timestamp
    if time_config.get("sort", True):
        sort_cols = ["case_id", "timestamp"]
        if "activity" in df.columns:
            sort_cols.append("activity")
        df = df.sort_values(sort_cols).reset_index(drop=True)

    # Drop duplicates if configured
    if time_config.get("drop_duplicates", False):
        before = len(df)
        df = df.drop_duplicates(subset=["case_id", "activity", "timestamp"], keep="first")
        dropped = before - len(df)
        if dropped > 0:
            logger.info("Dropped %d duplicate events", dropped)

    # Validate
    validate_min_schema(df)

    # Compute stats
    stats = compute_ingest_stats(df)

    # Save
    save_clean_log(df, output_path)

    return stats
