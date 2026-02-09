from __future__ import annotations

from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> Any:
    import json

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_event_log(df, schema_path: str | Path) -> None:
    """Validate event log DataFrame against JSON schema (stub: checks columns only)."""
    schema = load_json(schema_path)
    required = [p for p in schema.get("required", [])]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in event log: {missing}")



