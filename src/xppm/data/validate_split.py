from __future__ import annotations

from pathlib import Path

from xppm.utils.logging import get_logger


logger = get_logger(__name__)


def validate_and_split(clean_log_path: str | Path, splits_path: str | Path) -> None:
    """Placeholder for 01b_validate_and_split logic.

    Intended to:
    - validate event log schema / ranges
    - create train/val/test split indices
    """
    logger.info("Validating event log and creating splits (stub).")
    Path(splits_path).parent.mkdir(parents=True, exist_ok=True)
    Path(splits_path).write_text('{"train": [], "val": [], "test": []}\n', encoding="utf-8")


