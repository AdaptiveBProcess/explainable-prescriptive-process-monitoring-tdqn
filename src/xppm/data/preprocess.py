from __future__ import annotations

from pathlib import Path

import pandas as pd

from xppm.utils.io import load_parquet, save_parquet
from xppm.utils.logging import get_logger


logger = get_logger(__name__)


def preprocess_event_log(input_path: str | Path, output_path: str | Path) -> None:
    """Placeholder for 01_preprocess_log logic."""
    df = pd.read_csv(input_path) if str(input_path).endswith(".csv") else load_parquet(input_path)
    # TODO: implement cleaning, filtering, feature engineering
    logger.info("Preprocessing event log with %d rows", len(df))
    save_parquet(df, output_path)


