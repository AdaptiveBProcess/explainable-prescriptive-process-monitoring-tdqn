from __future__ import annotations

from pathlib import Path

import numpy as np

from xppm.utils.io import load_parquet, save_npz
from xppm.utils.logging import get_logger


logger = get_logger(__name__)


def encode_prefixes(clean_log_path: str | Path, output_path: str | Path) -> None:
    """Placeholder for 02_encode_prefixes logic."""
    df = load_parquet(clean_log_path)
    logger.info("Encoding prefixes from cleaned log with %d rows", len(df))
    # TODO: implement prefix encoding and vocab building
    dummy_prefixes = np.zeros((1, 1), dtype="int64")
    save_npz(output_path, prefixes=dummy_prefixes)


