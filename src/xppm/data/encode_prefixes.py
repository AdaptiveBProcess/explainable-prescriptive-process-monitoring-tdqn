from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from xppm.utils.io import load_parquet, save_npz
from xppm.utils.logging import ensure_dir, get_logger

logger = get_logger(__name__)

# Special token IDs
PAD_ID = 0
UNK_ID = 1
FIRST_REAL_ID = 2


def build_vocab(
    series: pd.Series,
    min_freq: int = 1,
    max_size: int | None = None,
    add_unk: bool = True,
    add_pad: bool = True,
) -> tuple[dict[str, int], list[str], dict[str, int]]:
    """Build vocabulary from a pandas Series.

    Args:
        series: Series with tokens (e.g., activities)
        min_freq: Minimum frequency to include a token
        max_size: Maximum vocabulary size (None = no limit)
        add_unk: Whether to add UNK token
        add_pad: Whether to add PAD token

    Returns:
        Tuple of (token2id dict, id2token list, counts dict)
    """
    counts_counter = Counter(series.dropna())
    # Filter by min_freq and convert to dict
    counts: dict[str, int] = {k: v for k, v in counts_counter.items() if v >= min_freq}

    # Sort deterministically: by frequency desc, then alphabetically
    sorted_tokens = sorted(counts.items(), key=lambda x: (-x[1], x[0]))

    # Apply max_size if specified
    if max_size is not None:
        sorted_tokens = sorted_tokens[:max_size]

    # Build token2id mapping
    token2id: dict[str, int] = {}
    id2token: list[str] = []

    # Add special tokens
    if add_pad:
        token2id["<PAD>"] = PAD_ID
        id2token.append("<PAD>")
    if add_unk:
        token2id["<UNK>"] = UNK_ID
        id2token.append("<UNK>")

    # Add real tokens starting from FIRST_REAL_ID
    current_id = FIRST_REAL_ID
    for token, _ in sorted_tokens:
        token2id[token] = current_id
        id2token.append(token)
        current_id += 1

    logger.info(
        "Built vocabulary: %d tokens (pad=%s, unk=%s, real=%d)",
        len(token2id),
        add_pad,
        add_unk,
        len(token2id) - (2 if add_pad and add_unk else 1 if add_pad or add_unk else 0),
    )

    return token2id, id2token, dict(counts)


def save_vocab(
    path: str | Path,
    token2id: dict[str, int],
    id2token: list[str],
    counts: dict[str, int],
    special_tokens_meta: dict[str, Any] | None = None,
) -> None:
    """Save vocabulary to JSON file.

    Args:
        path: Output path
        token2id: Token to ID mapping
        id2token: ID to token list
        counts: Token frequency counts
        special_tokens_meta: Optional metadata about special tokens
    """
    ensure_dir(Path(path).parent)
    vocab_data = {
        "token2id": token2id,
        "id2token": id2token,
        "counts": counts,
        "num_tokens": len(token2id),
        "special_tokens": special_tokens_meta or {"pad_id": PAD_ID, "unk_id": UNK_ID},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab_data, f, indent=2)
    logger.info("Saved vocabulary to %s", path)


def encode_sequence(
    tokens: list[int],
    max_len: int,
    pad_id: int = PAD_ID,
    truncation: str = "left",
    padding: str = "left",
) -> tuple[np.ndarray, np.ndarray, int]:
    """Encode a sequence with padding and truncation.

    Args:
        tokens: List of token IDs
        max_len: Maximum sequence length
        pad_id: Padding token ID
        truncation: "left" (keep last max_len) or "right" (keep first max_len)
        padding: "left" or "right"

    Returns:
        Tuple of (encoded array, mask array, true length)
    """
    true_len = len(tokens)

    # Truncate if needed
    if true_len > max_len:
        if truncation == "left":
            tokens = tokens[-max_len:]
        else:  # right
            tokens = tokens[:max_len]
        true_len = max_len

    # Create arrays
    seq_arr = np.full(max_len, pad_id, dtype=np.int32)
    mask_arr = np.zeros(max_len, dtype=np.uint8)

    # Fill sequence
    if padding == "left":
        # Pad on left, tokens on right
        seq_arr[-true_len:] = tokens
        mask_arr[-true_len:] = 1
    else:  # right
        # Tokens on left, pad on right
        seq_arr[:true_len] = tokens
        mask_arr[:true_len] = 1

    return seq_arr, mask_arr, true_len


def generate_prefix_rows(
    df_case: pd.DataFrame,
    activity2id: dict[str, int],
    max_len: int,
    min_prefix_len: int = 1,
    truncation: str = "left",
    padding: str = "left",
    unk_id: int = UNK_ID,
) -> dict[str, list]:
    """Generate prefix rows for a single case.

    Args:
        df_case: DataFrame for one case (already sorted by timestamp)
        activity2id: Activity to ID mapping
        max_len: Maximum prefix length
        min_prefix_len: Minimum prefix length to generate
        truncation: Truncation side
        padding: Padding side
        unk_id: UNK token ID

    Returns:
        Dictionary with lists of X, mask, case_ptr, t_ptr, ts_last
    """
    case_id = df_case["case_id"].iloc[0]
    activities = df_case["activity"].tolist()
    timestamps = df_case["timestamp"].tolist()

    # Convert activities to token IDs
    activity_ids = [activity2id.get(act, unk_id) for act in activities]

    # Generate prefixes
    X_list = []
    mask_list = []
    case_ptr_list = []
    t_ptr_list = []
    ts_last_list = []

    L = len(activities)
    for t in range(min_prefix_len, L + 1):
        prefix_tokens = activity_ids[:t]
        seq_arr, mask_arr, true_len = encode_sequence(
            prefix_tokens, max_len, truncation=truncation, padding=padding
        )

        X_list.append(seq_arr)
        mask_list.append(mask_arr)
        case_ptr_list.append(case_id)
        # Use actual trace position t (not true_len which is capped at max_len).
        # For cases longer than max_len, t_ptr keeps growing while the encoded
        # sequence stays at max_len tokens (truncated). This preserves monotonicity
        # per case and allows build_mdp to correctly locate events in the trace.
        t_ptr_list.append(t)
        # Convert timestamp to Unix seconds (handle overflow for far-future dates)
        ts = timestamps[t - 1]
        if isinstance(ts, pd.Timestamp):
            try:
                ts_seconds = int(ts.value / 1e9)
            except OverflowError:
                # Use asm8.view for timestamps that overflow
                ts_seconds = int(ts.asm8.view("i8") / 1e9)
        else:
            ts_seconds = int(pd.Timestamp(ts).value / 1e9)
        ts_last_list.append(ts_seconds)

    return {
        "X": X_list,
        "mask": mask_list,
        "case_ptr": case_ptr_list,
        "t_ptr": t_ptr_list,
        "ts_last": ts_last_list,
    }


def encode_prefix_dataset(
    clean_parquet_path: str | Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Encode prefix dataset from clean parquet.

    Args:
        clean_parquet_path: Path to clean parquet file
        config: Encoding configuration dict

    Returns:
        Dictionary with statistics
    """
    # Load data
    df = load_parquet(clean_parquet_path)
    logger.info("Loaded clean log: %d rows, %d cases", len(df), df["case_id"].nunique())

    # Get config
    encoding_cfg = config.get("encoding", {})
    vocab_cfg = encoding_cfg.get("vocab", {})
    fields_cfg = encoding_cfg.get("fields", {})
    output_cfg = encoding_cfg.get("output", {})

    max_len = encoding_cfg.get("max_len", 50)
    min_prefix_len = encoding_cfg.get("min_prefix_len", 1)
    truncation = encoding_cfg.get("truncation", "left")
    padding = encoding_cfg.get("padding", "left")

    activity_col = fields_cfg.get("activity_col", "activity")

    # Build vocabulary
    activity2id, id2token, counts = build_vocab(
        df[activity_col],
        min_freq=vocab_cfg.get("min_freq", 1),
        max_size=vocab_cfg.get("max_size"),
        add_unk=vocab_cfg.get("add_unk", True),
        add_pad=vocab_cfg.get("add_pad", True),
    )

    # Save vocabulary
    vocab_path = Path(output_cfg.get("vocab_activity_path", "data/interim/vocab_activity.json"))
    save_vocab(vocab_path, activity2id, id2token, counts)

    # Generate prefixes for all cases
    all_X = []
    all_mask = []
    all_case_ptr = []
    all_t_ptr = []
    all_ts_last = []

    n_unk_tokens = 0
    n_total_tokens = 0

    for case_id, df_case in df.groupby("case_id"):
        df_case = df_case.sort_values("timestamp").reset_index(drop=True)
        prefix_data = generate_prefix_rows(
            df_case,
            activity2id,
            max_len,
            min_prefix_len=min_prefix_len,
            truncation=truncation,
            padding=padding,
        )

        all_X.extend(prefix_data["X"])
        all_mask.extend(prefix_data["mask"])
        all_case_ptr.extend(prefix_data["case_ptr"])
        all_t_ptr.extend(prefix_data["t_ptr"])
        all_ts_last.extend(prefix_data["ts_last"])

        # Count UNK tokens
        activities = df_case[activity_col].tolist()
        for act in activities:
            n_total_tokens += 1
            if act not in activity2id:
                n_unk_tokens += 1

    # Convert to numpy arrays
    X = np.array(all_X, dtype=np.int32)
    mask = np.array(all_mask, dtype=np.uint8)
    case_ptr = np.array(all_case_ptr, dtype=np.int32)
    t_ptr = np.array(all_t_ptr, dtype=np.int32)
    ts_last = np.array(all_ts_last, dtype=np.int64)

    n_prefixes = len(X)
    logger.info("Generated %d prefixes", n_prefixes)

    # Save prefixes.npz
    prefixes_path = Path(output_cfg.get("prefixes_path", "data/interim/prefixes.npz"))
    ensure_dir(prefixes_path.parent)
    save_npz(
        prefixes_path,
        X=X,
        mask=mask,
        case_ptr=case_ptr,
        t_ptr=t_ptr,
        ts_last=ts_last,
    )
    logger.info("Saved prefixes to %s", prefixes_path)

    # Compute statistics
    prefix_lengths = t_ptr.tolist()
    stats = {
        "n_cases": df["case_id"].nunique(),
        "n_events": len(df),
        "n_prefixes": n_prefixes,
        "vocab_size": len(activity2id),
        "unk_tokens_pct": (n_unk_tokens / n_total_tokens * 100) if n_total_tokens > 0 else 0.0,
        "prefix_length_min": int(min(prefix_lengths)),
        "prefix_length_mean": float(np.mean(prefix_lengths)),
        "prefix_length_p50": float(np.median(prefix_lengths)),
        "prefix_length_p95": float(np.percentile(prefix_lengths, 95)),
        "prefix_length_max": int(max(prefix_lengths)),
        "max_len": max_len,
        "padding": padding,
        "truncation": truncation,
        "top_activities": dict(Counter(df[activity_col]).most_common(10)),
    }

    return stats


def encode_prefixes(
    clean_log_path: str | Path,
    output_path: str | Path,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Encode prefixes from cleaned log (wrapper for backward compatibility).

    Args:
        clean_log_path: Path to clean parquet
        output_path: Path to output prefixes.npz
        config: Optional encoding config dict

    Returns:
        Statistics dictionary
    """
    if config is None:
        config = {"encoding": {"max_len": 50, "min_prefix_len": 1}}
    # Override output path if provided
    if output_path:
        config.setdefault("encoding", {}).setdefault("output", {})["prefixes_path"] = str(
            output_path
        )

    return encode_prefix_dataset(clean_log_path, config)
