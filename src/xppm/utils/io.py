from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f)


def load_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_npz(path: str | Path) -> dict[str, Any]:
    return dict(np.load(path, allow_pickle=True))


def save_npz(path: str | Path, **arrays: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def get_git_commit() -> dict[str, str | bool]:
    """Get current git commit hash and dirty status.
    
    Returns:
        dict with 'commit' (hash string) and 'dirty' (bool)
    """
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        # Check if working directory is dirty
        try:
            subprocess.check_output(
                ["git", "diff", "--quiet"], stderr=subprocess.DEVNULL
            )
            dirty = False
        except subprocess.CalledProcessError:
            dirty = True
        return {"commit": commit, "dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": "unknown", "dirty": True}


def fingerprint_data(paths: list[str | Path], use_dvc: bool = True) -> str:
    """Generate fingerprint for data files.
    
    Args:
        paths: List of data file paths
        use_dvc: If True, try to use DVC hash; otherwise use file sha256
    
    Returns:
        Combined hash string
    """
    if use_dvc:
        try:
            # Try to get DVC hash from dvc.lock or .dvc files
            dvc_lock = Path("dvc.lock")
            if dvc_lock.exists():
                with open(dvc_lock, "rb") as f:
                    content = f.read()
                return hashlib.sha256(content).hexdigest()[:16]
        except Exception:
            pass
    
    # Fallback: hash file contents
    hasher = hashlib.sha256()
    for path in paths:
        p = Path(path)
        if p.exists():
            with open(p, "rb") as f:
                # Read in chunks to handle large files
                while chunk := f.read(8192):
                    hasher.update(chunk)
    return hasher.hexdigest()[:16]


def dvc_out_hash(dvc_file: str | Path) -> str:
    """Read DVC hash from .dvc file.
    
    Args:
        dvc_file: Path to .dvc file (e.g., "data/interim/clean.parquet.dvc")
    
    Returns:
        MD5/hash string from DVC, or "UNKNOWN" if not found
    """
    try:
        import yaml
        
        dvc_path = Path(dvc_file)
        if not dvc_path.exists():
            return "UNKNOWN"
        
        with open(dvc_path, "r", encoding="utf-8") as f:
            dvc_data = yaml.safe_load(f)
        
        # DVC structure: {"outs": [{"path": "...", "md5": "...", ...}]}
        if "outs" in dvc_data and len(dvc_data["outs"]) > 0:
            out = dvc_data["outs"][0]
            return out.get("md5") or out.get("etag") or out.get("hash") or "UNKNOWN"
        
        return "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def get_dvc_hashes(data_paths: dict[str, str]) -> dict[str, str]:
    """Get DVC hashes for multiple data files.
    
    Args:
        data_paths: Dict mapping names to .dvc file paths
                   e.g., {"clean_parquet": "data/interim/clean.parquet.dvc"}
    
    Returns:
        Dict mapping names to DVC hashes
    """
    return {name: dvc_out_hash(path) for name, path in data_paths.items()}


