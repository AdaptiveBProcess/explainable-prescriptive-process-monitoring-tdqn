"""Build deployment bundle with all artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


def compute_hash(file_path: Path) -> str:
    """Compute SHA256 hash of file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_git_commit() -> str:
    """Get current git commit."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def extract_feature_stats(distill_selection_path: Path) -> dict:
    """Extract feature statistics from distill selection for OOD detection."""
    import json

    import numpy as np

    from xppm.distill.distill_policy import extract_tabular_features
    from xppm.utils.config import Config
    from xppm.utils.io import load_npz, load_parquet

    try:
        with open(distill_selection_path) as f:
            selection = json.load(f)

        # Load dataset
        dataset = load_npz("data/processed/D_offline.npz")
        clean_df = load_parquet("data/interim/clean.parquet")

        # Load config for feature extraction
        try:
            cfg = Config.from_yaml("configs/config.yaml").raw
        except Exception:
            # Fallback minimal config
            cfg = {
                "encoding": {"output": {"vocab_activity_path": "data/interim/vocab_activity.json"}}
            }

        # Extract features for selected indices
        indices = np.array(selection["indices"])
        features, _ = extract_tabular_features(dataset, clean_df, indices, cfg)

        # Compute stats
        feature_names = [
            "amount",
            "est_quality",
            "unc_quality",
            "cum_cost",
            "elapsed_time",
            "prefix_len",
            "count_validate_application",
            "count_skip_contact",
            "count_contact_headquarters",
        ]

        stats = {}
        for i, name in enumerate(feature_names):
            values = features[:, i]
            stats[name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }

        return stats
    except Exception as e:
        print(f"⚠️  Could not extract feature stats: {e}")
        return {}


def build_deploy_bundle(
    distill_dir: Path,
    xai_dir: Path,
    fidelity_path: Path,
    config_path: Path,
    output_dir: Path,
):
    """
    Build deploy bundle with all necessary artifacts.

    Bundle structure:
        output_dir/
        ├── schema.json
        ├── policy_guard_config.json
        ├── tree.pkl
        ├── rules_metadata.json
        ├── fidelity.csv
        ├── xai/
        │   ├── policy_summary.json
        │   ├── risk_explanations.json
        │   └── deltaQ_explanations.json
        └── versions.json
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy artifacts
    print("📦 Building deploy bundle...")

    # 1. Surrogate
    shutil.copy(distill_dir / "tree.pkl", output_dir / "tree.pkl")
    shutil.copy(distill_dir / "rules_metadata.json", output_dir / "rules_metadata.json")
    print("✅ Copied surrogate")

    # 2. Fidelity
    if fidelity_path.exists():
        shutil.copy(fidelity_path, output_dir / "fidelity.csv")
        print("✅ Copied fidelity report")
    else:
        print("⚠️  Fidelity CSV not found, skipping")

    # 3. XAI
    xai_out = output_dir / "xai"
    xai_out.mkdir(exist_ok=True)
    xai_files = [
        "policy_summary.json",
        "risk_explanations.json",
        "deltaQ_explanations.json",
    ]
    for file in xai_files:
        src = xai_dir / file
        if src.exists():
            shutil.copy(src, xai_out / file)
    print("✅ Copied XAI artifacts")

    # 4. Versions
    versions = {
        "model_version": compute_hash(distill_dir / "tree.pkl"),
        "surrogate_version": compute_hash(distill_dir / "tree.pkl"),
        "data_version": (
            compute_hash(Path("data/processed/D_offline.npz"))
            if Path("data/processed/D_offline.npz").exists()
            else "unknown"
        ),
        "config_version": compute_hash(config_path),
        "git_commit": get_git_commit(),
        "deployed_at": datetime.utcnow().isoformat(),
    }

    with open(output_dir / "versions.json", "w") as f:
        json.dump(versions, f, indent=2)
    print("✅ Created versions.json")

    # 5. Guard config (extract stats from distill dataset)
    guard_config_path = output_dir / "policy_guard_config.json"
    if not guard_config_path.exists():
        print("📊 Creating policy_guard_config.json...")
        distill_selection_path = distill_dir / "distill_selection.json"
        feature_stats = extract_feature_stats(distill_selection_path)

        template = {
            "tau_uncertainty": 0.3,
            "tau_ood_z": 3.0,
            "max_ood_features": 2,
            "fallback_action": {"action_id": 0, "action_name": "do_nothing"},
            "feature_stats": feature_stats,
        }
        with open(guard_config_path, "w") as f:
            json.dump(template, f, indent=2)
        print("✅ Created policy_guard_config.json with feature stats")
    else:
        print("✅ Using existing policy_guard_config.json")

    print(f"\n✅ Deploy bundle ready at {output_dir}")
    print(f"   Model version: {versions['model_version'][:16]}...")
    print(f"   Git commit: {versions['git_commit'][:8]}")

    return versions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build deployment bundle")
    parser.add_argument(
        "--distill-dir",
        default="artifacts/distill/final",
        help="Distill directory",
    )
    parser.add_argument("--xai-dir", default="artifacts/xai", help="XAI directory")
    parser.add_argument(
        "--fidelity",
        default="artifacts/fidelity/fidelity.csv",
        help="Fidelity CSV",
    )
    parser.add_argument("--config", default="configs/config.yaml", help="Config file")
    parser.add_argument("--output-dir", default="artifacts/deploy/v1", help="Output directory")

    args = parser.parse_args()

    build_deploy_bundle(
        Path(args.distill_dir),
        Path(args.xai_dir),
        Path(args.fidelity),
        Path(args.config),
        Path(args.output_dir),
    )
