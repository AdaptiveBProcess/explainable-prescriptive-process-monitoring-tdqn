"""Test deployment with real cases from distill dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from xppm.distill.distill_policy import extract_tabular_features
from xppm.utils.io import load_npz, load_parquet


def test_deployment(
    bundle_dir: Path,
    n_cases: int = 50,
    base_url: str = "http://localhost:8000",
    dataset_name: str | None = None,
):
    """Test deployment with real cases from distill dataset."""
    # Load distill selection
    distill_selection_path = bundle_dir.parent / "distill" / "final" / "distill_selection.json"
    if not distill_selection_path.exists():
        print(f"❌ Distill selection not found: {distill_selection_path}")
        return None

    with open(distill_selection_path) as f:
        selection = json.load(f)

    # Load dataset — paths differ when a named dataset is provided
    if dataset_name:
        npz_path = f"data/{dataset_name}/processed/D_offline.npz"
        parquet_path = f"data/{dataset_name}/interim/clean.parquet"
    else:
        npz_path = "data/processed/D_offline.npz"
        parquet_path = "data/interim/clean.parquet"

    dataset = load_npz(npz_path)
    clean_df = load_parquet(parquet_path)

    # Load config for feature extraction
    from xppm.utils.config import Config

    try:
        cfg = Config.from_yaml("configs/config.yaml").raw
    except Exception:
        cfg = {"encoding": {"output": {"vocab_activity_path": "data/interim/vocab_activity.json"}}}

    # Extract features for test cases
    test_indices = np.array(selection["indices"][:n_cases])
    features, _ = extract_tabular_features(dataset, clean_df, test_indices, cfg)

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

    # Test cases
    results = []
    for i, idx in enumerate(test_indices):
        case_id = str(dataset["case_ptr"][idx])
        t = int(dataset["t"][idx])

        # Build request
        request = {
            "request_id": f"test_{idx}",
            "case_id": case_id,
            "t": t,
            "features": {name: float(features[i, j]) for j, name in enumerate(feature_names)},
        }

        try:
            # Send to API
            response = requests.post(
                f"{base_url}/v1/decision",
                json=request,
                timeout=5.0,
            )

            if response.status_code == 200:
                result = response.json()
                results.append(
                    {
                        "request_id": request["request_id"],
                        "action_id": result["action_id"],
                        "source": result["source"],
                        "confidence": result["confidence"],
                        "latency_ms": result["latency_ms"],
                    }
                )
            else:
                print(f"❌ Request {request['request_id']} failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Request {request['request_id']} error: {e}")

    # Analyze
    if not results:
        print("❌ No successful requests")
        return None

    df = pd.DataFrame(results)
    print(f"\n✅ Deployment test results (n={len(df)}):")
    print(f"   Sources: {df['source'].value_counts().to_dict()}")
    print(f"   Mean confidence: {df['confidence'].mean():.3f}")
    print(f"   Mean latency: {df['latency_ms'].mean():.1f}ms")
    print(f"   p95 latency: {df['latency_ms'].quantile(0.95):.1f}ms")
    print(f"   p99 latency: {df['latency_ms'].quantile(0.99):.1f}ms")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test deployment")
    parser.add_argument(
        "--bundle-dir", default="artifacts/deploy/v1", help="Deploy bundle directory"
    )
    parser.add_argument("--n-cases", type=int, default=50, help="Number of test cases")
    parser.add_argument("--base-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset name (e.g. bpi2020-rfp). If omitted, uses SimBank defaults.",
    )
    args = parser.parse_args()

    test_deployment(Path(args.bundle_dir), args.n_cases, args.base_url, args.dataset)
