from __future__ import annotations

import argparse
import json
from pathlib import Path

from xppm.data.preprocess import preprocess_event_log
from xppm.utils.config import Config
from xppm.utils.io import fingerprint_data, get_git_commit
from xppm.utils.logging import (
    ensure_dir,
    finalize_run_metadata,
    get_logger,
    init_tracker,
    save_run_metadata,
    start_run_metadata,
)

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess event log (Phase 1 - Step 1)")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name. Loads configs/datasets/{name}.yaml on top of --config.",
    )
    args = parser.parse_args()

    # Load config
    config_obj = Config.for_dataset(args.config, args.dataset)
    cfg = config_obj.raw

    # Setup reproducibility
    seed = cfg.get("repro", {}).get("seed", 42)
    deterministic = cfg.get("repro", {}).get("deterministic", False)

    # Get paths from config
    data_cfg = cfg.get("data", {})
    paths_cfg = cfg.get("paths", {})
    preprocess_cfg = cfg.get("preprocess", {})

    input_path = data_cfg.get("raw_path") or paths_cfg.get("data_raw", "data/raw/log.csv")
    output_dir = Path(paths_cfg.get("data_interim_dir", "data/interim"))
    output_path = output_dir / preprocess_cfg.get("out_clean_parquet", "clean.parquet")

    # Prepare preprocess config
    # Prepare schema mapping
    schema_cfg = cfg.get("schema", {})
    preprocess_config = {
        "format": data_cfg.get("format", "auto"),
        "schema": {
            "case_id": schema_cfg.get("case_id", data_cfg.get("case_id_col", "case_id")),
            "activity": schema_cfg.get("activity", data_cfg.get("activity_col", "activity")),
            "timestamp": schema_cfg.get("timestamp", data_cfg.get("timestamp_col", "timestamp")),
        },
        "time": cfg.get("time", {}),
        "preprocess": cfg.get("preprocess", {}),
        "outcome_engineering": cfg.get("outcome_engineering", {}),
    }

    # Initialize tracking
    tracking_cfg = cfg.get("tracking", {})
    tracker = None
    if tracking_cfg.get("enabled", False):
        tracker = init_tracker(tracking_cfg)
        if tracker.enabled:
            git_info = get_git_commit()
            run_name = f"preprocess_{Path(input_path).stem}"
            commit_hash = git_info["commit"]
            commit_short = commit_hash[:8] if isinstance(commit_hash, str) else "unknown"
            tags = {
                "stage": "phase1_preprocess",
                "dataset": Path(input_path).stem,
                "git_commit": commit_short,
                "git_dirty": str(git_info["dirty"]),
                "config_hash": config_obj.hash_config()[:8],
            }
            # Add raw data hash
            try:
                raw_data_hash = fingerprint_data([input_path])[:16]
                tags["raw_data_hash"] = raw_data_hash
            except Exception as e:
                logger.warning("Could not compute raw data hash: %s", e)
                tags["raw_data_hash"] = "unknown"

            params = {
                "input_path": str(input_path),
                "format": preprocess_config["format"],
                "timezone": preprocess_config.get("time", {}).get("timezone"),
                "output_timezone": preprocess_config.get("time", {}).get("output_timezone"),
            }
            tracker.init_run(run_name=run_name, stage="phase1_preprocess", tags=tags, params=params)

    # Start run metadata
    artifacts_dir = Path(paths_cfg.get("artifacts_dir", "artifacts"))
    reports_dir = artifacts_dir / "reports"
    ensure_dir(reports_dir)

    metadata = start_run_metadata(
        stage="preprocess",
        config_path=args.config,
        config_hash=config_obj.hash_config(),
        seed=seed,
        deterministic=deterministic,
        data_fingerprint=fingerprint_data([input_path]) if Path(input_path).exists() else None,
    )

    try:
        # Run preprocessing
        logger.info("Starting preprocessing: %s -> %s", input_path, output_path)
        stats = preprocess_event_log(input_path, output_path, config=preprocess_config)

        # Log metrics to tracker
        if tracker and tracker.enabled:
            # Log main metrics
            tracker.log_metrics(
                {
                    "n_cases": stats.get("n_cases", 0),
                    "n_events": stats.get("n_events", 0),
                    "missing_case_id_pct": stats.get("missing_case_id_pct", 0),
                    "missing_activity_pct": stats.get("missing_activity_pct", 0),
                    "missing_timestamp_pct": stats.get("missing_timestamp_pct", 0),
                    "case_length_mean": stats.get("case_length_mean", 0),
                    "case_length_p50": stats.get("case_length_p50", 0),
                }
            )

            # Log params
            tracker.set_tags(
                {
                    "date_min": stats.get("date_min", "N/A"),
                    "date_max": stats.get("date_max", "N/A"),
                }
            )

        # Save ingest report
        ingest_report_path = reports_dir / "ingest_report.json"
        with open(ingest_report_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, default=str)
        logger.info("Saved ingest report to %s", ingest_report_path)

        # Log artifacts to tracker
        if tracker and tracker.enabled and tracking_cfg.get("log_artifacts", True):
            tracker.log_artifact(str(output_path), artifact_path="data")
            tracker.log_artifact(str(ingest_report_path), artifact_path="reports")
            if tracking_cfg.get("log_config", True):
                tracker.log_artifact(args.config, artifact_path="config")

        # Finalize metadata
        metadata = finalize_run_metadata(
            metadata,
            outputs=[str(output_path), str(ingest_report_path)],
            metrics=stats,
        )
        metadata["status"] = "completed"

        # Save run metadata
        run_metadata_path = reports_dir / "run_metadata_preprocess.json"
        save_run_metadata(metadata, run_metadata_path)

        logger.info("✅ Preprocessing completed successfully")
        logger.info("   Output: %s", output_path)
        logger.info(
            "   Stats: %d cases, %d events",
            stats.get("n_cases", 0),
            stats.get("n_events", 0),
        )

    except Exception as e:
        logger.error("❌ Preprocessing failed: %s", e, exc_info=True)
        metadata["status"] = "failed"
        metadata["error"] = str(e)
        run_metadata_path = reports_dir / "run_metadata_preprocess.json"
        save_run_metadata(metadata, run_metadata_path)
        if tracker and tracker.enabled:
            tracker.finish()
        raise

    finally:
        if tracker and tracker.enabled:
            tracker.finish()


if __name__ == "__main__":
    main()
