from pathlib import Path


def test_eventlog_schema_exists(project_root: Path | None = None) -> None:
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]
    schema_path = project_root / "configs" / "schemas" / "event_log.schema.json"
    assert schema_path.exists()


