"""Export API schema to JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from xppm.serve.schemas import DecisionRequest, DecisionResponse


def export_schema(output_path: Path):
    """Exporta schema OpenAPI/JSON Schema."""
    schema = {
        "title": "XPPM Decision API",
        "version": "v1",
        "description": "Explainable Prescriptive Process Monitoring Decision Support System",
        "schemas": {
            "DecisionRequest": DecisionRequest.model_json_schema(),
            "DecisionResponse": DecisionResponse.model_json_schema(),
        },
        "examples": {
            "request": DecisionRequest.Config.json_schema_extra["example"],
            "response": DecisionResponse.Config.json_schema_extra["example"],
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(schema, f, indent=2, default=str)

    print(f"✅ Schema exported to {output_path}")
    return schema


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/deploy/v1/schema.json")
    args = parser.parse_args()

    export_schema(Path(args.output))
