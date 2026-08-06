"""FastAPI app for the XPPM Decision Support System."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException

from xppm_serve.guard import PolicyGuard
from xppm_serve.logger import DecisionLogger
from xppm_serve.schemas import DecisionRequest, DecisionResponse, PolicyVersions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="XPPM Decision API",
    description="Explainable Prescriptive Process Monitoring Decision Support System",
    version="1.0.0",
)

BUNDLE_DIR: Path | None = None
SURROGATE = None
GUARD = None
VERSIONS = None
ACTION_NAMES = None
LOGGER = None
FEATURE_NAMES: list[str] = []


def set_bundle_dir(path: Path) -> None:
    """Set bundle directory before uvicorn starts (CLI usage)."""
    global BUNDLE_DIR
    BUNDLE_DIR = path


def load_bundle(bundle_dir: Path) -> None:
    """Load all bundle artifacts. Assigns globals atomically: either all succeed or none change."""
    global SURROGATE, GUARD, VERSIONS, ACTION_NAMES, FEATURE_NAMES, LOGGER

    logger.info("Loading bundle from %s", bundle_dir)

    import pickle

    # --- Collect all values locally; validate before touching global state ---

    tree_path = bundle_dir / "tree.pkl"
    with open(tree_path, "rb") as f:
        tree_data = pickle.load(f)
    _surrogate = tree_data["model"]
    _action_names = tree_data["action_names"]
    _feature_names: list[str] = tree_data.get("feature_names", [])

    if not _feature_names:
        raise RuntimeError(
            "FEATURE_NAMES not found in tree.pkl. "
            "Rebuild the deploy bundle with scripts/10_build_deploy_bundle.py."
        )

    expected_n_features = _surrogate.n_features_in_
    if len(_feature_names) != expected_n_features:
        raise ValueError(
            f"Feature count mismatch: tree.pkl has {len(_feature_names)} feature names "
            f"but model expects {expected_n_features}."
        )

    _guard = PolicyGuard(bundle_dir / "policy_guard_config.json")

    versions_path = bundle_dir / "versions.json"
    with open(versions_path) as f:
        _versions = json.load(f)

    _decision_logger = DecisionLogger(bundle_dir / "decisions.jsonl")

    # --- All validation passed: assign atomically ---
    SURROGATE = _surrogate
    ACTION_NAMES = _action_names
    FEATURE_NAMES = _feature_names
    GUARD = _guard
    VERSIONS = _versions
    LOGGER = _decision_logger

    logger.info(
        "Loaded surrogate: %d leaves, depth %d",
        SURROGATE.get_n_leaves(),
        SURROGATE.get_depth(),
    )
    logger.info("Feature names: %d total, first 5: %s", len(FEATURE_NAMES), FEATURE_NAMES[:5])
    logger.info("Bundle loaded. Model version: %s...", VERSIONS["model_version"][:8])


@app.on_event("startup")
async def startup_event() -> None:
    global BUNDLE_DIR
    if BUNDLE_DIR is None:
        BUNDLE_DIR = Path("artifacts/deploy/v1")
    load_bundle(BUNDLE_DIR)
    logger.info("Server ready")


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy", "timestamp": time.time()}


@app.get("/version")
async def version() -> dict:
    return VERSIONS


@app.get("/schema")
async def schema() -> dict:
    schema_path = BUNDLE_DIR / "schema.json"
    with open(schema_path) as f:
        return json.load(f)


@app.post("/v1/decision", response_model=DecisionResponse)
async def decide(request: DecisionRequest) -> DecisionResponse:
    start_time = time.time()
    try:
        features_dict = request.features.model_dump()

        # Build feature array in the same order as training.
        # FEATURE_NAMES comes directly from tree.pkl so the keys match features_dict.
        X_row = [float(features_dict.get(feat_name, 0.0)) for feat_name in FEATURE_NAMES]

        if len(X_row) != SURROGATE.n_features_in_:
            raise ValueError(
                f"Feature count mismatch: mapped {len(X_row)} features "
                f"but model expects {SURROGATE.n_features_in_}."
            )

        X = [X_row]
        action_id = int(SURROGATE.predict(X)[0])
        proba = SURROGATE.predict_proba(X)[0]
        confidence = float(proba[action_id])
        uncertainty = 1.0 - confidence

        surrogate_decision = {
            "action_id": action_id,
            "action_name": ACTION_NAMES[action_id],
            "confidence": confidence,
            "uncertainty": uncertainty,
        }

        final_decision = GUARD.process(request.dict(), surrogate_decision)
        latency_ms = (time.time() - start_time) * 1000

        response = DecisionResponse(
            request_id=request.request_id,
            case_id=request.case_id,
            t=request.t,
            action_id=final_decision["action_id"],
            action_name=final_decision["action_name"],
            source=final_decision["source"],
            confidence=final_decision.get("confidence", confidence),
            uncertainty=final_decision.get("uncertainty", uncertainty),
            ood=final_decision.get("ood", False),
            versions=PolicyVersions(**VERSIONS),
            latency_ms=latency_ms,
        )

        logger.info(
            "[%s] Decision: %s (source=%s, conf=%.2f, latency=%.1fms)",
            request.request_id,
            response.action_name,
            response.source,
            response.confidence,
            latency_ms,
        )

        if LOGGER:
            LOGGER.log_decision(request.dict(), response.dict())

        return response

    except Exception as e:
        logger.error("Error processing request %s: %s", request.request_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
