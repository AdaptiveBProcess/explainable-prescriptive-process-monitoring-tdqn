"""FastAPI server for XPPM Decision Support System."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException

from xppm.serve.guard import PolicyGuard
from xppm.serve.logger import DecisionLogger
from xppm.serve.schemas import DecisionRequest, DecisionResponse, PolicyVersions

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="XPPM Decision API",
    description="Explainable Prescriptive Process Monitoring Decision Support System",
    version="1.0.0",
)

# Global state
BUNDLE_DIR = None
SURROGATE = None
GUARD = None
VERSIONS = None
ACTION_NAMES = None
LOGGER = None


def load_bundle(bundle_dir: Path):
    """Carga el deploy bundle."""
    global SURROGATE, GUARD, VERSIONS, ACTION_NAMES, FEATURE_NAMES, LOGGER

    logger.info(f"Loading bundle from {bundle_dir}")

    # Load surrogate
    tree_path = bundle_dir / "tree.pkl"
    import pickle

    with open(tree_path, "rb") as f:
        tree_data = pickle.load(f)
        SURROGATE = tree_data["model"]
        ACTION_NAMES = tree_data["action_names"]
        FEATURE_NAMES = tree_data.get("feature_names", [])

    # Critical validation: FEATURE_NAMES must be loaded
    if not FEATURE_NAMES or len(FEATURE_NAMES) == 0:
        raise RuntimeError(
            "FEATURE_NAMES not loaded from tree.pkl. "
            "Cannot proceed without feature names for mapping."
        )

    # Validate feature count matches model expectations
    expected_n_features = SURROGATE.n_features_in_
    if len(FEATURE_NAMES) != expected_n_features:
        raise ValueError(
            f"Feature count mismatch: tree.pkl has {len(FEATURE_NAMES)} feature names, "
            f"but model expects {expected_n_features} features."
        )

    logger.info(
        f"Loaded surrogate: {SURROGATE.get_n_leaves()} leaves, depth {SURROGATE.get_depth()}"
    )
    logger.info(f"Feature names loaded: {len(FEATURE_NAMES)} features")
    logger.info(f"First 5 features: {FEATURE_NAMES[:5]}")

    # Load guard
    guard_config = bundle_dir / "policy_guard_config.json"
    GUARD = PolicyGuard(guard_config)

    # Load versions
    versions_path = bundle_dir / "versions.json"
    with open(versions_path) as f:
        VERSIONS = json.load(f)

    # Setup logger
    LOGGER = DecisionLogger(bundle_dir / "decisions.jsonl")

    logger.info(f"Bundle loaded. Model version: {VERSIONS['model_version'][:8]}...")


@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    global BUNDLE_DIR
    if BUNDLE_DIR is None:
        BUNDLE_DIR = Path("artifacts/deploy/v1")  # Default, override via CLI
    load_bundle(BUNDLE_DIR)
    logger.info("Server ready")


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy", "timestamp": time.time()}


@app.get("/version")
async def version():
    """Get version info."""
    return VERSIONS


@app.get("/schema")
async def schema():
    """Get API schema."""
    schema_path = BUNDLE_DIR / "schema.json"
    with open(schema_path) as f:
        return json.load(f)


@app.post("/v1/decision", response_model=DecisionResponse)
async def decide(request: DecisionRequest):
    """
    Main decision endpoint.

    Process:
    1. Extract features
    2. Predict with surrogate
    3. Apply guard checks
    4. Return decision + explanations + versions
    """
    start_time = time.time()

    try:
        # Critical validation: FEATURE_NAMES must be available
        if FEATURE_NAMES is None or len(FEATURE_NAMES) == 0:
            raise RuntimeError(
                "FEATURE_NAMES not loaded. Server cannot process requests without feature mapping."
            )

        # Extract features as array (must match tree.pkl feature order)
        features_dict = request.features.model_dump()

        # Build feature array in the same order as training
        X_row = []
        for feat_name in FEATURE_NAMES:
            if feat_name.startswith("count_"):
                # Activity count feature
                act_name = feat_name.replace("count_", "")
                # Map activity name to feature dict key
                if act_name == "validate_application":
                    X_row.append(float(features_dict.get("count_validate_application", 0.0)))
                elif act_name == "skip_contact":
                    X_row.append(float(features_dict.get("count_skip_contact", 0.0)))
                elif act_name == "contact_headquarters":
                    X_row.append(float(features_dict.get("count_contact_headquarters", 0.0)))
                else:
                    # Other activities not in schema (<PAD>, <UNK>, email_customer, etc.) - set to 0
                    X_row.append(0.0)
            else:
                # Regular feature (amount, est_quality, etc.)
                X_row.append(float(features_dict.get(feat_name, 0.0)))

        # Critical validation: feature count must match model expectations
        if len(X_row) != len(FEATURE_NAMES):
            raise ValueError(
                f"Feature length mismatch: mapped {len(X_row)} features, "
                f"but model expects {len(FEATURE_NAMES)} features."
            )

        if len(X_row) != SURROGATE.n_features_in_:
            raise ValueError(
                f"Feature count mismatch: mapped {len(X_row)} features, "
                f"but model expects {SURROGATE.n_features_in_} features."
            )

        X = [X_row]

        # Surrogate prediction
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

        # Apply guard
        final_decision = GUARD.process(request.dict(), surrogate_decision)

        # Build response
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

        # Log decision
        logger.info(
            f"[{request.request_id}] Decision: {response.action_name} "
            f"(source={response.source}, conf={response.confidence:.2f}, "
            f"latency={latency_ms:.1f}ms)"
        )

        # Log to JSONL
        if LOGGER:
            LOGGER.log_decision(request.dict(), response.dict())

        return response

    except Exception as e:
        logger.error(f"Error processing request {request.request_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default="artifacts/deploy/v1", help="Deploy bundle directory")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    BUNDLE_DIR = Path(args.bundle)
    load_bundle(BUNDLE_DIR)

    uvicorn.run(app, host=args.host, port=args.port)
