# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

xPPM-TDQN: Explainable Prescriptive Process Monitoring using Transformer-based Double Q-Network with offline reinforcement learning. An academic research pipeline for learning intervention policies from event logs, evaluating them off-policy, and generating explainable recommendations.

The codebase is in Spanish/English mixed documentation. Config and code are English; setup docs and comments are often in Spanish.

## Commands

### Installation
```bash
pip install -e .[dev]          # basic install with dev tools
pip install -e .[dev,tracking] # also install mlflow/wandb
```

### Pipeline (sequential, each step depends on previous)
```bash
make preprocess    # 01: event log -> clean.parquet
make build_rlset   # 02+03: encode prefixes + build MDP dataset (D_offline.npz)
# Then manually: python scripts/01b_validate_and_split.py --config configs/config.yaml --overwrite
make train         # 04: train TDQN offline -> Q_theta.ckpt
make ope           # 05: off-policy evaluation (doubly robust)
make xai           # 06: generate explanations
make distill       # 08: distill to decision tree
make serve         # run FastAPI policy server
```

All scripts accept `--config configs/config.yaml`. Some require `--overwrite` to regenerate existing outputs, and support `--dry-run` for validation without execution.

### Testing & Code Quality
```bash
pytest                         # run tests (slow tests excluded by default via addopts)
pytest -m slow                 # run only slow tests
pytest -m "not slow" -v        # verbose non-slow tests
pytest tests/unit/test_X.py    # run a single test file
ruff check .                   # lint
ruff check --fix .             # lint with auto-fix
mypy src                       # type check
pre-commit run --all-files     # all pre-commit hooks (ruff, formatting, yaml)
```

### DVC Pipeline
```bash
dvc repro    # rerun full pipeline based on dependency graph
dvc pull     # fetch versioned data from remote
dvc push     # upload artifacts to remote
```

## Architecture

### Two-layer structure: library (`src/xppm/`) vs CLI (`scripts/`)

- `src/xppm/` — importable library with all logic, organized by domain
- `scripts/01-18` — thin CLI entrypoints numbered to match the paper's phases, each calling into `src/xppm/`

### Pipeline phases (data flows top-to-bottom)

**Phase 1 — Data to Offline RL Dataset:**
`Event Log` → `preprocess` → `clean.parquet` → `encode_prefixes` → `prefixes.npz` + `vocab_activity.json` → `build_mdp` → `D_offline.npz` → `validate_and_split` → `splits.json`

**Phase 2 — Training & Evaluation:**
`D_offline.npz` + `splits.json` → `train_tdqn` → `Q_theta.ckpt` → `ope_dr` → `ope_dr.json`

**Phase 3 — Explainability & Deployment:**
`Q_theta.ckpt` → `explain_policy` (risk attributions, delta-Q, policy summary) → `fidelity_tests` → `distill_policy` (VIPER → decision tree) → `export_schema` → `build_deploy_bundle` → `test_deployment` → `policy_server` (FastAPI + guard)

**Phase 4 — Production Monitoring (scripts 13-18):**
`decisions.jsonl` → `compute_monitoring_metrics` → `detect_drift` → `send_alerts` → `consolidate_feedback` → `check_retraining_triggers` → `generate_dashboard`

### Complete scripts inventory

| Script | Purpose | Output |
|--------|---------|--------|
| `01_preprocess_log.py` | Clean event log | `data/interim/clean.parquet`, `artifacts/reports/ingest_report.json` |
| `01b_validate_and_split.py` | Train/val/test split | `data/processed/splits.json`, `artifacts/reports/split_report.json` |
| `02_encode_prefixes.py` | Tokenize prefixes | `data/interim/prefixes.npz`, `data/interim/vocab_activity.json` |
| `03_build_mdp_dataset.py` | Build MDP tuples | `data/processed/D_offline.npz`, `artifacts/reports/mdp_build_report.json` |
| `04_train_tdqn_offline.py` | Train TDQN | `artifacts/models/tdqn/{run_id}/Q_theta.ckpt`, `target_Q.ckpt`, `train_history.json` |
| `05_run_ope_dr.py` | Off-policy eval | `artifacts/ope/ope_dr.json` |
| `06_explain_policy.py` | Generate explanations | `artifacts/xai/{risk,deltaQ,ig_grad,policy_summary,explanations_selection}` |
| `07_fidelity_tests.py` | Q-drop, action-flip | `artifacts/fidelity/fidelity.csv` |
| `08_distill_policy.py` | VIPER distillation | `artifacts/distill/final/{tree.pkl, tree_rules.txt, rules.sql, fidelity_metrics.json}` |
| `09_export_schema.py` | API schema | `artifacts/deploy/v1/schema.json` |
| `10_build_deploy_bundle.py` | Bundle artifacts | `artifacts/deploy/v1/` (tree, metadata, fidelity, xai, versions) |
| `11_test_deployment.py` | Smoke test bundle | reads from `artifacts/deploy/v1/` |
| `13_compute_monitoring_metrics.py` | Daily metrics | `artifacts/monitoring/metrics_{date}.json`, `daily_metrics.csv` |
| `14_detect_drift.py` | Drift detection | `artifacts/monitoring/drift_report.json` |
| `15_send_alerts.py` | Alert dispatch | sends alerts based on monitoring |
| `16_consolidate_feedback_to_offline_dataset.py` | Feedback loop | consolidated offline dataset |
| `17_check_retraining_triggers.py` | Retrain triggers | retraining ticket JSONs |
| `18_generate_dashboard.py` | Dashboard | monitoring dashboard |
| `19_generate_results_figures.py` | Paper figures | Q-drop, distillation, latency plots |
| `generate_fidelity_plots.py` | Fidelity plots | `artifacts/fidelity/{q_drop,action_flip,rank_consistency}_final.png` |
| `policy_server.py` (scripts/) | Server wrapper | imports `xppm.serving.server` |

Root-level `policy_server.py` is the full FastAPI implementation (uses `xppm.serve.*`).

### Key modules in `src/xppm/`

- **`data/`** — `preprocess.py`, `encode_prefixes.py`, `build_mdp.py`, `validate_split.py`, `schemas.py`
- **`rl/`** — `train_tdqn.py` (training loop), `replay.py` (offline buffer), `evaluation.py`
- **`rl/models/`** — `transformer.py` (SimpleTransformerEncoder), `q_network.py` (MLP Q-head), `masking.py` (action mask → -inf)
- **`ope/`** — `behavior_model.py` (frozen TDQN encoder + softmax head), `doubly_robust.py` (DR estimator + bootstrap CIs), `baselines.py`, `report.py`
- **`xai/`** — `attributions.py` (integrated gradients), `explain_policy.py` (delta-Q contrastive), `fidelity_tests.py` (Q-drop, action-flip, rank-consistency), `policy_summary.py` (clustering)
- **`distill/`** — `distill_policy.py` (DecisionTreeClassifier on stratified sample), `export_rules.py` (SQL export)
- **`serve/`** — `guard.py` (uncertainty + OOD via z-score univariate), `logger.py` (decision logging), `schemas.py` (request/response models) — **active implementation**
- **`serving/`** — `server.py`, `guard.py`, `schemas.py` — **legacy minimal wrappers**
- **`utils/`** — `config.py` (SHA-256 hashing), `io.py` (parquet/npz/json/yaml), `seed.py`, `logging.py`

### Configuration

All behavior is driven by `configs/config.yaml` (~295 lines). Key sections: `schema` (column mapping), `encoding` (max_len=50, vocab), `mdp` (actions: do_nothing/contact_headquarters, action masks by activity, terminal delayed reward), `training` (Double-DQN, cosine LR scheduler, batch_size=128), `ope`, `xai`, `fidelity`, `distill`, `serving`.

Hyperparameter defaults for DVC are in `params.yaml`. Sweep templates in `configs/sweeps/`.

### MDP Dataset Format (`D_offline.npz`)

- `s`: state prefix token IDs `[N, max_len=50]`
- `a`: action taken (0=do_nothing, 1=contact_headquarters)
- `r`: delayed terminal reward (0.0 intermediate, outcome at final state)
- `s_next`, `valid_actions` (binary mask), `propensity` (-1.0 placeholder until OPE step estimates it)

### Model Architecture

Token embedding (+ learned positional embedding) → TransformerEncoder (d_model=128, 4 heads,
3 layers, PAD masked out of attention) → pool at the last real event → MLP Q-head → Q-values per
action. Action masking sets invalid actions to -inf before argmax. Stabilized with Double-DQN,
target network (updated every 2000 steps), gradient clipping (norm=10), Huber loss.

**Encoder versions.** `TransformerQNetwork(encoder_version=...)`, recorded in every checkpoint:

- **v2** (default, `training.encoder_version: 2`): learned positional embeddings, attention padding
  mask, pooling at the last real event.
- **v1** (all checkpoints trained before this change): none of the three. The encoder is
  permutation-equivariant, so the state is a *multiset* of activities — event order informs neither
  the policy nor its attributions, and two occurrences of one activity are indistinguishable. Its
  pooling index `mask.sum()-1` lands in the PAD region under left padding, which was harmless only
  because every PAD position then carried the same representation.

`infer_encoder_version()` reads the version from the checkpoint (falling back to the presence of
`pos_embedding.weight`), so old checkpoints keep reproducing their trained behaviour exactly.
Loaders raise instead of silently accepting a mismatch. Anything that mirrors the pooling rule must
delegate to `q_net._pool_index` / `gather_last_position(..., q_net=...)` rather than reimplement it.

**Chain-version guards (Fase 1).** Checkpoint pins propagate silently: `05_run_ope_dr.py` writes
the checkpoint path into `ope_dr.json` metadata, and downstream paper scripts resolve it from
there. Three guards prevent v1/v2 mixing during regeneration:

- `05_run_ope_dr.py` has **no hardcoded default checkpoint** — it requires `--ckpt` or
  `experiment.checkpoint_path` and fails loudly otherwise.
- `export XPPM_EXPECT_ENCODER_VERSION=2` before regenerating: the canonical loader
  (`factory.load_q_network`, used by every script) then aborts on any checkpoint of a different
  version, wherever its path came from. Unset, legacy v1 checkpoints load normally.
- `python scripts/34_audit_chain_versions.py --expect 2` walks every checkpoint reference in
  configs and `artifacts/**/*.json` and reports which checkpoint/version produced each artifact.
  Run it after every regeneration; `--expect` makes it exit 1 on stale pins.

Do not point configs at `artifacts/checkpoints/Q_theta.ckpt` — that mutable alias has held
checkpoints from different datasets over time and is unauditable.

## Code Style

- **Ruff**: line-length=100, target Python 3.10, select rules E/F/I
- **MyPy**: Python 3.10, ignore_missing_imports=True
- **Pre-commit**: ruff (lint+format), end-of-file-fixer, trailing-whitespace, check-yaml, max file 10MB
- Slow tests are marked with `@pytest.mark.slow` and excluded by default
