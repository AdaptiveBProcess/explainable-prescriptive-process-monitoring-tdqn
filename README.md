# xPPM

Pipeline for policy-level explainable prescriptive process monitoring using offline reinforcement learning and Integrated Gradients.

---

## Overview

This repository implements the PL-xPsPM framework: a sequential offline intervention policy (Transformer-based Double Q-Network) combined with a dual-level attribution module that separates **why a case is at risk** ($\phi^V$ on $V(s)$) from **why a specific intervention is expected to improve outcomes** ($\phi^{\Delta Q}$ on $\Delta Q$), both computed via Integrated Gradients.

**Design principles**

- `src/xppm/` — reusable library (data, RL, OPE, XAI, distillation, serving).
- `scripts/` — thin CLI entrypoints (`01`–`08` + `policy_server.py`).
- Config-driven: `configs/config.yaml` + `params.yaml`.
- Reproducible pipelines: `dvc.yaml` (data → RL dataset → training → OPE → XAI).

---

## Reproducing the paper

**Start at [`docs/paper/MAPPING.md`](docs/paper/MAPPING.md)** — it maps every
number in the ICPM paper to its artifact and generator script, and lists what
is historical/non-citable. `make paper-tables` regenerates every cited JSON
(OPE for the 10 configurations, both estimators, plus the fidelity /
evaluability / statistics artifacts); `python scripts/34_audit_chain_versions.py
--expect 2` verifies no stale checkpoint is referenced.

⚠️ The operational pipeline's `07_fidelity_tests.py` output
(`artifacts/fidelity/fidelity.csv`) uses the original **signed** Q-drop
convention and does **not** back the paper's Table 2 — the paper's fidelity
numbers come from scripts 23–26/28/32 under the |displacement| convention of
its Defs. 3–4. See MAPPING.md's "Historical / non-citable" list before
comparing anything against the paper.

---

## Quick start

```bash
pip install -e .

python scripts/01_preprocess_log.py    --config configs/config.yaml
python scripts/02_encode_prefixes.py   --config configs/config.yaml
python scripts/03_build_mdp_dataset.py --config configs/config.yaml
python scripts/04_train_tdqn_offline.py --config configs/config.yaml
python scripts/05_run_ope_dr.py        --config configs/config.yaml
python scripts/06_explain_policy.py    --config configs/config.yaml
python scripts/07_fidelity_tests.py    --config configs/config.yaml
python scripts/08_distill_policy.py    --config configs/config.yaml
```

---

## Installation

### Option 1: uv (recommended)
```bash
uv venv
source .venv/bin/activate
uv pip install -e .[dev]
```

### Option 2: poetry
```bash
poetry install
```

### Option 3: pip
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

For full reproducibility use `uv.lock` or `poetry.lock` rather than loose version ranges.

---

## Pipeline

### Phase 1 — Data to Offline RL Dataset

```
Event Log → preprocess → clean.parquet
          → encode_prefixes → prefixes.npz + vocab_activity.json
          → build_mdp → D_offline.npz
          → validate_and_split → splits.json
```

### Phase 2 — Training and Off-Policy Evaluation

```
D_offline.npz + splits.json → train_tdqn → Q_theta.ckpt
                             → ope_dr → ope_dr.json
```

### Phase 3 — Explainability and Deployment

```
Q_theta.ckpt → explain_policy (φ^V, φ^ΔQ, policy summary)
             → fidelity_tests (Q-drop, action-flip)
             → distill_policy (VIPER → decision tree)
             → export_schema → build_deploy_bundle → policy_server
```

---

## Scripts

| Script | Purpose | Output |
|--------|---------|--------|
| `01_preprocess_log.py` | Clean event log | `data/interim/clean.parquet` |
| `01b_validate_and_split.py` | Train/val/test split | `data/processed/splits.json` |
| `02_encode_prefixes.py` | Tokenize prefixes | `data/interim/prefixes.npz`, `vocab_activity.json` |
| `03_build_mdp_dataset.py` | Build MDP tuples | `data/processed/D_offline.npz` |
| `04_train_tdqn_offline.py` | Train TDQN | `artifacts/models/tdqn/{run_id}/Q_theta.ckpt` |
| `05_run_ope_dr.py` | Off-policy evaluation (doubly robust) | `artifacts/ope/ope_dr.json` |
| `06_explain_policy.py` | Generate IG attributions | `artifacts/xai/` |
| `07_fidelity_tests.py` | Q-drop and action-flip tests | `artifacts/fidelity/fidelity.csv` |
| `08_distill_policy.py` | VIPER distillation to decision tree | `artifacts/distill/final/` |

---

## Data versioning (DVC)

Large data files are versioned with DVC (not in Git).

```bash
# Pull exact data for this commit
dvc pull

# Recompute full pipeline
dvc repro
dvc push
```

Tracked files:
- `data/interim/clean.parquet`
- `data/processed/D_offline.npz`
- `data/processed/splits.json`

See `dvc.yaml` for the full pipeline definition.

**Note on `propensity` in `D_offline.npz`:** The `propensity` field (behavior policy $\mu(a|s)$) is set to `-1.0` as a placeholder during dataset building and estimated in `05_run_ope_dr.py` via `behavior_model.py`.

---

## Building the MDP Dataset (Step 3)

```bash
# Rebuild from scratch (requires --overwrite to prevent accidental overwrites)
python scripts/03_build_mdp_dataset.py --config configs/config.yaml --overwrite

# Validate inputs without building
python scripts/03_build_mdp_dataset.py --config configs/config.yaml --dry-run
```

---

## Train/Val/Test Split (Step 01b)

```bash
# Rebuild splits
python scripts/01b_validate_and_split.py --config configs/config.yaml --overwrite

# Validate without creating splits
python scripts/01b_validate_and_split.py --config configs/config.yaml --dry-run
```

Outputs:
- `data/processed/splits.json` — case assignments
- `artifacts/reports/split_report.json` — validation and drift statistics

---

## MDP Dataset format (`D_offline.npz`)

| Field | Shape | Description |
|-------|-------|-------------|
| `s` | `[N, 50]` | State prefix token IDs (left-padded to `max_len=50`) |
| `a` | `[N]` | Action taken (0 = do\_nothing, 1 = intervention) |
| `r` | `[N]` | Delayed terminal reward (0.0 at intermediate steps) |
| `s_next` | `[N, 50]` | Next state |
| `valid_actions` | `[N, A]` | Binary action mask |
| `propensity` | `[N]` | Behavior policy estimate (placeholder: −1.0) |

---

## Model architecture

`Embedding → TransformerEncoder (d_model=128, 4 heads, 3 layers) → MLP Q-head → Q-values per action`

Action masking sets invalid actions to −∞ before argmax. Training uses Double-DQN with a frozen target network (updated every 2000 steps), Huber loss, and gradient clipping (norm=10).

---

## Experiment tracking

Configure in `configs/config.yaml`:

```yaml
tracking:
  enabled: true
  backend: wandb   # or mlflow
  wandb:
    project: "xppm-tdqn"
  mlflow:
    experiment_name: "xppm-tdqn"
```

Each run logs: config hash, DVC data hashes, metrics, and artifacts.

---

## Testing and code quality

```bash
pytest                          # run tests (slow tests excluded by default)
pytest -m slow                  # run only slow tests
pytest -m "not slow" -v         # verbose non-slow tests
ruff check .                    # lint
ruff check --fix .              # lint with auto-fix
mypy src                        # type check
pre-commit run --all-files      # all pre-commit hooks
```
