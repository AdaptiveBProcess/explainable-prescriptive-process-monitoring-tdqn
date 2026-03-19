# Multi-Dataset Refactoring

This document describes the changes made to enable the pipeline to run over
multiple datasets without file conflicts, and how to add new datasets.

## What changed

### Problem
The pipeline hard-coded SimBank-specific names in three places and used flat
paths (`data/interim/`, `data/processed/`) so running two datasets in
parallel overwrote each other's artifacts.

### Solution
Every dataset now gets its own namespace: `data/<dataset_name>/interim/` and
`data/<dataset_name>/processed/`. The dataset name flows through a single
`--dataset` CLI argument accepted by every script.

### Files modified

| File | Change |
|------|--------|
| `src/xppm/utils/config.py` | Added `deep_merge()`, `Config.for_dataset()`, `Config.resolve_paths()` |
| `src/xppm/data/build_mdp.py` | Fixed 3 hardcoded strings (`contact_headquarters`, `outcome`) |
| `configs/config.yaml` | Added `dataset_name`, path templates `{dataset_name}`, `behavior_trigger_activity` |
| `configs/datasets/simbank.yaml` | New — SimBank-specific overlay |
| `configs/datasets/template.yaml` | New — annotated template for new datasets |
| `scripts/01_preprocess_log.py` through `10_build_deploy_bundle.py` | Added `--dataset` arg, switched to `Config.for_dataset()` |
| `params.yaml` | Added `dataset_name: "simbank"` |
| `dvc.yaml` | Added `params`, `--dataset ${dataset_name}`, dataset-namespaced paths |
| `scripts/run_pipeline.sh` | New — convenience wrapper |

## How it works

`Config.for_dataset(base_yaml, dataset_name)` does three things:

1. Loads `configs/config.yaml` (base)
2. Deep-merges `configs/datasets/<dataset_name>.yaml` on top (dataset overlay)
3. Substitutes every `{dataset_name}` placeholder in all string values

This means path templates like `data/{dataset_name}/interim/clean.parquet`
expand to `data/simbank/interim/clean.parquet` automatically, without any
script needing to know the dataset name explicitly.

## How to add a new dataset

```bash
# 1. Create the dataset config overlay
cp configs/datasets/template.yaml configs/datasets/mybpic.yaml
# Edit mybpic.yaml: set raw_path, schema column names, behavior_trigger_activity,
# reward terminal_column, and action names.

# 2. Put the raw data in place
mkdir -p data/raw/mybpic/
cp /path/to/log.csv data/raw/mybpic/

# 3. Run the full pipeline
./scripts/run_pipeline.sh --dataset mybpic

# Or step by step:
python scripts/01_preprocess_log.py  --dataset mybpic
python scripts/01b_validate_and_split.py --dataset mybpic
python scripts/02_encode_prefixes.py  --dataset mybpic
python scripts/03_build_mdp_dataset.py --dataset mybpic --overwrite
python scripts/04_train_tdqn_offline.py --dataset mybpic
python scripts/05_run_ope_dr.py       --dataset mybpic
python scripts/06_explain_policy.py   --dataset mybpic
python scripts/07_fidelity_tests.py   --dataset mybpic
python scripts/08_distill_policy.py   --dataset mybpic
```

### Dataset overlay keys

See `configs/datasets/template.yaml` for a fully-annotated example. The
minimum required keys are:

```yaml
data:
  raw_path: "data/raw/mybpic/log.csv"

schema:
  case_id: "column_name_for_case_id"
  activity: "column_name_for_activity"
  timestamp: "column_name_for_timestamp"

mdp:
  behavior_trigger_activity: "name_of_intervention_activity_in_log"
  actions:
    id2name: ["do_nothing", "name_of_intervention_activity_in_log"]
  reward:
    terminal_column: "name_of_outcome_column"
```

## Running with DVC

```bash
# Set dataset in params.yaml
echo "dataset_name: mybpic" >> params.yaml   # or edit directly

# Reproduce
dvc repro
```

## Artifact layout after refactoring

```
data/
  simbank/
    interim/
      clean.parquet
      prefixes.npz
      vocab_activity.json
    processed/
      splits.json
      D_offline.npz
  mybpic/
    interim/  ...
    processed/ ...
artifacts/
  checkpoints/Q_theta.ckpt   (still flat — scoped by run_id internally)
  xai/
  ope/
  fidelity/
  distill/
```

Note: `artifacts/` is not namespaced by dataset yet. When running multiple
datasets concurrently, pass different `--output-dir` flags to each script, or
use separate DVC experiments (`dvc exp run`).
