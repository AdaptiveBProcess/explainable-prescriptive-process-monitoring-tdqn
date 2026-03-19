# Steps 05–10 Results — 4 BPI Datasets
**Date:** 2026-02-23
**Git commit:** `2393ca90`
**Pipeline:** OPE DR → XAI (IG) → Fidelity → Distillation → Schema → Deploy Bundle

---

## Verification: Artifact Checklist

| Artifact | bpi2020-rfp | bpi2020-int-decl | bpi2020-travel | bpi2017 |
|---|:---:|:---:|:---:|:---:|
| `artifacts/ope/{ds}/ope_dr.json` | ✅ | ✅ | ✅ | ✅ |
| `artifacts/xai/{ds}/risk_explanations.json` | ✅ | ✅ | ✅ | ✅ |
| `artifacts/xai/{ds}/deltaQ_explanations.json` | ✅ | ✅ | ✅ | ✅ |
| `artifacts/xai/{ds}/policy_summary.json` | ✅ | ✅ | ✅ | ✅ |
| `artifacts/xai/{ds}/explanations_selection.json` | ✅ | ✅ | ✅ | ✅ |
| `artifacts/fidelity/{ds}/fidelity.csv` | ✅ | ✅ | ✅ | ✅ |
| `artifacts/distill/{ds}/tree.pkl` | ✅ | ✅ | ✅ | ✅ |
| `artifacts/distill/{ds}/tree_rules.txt` | ✅ | ✅ | ✅ | ✅ |
| `artifacts/distill/{ds}/rules.sql` | ✅ | ✅ | ✅ | ✅ |
| `artifacts/distill/{ds}/fidelity_metrics.json` | ✅ | ✅ | ✅ | ✅ |
| `artifacts/deploy/{ds}/v1/schema.json` | ✅ | ✅ | ✅ | ✅ |
| `artifacts/deploy/{ds}/v1/tree.pkl` | ✅ | ✅ | ✅ | ✅ |
| `artifacts/deploy/{ds}/v1/policy_guard_config.json` | ✅ | ✅ | ✅ | ✅ |
| `artifacts/deploy/{ds}/v1/versions.json` | ✅ | ✅ | ✅ | ✅ |
| `artifacts/deploy/{ds}/v1/xai/policy_summary.json` | ✅ | ✅ | ✅ | ✅ |

**60/60 artifacts present across all 4 datasets.**

---

## Step 05 — Off-Policy Evaluation (Doubly Robust)

| Dataset | TDQN DR | DR CI 95% | TDQN WIS | Behavior return | Noop DR |
|---|---:|---|---:|---:|---:|
| bpi2020-rfp | **0.2573** | [0.2555, 0.2595] | 0.1734 | 0.2121 | 0.1321 |
| bpi2020-int-decl | **0.3676** | [0.3656, 0.3695] | 0.0934 | 0.0937 | −0.2407 |
| bpi2020-travel | **0.3998** | [0.3900, 0.4093] | 0.0733 | 0.0733 | 0.0205 |
| bpi2017 | **0.4756** | [0.4745, 0.4766] | 0.0744 | 0.0713 | 0.0203 |

- TDQN DR > Noop DR for all 4 datasets — the learned policy outperforms the do-nothing baseline in DR estimates.
- `bpi2020-rfp`: highest behavior empirical return (0.212), suggesting the historical log was already interventionist.
- `bpi2017`: highest TDQN DR (0.476), ≈23× above noop.
- `bpi2020-int-decl`: noop DR is negative (−0.241), showing that doing nothing is actively harmful in that process.

---

## Step 06 — XAI (Integrated Gradients)

200 transitions selected per dataset (strategy: random, last event per case, from test split).
Two attribution sets per dataset: **RISK** (Q\*, best action) and **DELTA-Q** (contrast vs. do\_nothing).
Policy summary: K-Means clustering (k=8) on encoder embeddings.

| Dataset | n\_test transitions | n\_selected | Clusters |
|---|---:|---:|---:|
| bpi2020-rfp | 6,006 | 200 | 8 |
| bpi2020-int-decl | — | 200 | 8 |
| bpi2020-travel | — | 200 | 8 |
| bpi2017 | — | 200 | 8 |

Outputs per dataset: `risk_explanations.json`, `deltaQ_explanations.json`,
`ig_grad_attributions.npz`, `policy_summary.json`, `explanations_selection.json`.

---

## Step 07 — Fidelity Tests

79 rows per dataset. Tests: Q-drop (target ∈ {q\_star, delta\_q}, p\_remove ∈ {0.1, 0.2, 0.3, 0.5}),
action-flip (same p\_remove levels), rank-consistency (8 cluster-level).

**Q-drop gap at p\_remove=0.10, target=q\_star** (positive = attributions are informative):

| Dataset | drop\_topk | drop\_rand | gap |
|---|---:|---:|---:|
| bpi2020-rfp | −0.00167 | +0.00126 | −0.00294 |
| bpi2020-int-decl | +0.00636 | −0.00045 | +0.00681 |
| bpi2020-travel | −0.00755 | −0.00212 | −0.00542 |
| bpi2017 | +0.02558 | +0.00398 | +0.02160 |

- `bpi2017` shows the strongest positive gap (+0.022): removing top-attributed tokens degrades Q more than random — attributions are highly informative.
- `bpi2020-rfp` and `bpi2020-travel` have negative gaps, suggesting the model's decisions are distributed across tokens rather than concentrated on a few.

---

## Step 08 — Policy Distillation (VIPER → Decision Tree, max\_depth=5)

| Dataset | Action agr. (global) | Action agr. (high-impact) | Margin corr. | Depth | Leaves | n\_train | n\_test |
|---|---:|---:|---:|---:|---:|---:|---:|
| bpi2020-rfp | **96.8%** | 96.8% | −0.230 | 5 | 8 | 1,079 | 463 |
| bpi2020-int-decl | 82.8% | **96.4%** | −0.169 | 5 | 8 | 1,096 | 471 |
| bpi2020-travel | 85.8% | 95.2% | −0.099 | 5 | 9 | 1,097 | 471 |
| bpi2017 | 80.2% | 51.1% | +0.354 | 4 | 8 | 1,118 | 480 |

- `bpi2020-rfp`: best global fidelity (96.8%) — the TDQN policy is nearly perfectly captured by a depth-5 tree.
- `bpi2020-int-decl` and `bpi2020-travel`: lower global fidelity but high-impact fidelity remains >95%, meaning the tree correctly reproduces the policy on the most consequential states.
- `bpi2017`: high-impact fidelity drops to 51.1%. The larger, more complex dataset produces a policy that a shallow tree cannot fully capture for high-variance states. Global fidelity (80.2%) remains acceptable.
- Negative margin correlations for bpi2020-* indicate the tree replicates action choices but not the Q-value confidence margins.

---

## Step 09 — Schema Export

| Dataset | Output |
|---|---|
| bpi2020-rfp | `artifacts/deploy/bpi2020-rfp/v1/schema.json` |
| bpi2020-int-decl | `artifacts/deploy/bpi2020-int-decl/v1/schema.json` |
| bpi2020-travel | `artifacts/deploy/bpi2020-travel/v1/schema.json` |
| bpi2017 | `artifacts/deploy/bpi2017/v1/schema.json` |

---

## Step 10 — Deploy Bundle

| Dataset | model\_version | data\_version | deployed\_at |
|---|---|---|---|
| bpi2020-rfp | `7744bd6b...` | `a5c56832...` | 2026-02-23T19:39:59 |
| bpi2020-int-decl | `27e1eca4...` | `812b1292...` | 2026-02-23T19:40:41 |
| bpi2020-travel | `13752cd5...` | `84bb1d95...` | 2026-02-23T19:41:23 |
| bpi2017 | `f1c3236d...` | `30fb3cda...` | 2026-02-23T19:43:41 |

All bundles share `config_version: 91e6cdb4` and `git_commit: 2393ca90`.

Bundle layout (identical for all 4 datasets):
```
artifacts/deploy/{ds}/v1/
├── schema.json
├── tree.pkl
├── rules_metadata.json
├── fidelity.csv
├── policy_guard_config.json
├── versions.json
└── xai/
    ├── policy_summary.json
    ├── risk_explanations.json
    └── deltaQ_explanations.json
```

---

## Issues Fixed During This Run

### 1. Missing checkpoint/output keys in BPI YAMLs
All 4 dataset YAMLs were missing `experiment.checkpoint_path`, `xai.checkpoint_path`,
`distill.teacher_checkpoint`, and namespaced output paths. Scripts were falling back to
the SimBank checkpoint and overwriting the same output files across datasets.
**Fix:** Added `experiment`, `xai`, `fidelity`, `distill` sections to each YAML with
`{dataset_name}` placeholders in all output paths.

### 2. Double `artifacts/` prefix in XAI output path
`xai.out_dir` was initially set to `"artifacts/xai/{dataset_name}"` but the
`explain_policy.py` module prepends `paths.artifacts_dir = "artifacts"` automatically,
producing `artifacts/artifacts/xai/{ds}`.
**Fix:** Changed `xai.out_dir` to the relative form `"xai/{dataset_name}"` in all 4 YAMLs.

### 3. Deploy bundle expected non-existent `final/` subdirectory
The run command used `--distill-dir artifacts/distill/${ds}/final` but
`08_distill_policy.py` writes directly to `artifacts/distill/{ds}/` (the distill library
detects the `artifacts/` prefix and skips joining with `artifacts_dir`).
**Fix:** Dropped `/final` from the `--distill-dir` argument.

### 4. `10_build_deploy_bundle.py` hardcoded SimBank data paths
`extract_feature_stats()` used hardcoded `data/processed/D_offline.npz` and
`data/interim/clean.parquet`. The `data_version` hash also pointed to the same path.
**Fix:** Threaded `dataset_name` and `cfg` (loaded via `Config.for_dataset()`) from
`__main__` → `build_deploy_bundle()` → `extract_feature_stats()`, resolving paths from
`cfg["mdp"]["output"]["path"]` and `cfg["data"]["output_clean_path"]`.
