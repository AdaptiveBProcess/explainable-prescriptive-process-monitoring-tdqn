# BPI Challenge Datasets — Pipeline Results

Fecha: 2026-02-23

## Resumen

Se agregaron 4 datasets del BPI Challenge y se corrió el pipeline hasta la fase de entrenamiento (steps 01–04). Los pasos siguientes (OPE, XAI, fidelity, distillation, deploy) **aún no se han ejecutado**.

---

## Datasets agregados

| Dataset | Archivo raw | Casos | Eventos (raw) | Trigger (intervención) | Outcome positivo |
|---------|-------------|-------|----------------|------------------------|------------------|
| `bpi2017` | `bpi2017.xes.gz` | 31,509 | 1,202,267 | `W_Call after offers` | `A_Accepted` |
| `bpi2020-int-decl` | `bpi2020-international-declarations.xes.gz` | 6,449 | 72,151 | `Send Reminder` | `Payment Handled` |
| `bpi2020-rfp` | `bpi2020-request-for-payment.xes.gz` | 6,886 | 36,796 | `Request Payment` | `Payment Handled` |
| `bpi2020-travel` | `bpi2020-travel-permit-data.xes.gz` | 7,065 | 86,581 | `Send Reminder` | `Payment Handled` |

---

## Step 01 — Preprocess (`01_preprocess_log.py`)

| Dataset | Eventos raw | Filtro lifecycle | Eventos limpios | Casos positivos (outcome) |
|---------|-------------|-----------------|-----------------|---------------------------|
| `bpi2017` | 1,202,267 | → `complete` only → 475,306 | 475,306 | 31,509/31,509 (100%)¹ |
| `bpi2020-int-decl` | 72,151 | — | 72,151 | 6,187/6,449 (95.9%) |
| `bpi2020-rfp` | 36,796 | — | 36,796 | 6,307/6,886 (91.6%) |
| `bpi2020-travel` | 86,581 | — | 86,581 | 5,721/7,065 (81.0%) |

> ¹ **BPI2017**: `A_Accepted` aparece como estado intermedio en todos los casos (sub-proceso de ofertas). La definición de outcome positivo debe refinarse para usar la *última* actividad `A_*` (accepted vs denied). Por ahora el pipeline funciona correctamente con reward=1 para todos los casos.

**Cambios en `preprocess.py` para soportar XES:**
- `filter_lifecycle_transitions()` — filtra eventos a `complete` (BPI2017 tiene start+complete)
- `compute_outcome_from_activities()` — calcula columna binaria `outcome` por presencia de actividad
- Codificación de case IDs string → enteros (XES usa IDs como `"declaration 1002"`)
- `select_cols` — descarta las ~150 columnas extra de atributos de caso en XES
- Fix: `timestamp_col` usa el nombre estándar post-`normalize_schema()`

---

## Step 02+03 — Encode & Build MDP

| Dataset | Prefijos | Vocab | Transiciones MDP | Casos MDP | reward_mean |
|---------|----------|-------|-----------------|-----------|-------------|
| `bpi2017` | 475,306 | 26 | 443,797 | 31,509 | 0.07 |
| `bpi2020-int-decl` | 72,151 | 36 | 65,702 | 6,449 | 0.09² |
| `bpi2020-rfp` | 36,796 | 21 | 29,910 | 6,814 | 0.21² |
| `bpi2020-travel` | 86,581 | 53 | 79,516 | 7,065 | 0.07 |

> ² `reward_mean` = fracción de transiciones terminales con outcome=1 sobre el total. Es bajo porque el reward es terminal-delayed (0 en pasos intermedios, `outcome` solo en el último paso).

**Fix en `encode_prefixes.py`:**
`t_ptr` ahora almacena la posición real en el trace (`t`) en vez de `min(t, max_len)`. Para casos con >50 eventos, el `t_ptr` seguía siendo único y monótonamente creciente, lo que es requerido por `validate_split.py` y `build_mdp.py`.

---

## Step 01b — Validate & Split

| Dataset | Train (casos / trans.) | Val (casos / trans.) | Test (casos / trans.) |
|---------|------------------------|----------------------|-----------------------|
| `bpi2017` | 22,056 / 311,179 | 3,150 / 44,159 | 6,303 / 88,459 |
| `bpi2020-int-decl` | 4,514 / 45,841 | 644 / 6,634 | 1,291 / 13,227 |
| `bpi2020-rfp` | 4,769 / 20,912 | 681 / 2,992 | 1,364 / 6,006 |
| `bpi2020-travel` | 4,945 / 55,788 | 706 / 8,005 | 1,414 / 15,723 |

Split por case ID aleatorio (70/10/20). Sin leakage entre splits.

---

## Step 04 — Train TDQN (`04_train_tdqn_offline.py`)

| Dataset | Run ID | Steps | Duración | Final loss | Final q_mean |
|---------|--------|-------|----------|------------|-------------|
| `bpi2020-rfp` | `20260223_112612` | 30,000 | 19 min | 0.008 | 0.970 |
| `bpi2020-int-decl` | `20260223_114505` | 50,000 | 31 min | 0.052 | 0.931 |
| `bpi2020-travel` | `20260223_121627` | 50,000 | 30 min | 0.0002 | 0.969 |
| `bpi2017` | `20260223_124657` | 100,000 | 62 min | 0.00005 | 0.935 |

GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU
Todos los modelos convergieron (loss < 0.1, gradientes estables, sin gradient clipping).

---

## Estado del pipeline

```
Step 01  ✅ preprocess          → clean.parquet
Step 01b ✅ validate_and_split  → splits.json
Step 02  ✅ encode_prefixes     → prefixes.npz + vocab_activity.json
Step 03  ✅ build_mdp_dataset   → D_offline.npz
Step 04  ✅ train_tdqn_offline  → Q_theta.ckpt

Step 05  ⏳ run_ope_dr          → ope_dr.json
Step 06  ⏳ explain_policy      → risk_attr, deltaQ, ig_grad, policy_summary, explanations_selection
Step 07  ⏳ fidelity_tests      → fidelity.csv
Step 08  ⏳ distill_policy      → tree.pkl, tree_rules.txt, rules.sql
Step 09  ⏳ export_schema       → schema.json
Step 10  ⏳ build_deploy_bundle → artifacts/deploy/v1/
```

---

## Archivos modificados / creados

**Nuevos:**
- `configs/datasets/bpi2017.yaml`
- `configs/datasets/bpi2020-int-decl.yaml`
- `configs/datasets/bpi2020-rfp.yaml`
- `configs/datasets/bpi2020-travel.yaml`

**Modificados:**
- `src/xppm/data/preprocess.py` — lifecycle filter, outcome engineering, case ID encoding, select_cols, timestamp fix
- `src/xppm/data/encode_prefixes.py` — fix `t_ptr` para casos >max_len
- `scripts/01_preprocess_log.py` — pasar `preprocess` y `outcome_engineering` al preprocessor

---

## Para continuar el pipeline

```bash
# OPE + XAI + fidelity + distillation para cada dataset
for ds in bpi2020-rfp bpi2020-int-decl bpi2020-travel bpi2017; do
  python scripts/05_run_ope_dr.py        --config configs/config.yaml --dataset $ds
  python scripts/06_explain_policy.py    --config configs/config.yaml --dataset $ds
  python scripts/07_fidelity_tests.py    --config configs/config.yaml --dataset $ds
  python scripts/08_distill_policy.py    --config configs/config.yaml --dataset $ds
  python scripts/09_export_schema.py     --config configs/config.yaml --dataset $ds
  python scripts/10_build_deploy_bundle.py --config configs/config.yaml --dataset $ds
done
```

> Nota: Los scripts 05–10 necesitan el `ckpt_path` del run de entrenamiento correspondiente a cada dataset. Verificar que `configs/datasets/<name>.yaml` apunte al checkpoint correcto o pasarlo por CLI.
