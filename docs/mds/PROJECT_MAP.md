# PROJECT MAP — xPPM-TDQN

Mapa completo del proyecto: archivos, scripts, modulos, artifacts y sus relaciones de entrada/salida.

---

## Scripts (`scripts/`)

### Phase 1 — Data to Offline RL Dataset

| Script | Entrada | Salida | Modulo interno |
|--------|---------|--------|----------------|
| `01_preprocess_log.py` | Event log (CSV/XES) en `data/raw/` | `data/interim/clean.parquet`, `artifacts/reports/ingest_report.json` | `src/xppm/data/preprocess.py` |
| `01b_validate_and_split.py` | `data/processed/D_offline.npz` | `data/processed/splits.json`, `artifacts/reports/split_report.json` | `src/xppm/data/validate_split.py` |
| `02_encode_prefixes.py` | `data/interim/clean.parquet` | `data/interim/prefixes.npz`, `data/interim/vocab_activity.json`, `artifacts/reports/encoding_report.json` | `src/xppm/data/encode_prefixes.py` |
| `03_build_mdp_dataset.py` | `data/interim/prefixes.npz`, `vocab_activity.json` | `data/processed/D_offline.npz`, `artifacts/reports/mdp_build_report.json`, `configs/schemas/offline_rlset.schema.json` | `src/xppm/data/build_mdp.py` |

### Phase 2 — Training & Evaluation

| Script | Entrada | Salida | Modulo interno |
|--------|---------|--------|----------------|
| `04_train_tdqn_offline.py` | `data/processed/D_offline.npz`, `splits.json` | `artifacts/models/tdqn/{run_id}/Q_theta.ckpt`, `target_Q.ckpt`, `config.yaml`, `train_history.json` | `src/xppm/rl/train_tdqn.py` |
| `05_run_ope_dr.py` | `Q_theta.ckpt`, `D_offline.npz`, `splits.json` | `artifacts/ope/ope_dr.json` | `src/xppm/ope/doubly_robust.py`, `behavior_model.py` |

### Phase 3 — Explainability, Distillation & Deploy Bundle

| Script | Entrada | Salida | Modulo interno |
|--------|---------|--------|----------------|
| `06_explain_policy.py` | `Q_theta.ckpt`, `D_offline.npz` | `artifacts/xai/risk_explanations.json`, `deltaQ_explanations.json`, `ig_grad_attributions.npz`, `policy_summary.json`, `explanations_selection.json` | `src/xppm/xai/attributions.py`, `explain_policy.py`, `policy_summary.py` |
| `07_fidelity_tests.py` | `Q_theta.ckpt`, `D_offline.npz`, attributions | `artifacts/fidelity/fidelity.csv` | `src/xppm/xai/fidelity_tests.py` |
| `08_distill_policy.py` | `Q_theta.ckpt`, `D_offline.npz` | `artifacts/distill/final/tree.pkl`, `tree_rules.txt`, `rules.sql`, `rules_metadata.json`, `distill_selection.json`, `fidelity_metrics.json` | `src/xppm/distill/distill_policy.py`, `export_rules.py` |
| `09_export_schema.py` | Configuracion | `artifacts/deploy/v1/schema.json` | `src/xppm/serve/schemas.py` |
| `10_build_deploy_bundle.py` | Artifacts de distill, fidelity, xai | `artifacts/deploy/v1/` (tree.pkl, rules_metadata.json, fidelity.csv, xai/, versions.json, policy_guard_config.json) | — |
| `11_test_deployment.py` | `artifacts/deploy/v1/` | Smoke test (no genera artifacts nuevos) | — |

### Phase 4 — Production Monitoring

| Script | Entrada | Salida | Modulo interno |
|--------|---------|--------|----------------|
| `13_compute_monitoring_metrics.py` | `artifacts/deploy/v1/decisions.jsonl` | `artifacts/monitoring/metrics_{date}.json`, `daily_metrics.csv` | — |
| `14_detect_drift.py` | `artifacts/monitoring/` | `artifacts/monitoring/drift_report.json` | — |
| `15_send_alerts.py` | `artifacts/monitoring/` | Alertas (email/webhook) | — |
| `16_consolidate_feedback_to_offline_dataset.py` | `artifacts/deploy/v1/feedback.jsonl` | Dataset offline consolidado | — |
| `17_check_retraining_triggers.py` | `artifacts/monitoring/` | Retraining ticket JSONs | — |
| `18_generate_dashboard.py` | `artifacts/monitoring/` | Dashboard de monitoreo | — |

### Visualizacion

| Script | Entrada | Salida |
|--------|---------|--------|
| `19_generate_results_figures.py` | Varios artifacts | Figuras del paper (Q-drop, distillation, latency plots) |
| `generate_fidelity_plots.py` | `artifacts/fidelity/fidelity.csv`, `artifacts/xai/policy_summary.json` | `artifacts/fidelity/q_drop_gap_final.png`, `action_flip_final.png`, `rank_consistency_final.png` |

### Servidor

| Script | Descripcion |
|--------|-------------|
| `policy_server.py` (scripts/) | Wrapper simple, importa de `xppm.serving.server` |
| `policy_server.py` (raiz) | Implementacion completa FastAPI (240 lineas), usa `xppm.serve.*`, bundle default `artifacts/deploy/v1`, logs a `decisions.jsonl` |

---

## Modulos (`src/xppm/`)

### `data/` — Preprocesamiento y construccion de dataset

| Archivo | Descripcion |
|---------|-------------|
| `preprocess.py` | Limpieza de event log, normalizacion de columnas |
| `encode_prefixes.py` | Tokenizacion de prefijos de actividades, padding |
| `build_mdp.py` | Construccion de tuplas MDP (s, a, r, s_next, valid_actions) |
| `validate_split.py` | Validacion y particion train/val/test |
| `schemas.py` | Schemas de validacion de datos |
| `__init__.py` | — |

### `rl/` — Entrenamiento TDQN

| Archivo | Descripcion |
|---------|-------------|
| `train_tdqn.py` | Loop de entrenamiento (Double-DQN, Huber loss, cosine LR) |
| `replay.py` | Buffer de replay offline |
| `evaluation.py` | Metricas de evaluacion del entrenamiento |
| `__init__.py` | — |

### `rl/models/` — Arquitectura del modelo

| Archivo | Descripcion |
|---------|-------------|
| `transformer.py` | `SimpleTransformerEncoder` (d_model=128, 4 heads, 3 layers) |
| `q_network.py` | MLP Q-head (mapea embedding a Q-values por accion) |
| `masking.py` | Action masking (acciones invalidas → -inf antes de argmax) |
| `__init__.py` | — |

### `ope/` — Off-Policy Evaluation

| Archivo | Descripcion |
|---------|-------------|
| `doubly_robust.py` | Estimador Doubly Robust con bootstrap confidence intervals |
| `behavior_model.py` | Modelo de behavior policy (regresion logistica multiclase) |
| `baselines.py` | Baselines de comparacion (random, always-treat, never-treat) |
| `report.py` | Generacion de reportes OPE |
| `__init__.py` | — |

### `xai/` — Explicabilidad

| Archivo | Descripcion |
|---------|-------------|
| `attributions.py` | Integrated Gradients para atribuciones de features |
| `explain_policy.py` | Explicaciones contrastivas delta-Q (risk, benefit, timing) |
| `fidelity_tests.py` | Tests de fidelidad: Q-drop, action-flip, rank-consistency |
| `policy_summary.py` | Resumen de politica por clustering de estados |
| `__init__.py` | — |

### `distill/` — Destilacion de politica

| Archivo | Descripcion |
|---------|-------------|
| `distill_policy.py` | VIPER-style distillation a decision tree |
| `export_rules.py` | Exportacion de reglas a SQL |
| `__init__.py` | — |

### `serve/` — Servidor de produccion (implementacion activa)

| Archivo | Tamano | Descripcion |
|---------|--------|-------------|
| `guard.py` | ~5.9k | Policy guard: deteccion OOD con distancia Mahalanobis, uncertainty check, fallback a no-op |
| `logger.py` | ~2.2k | Logger de decisiones a JSONL |
| `schemas.py` | ~6.1k | Schemas Pydantic de request/response |
| `__init__.py` | — | — |

### `serving/` — Servidor legacy (wrappers minimos)

| Archivo | Tamano | Descripcion |
|---------|--------|-------------|
| `server.py` | ~551b | Wrapper simple del servidor |
| `guard.py` | ~273b | Guard minimo |
| `schemas.py` | ~328b | Schemas minimos |
| `__init__.py` | — | — |

**Nota:** `serve/` es la implementacion activa y completa. `serving/` son wrappers legacy minimos. El `policy_server.py` de la raiz usa `serve.*`; el de `scripts/` usa `serving.*`.

### `utils/` — Utilidades

| Archivo | Descripcion |
|---------|-------------|
| `config.py` | Carga de configuracion YAML con hashing SHA-256 para reproducibilidad |
| `io.py` | Helpers de I/O: parquet, npz, json, yaml |
| `seed.py` | Gestion de semillas para reproducibilidad |
| `logging.py` | Configuracion de logging |

---

## Estructura de Datos (`data/`)

```
data/
  raw/          # Event logs originales (CSV/XES)
  interim/      # Datos intermedios
    clean.parquet
    prefixes.npz
    vocab_activity.json
  processed/    # Dataset final
    D_offline.npz    # Tuplas MDP: s, a, r, s_next, valid_actions, propensity
    splits.json      # Indices train/val/test
```

### Formato de `D_offline.npz`

| Campo | Shape | Descripcion |
|-------|-------|-------------|
| `s` | `[N, max_len=50]` | State prefix token IDs |
| `a` | `[N]` | Action (0=do_nothing, 1=contact_headquarters) |
| `r` | `[N]` | Delayed terminal reward (0.0 intermediate) |
| `s_next` | `[N, max_len=50]` | Next state |
| `valid_actions` | `[N, n_actions]` | Binary mask de acciones validas |
| `propensity` | `[N]` | -1.0 placeholder hasta que OPE lo estime |

---

## Estructura de Artifacts (`artifacts/`)

```
artifacts/
  reports/                    # Reportes de cada paso del pipeline
    ingest_report.json
    split_report.json
    encoding_report.json
    mdp_build_report.json
  models/
    tdqn/
      {run_id}/               # YYYYMMDD_HHMMSS
        Q_theta.ckpt          # Red Q entrenada
        target_Q.ckpt         # Red target
        config.yaml           # Config usada
        train_history.json    # Historial de entrenamiento
  ope/
    ope_dr.json               # Resultados Doubly Robust
    sensitivity/              # Analisis de sensibilidad
  xai/
    risk_explanations.json
    deltaQ_explanations.json
    ig_grad_attributions.npz
    policy_summary.json
    explanations_selection.json
    final/                    # Copias finales
  fidelity/
    fidelity.csv              # Metricas Q-drop, action-flip, rank-consistency
    q_drop_gap_final.png
    action_flip_final.png
    rank_consistency_final.png
  distill/
    final/                    # Arbol destilado final
      tree.pkl
      tree_rules.txt
      rules.sql
      rules_metadata.json
      distill_selection.json
      fidelity_metrics.json
    full/                     # Run completo
    smoke/                    # Smoke tests
  deploy/
    v1/                       # Bundle de despliegue
      tree.pkl
      rules_metadata.json
      fidelity.csv
      schema.json
      versions.json
      policy_guard_config.json
      xai/                    # Artifacts XAI
      decisions.jsonl         # Log de decisiones en produccion
      feedback.jsonl          # Feedback de usuarios
  monitoring/
    metrics_{date}.json
    daily_metrics.csv
    drift_report.json
  checkpoints/
  logs/
```

---

## Configuracion (`configs/`)

| Archivo | Descripcion |
|---------|-------------|
| `config.yaml` (~10k) | Configuracion principal: schema, encoding, mdp, training, ope, xai, fidelity, distill, serving |
| `monitoring.yaml` (~2.8k) | Configuracion de monitoreo y alertas |
| `schemas/api.schema.json` | Schema JSON de la API |
| `schemas/event_log.schema.json` | Schema del event log |
| `schemas/offline_rlset.schema.json` | Schema del dataset MDP |
| `sweeps/example_sweep.yaml` | Template para sweeps de hiperparametros |

Parametros DVC en `params.yaml` (raiz).

---

## Archivos Raiz

### Ejecucion
| Archivo | Descripcion |
|---------|-------------|
| `Makefile` | Targets: preprocess, build_rlset, train, ope, xai, distill, serve |
| `policy_server.py` | Servidor FastAPI completo (240 lineas) |
| `ci-local.sh` | Script de CI local |
| `quickcheck.sh` | Verificacion rapida |

### Proyecto
| Archivo | Descripcion |
|---------|-------------|
| `pyproject.toml` | Metadata del paquete Python, extras [dev], [tracking] |
| `ruff.toml` | Linter: line-length=100, Python 3.10, rules E/F/I |
| `mypy.ini` | Type checker: Python 3.10, ignore_missing_imports |
| `.pre-commit-config.yaml` | Hooks: ruff, end-of-file, trailing-whitespace, check-yaml |
| `dvc.yaml` | Pipeline DVC completo |
| `params.yaml` | Parametros DVC |
| `.dvcignore` | Exclusiones DVC |
| `.gitignore` | Exclusiones Git |
| `SMOKE_RUN_CONFIG.yaml` | Config para smoke tests |

### Documentacion
| Archivo | Descripcion |
|---------|-------------|
| `README.md` | Readme del proyecto |
| `CLAUDE.md` | Guia para Claude Code |
| `PROJECT_MAP.md` | Este archivo |
| `DEPLOYMENT_CHECKLIST.md` | Checklist de despliegue |
| `DISTILLATION_PAPER_SECTION.md` | Seccion del paper sobre destilacion |
| `PAPER_FIDELITY_SECTION.md` | Seccion del paper sobre fidelidad |

### Ejemplos
| Archivo | Descripcion |
|---------|-------------|
| `example_request.json` | Request de ejemplo para el servidor |
| `example_override.json` | Override de ejemplo |
| `example_ood.json` | Ejemplo out-of-distribution |

### Deploy (`deploy/`)
| Archivo | Descripcion |
|---------|-------------|
| `DEPLOYMENT_GUIDE.md` | Guia de despliegue |
| `README.md` | Readme de deploy |
| `xppm-dss.service` | Archivo systemd |
| `logrotate.conf` | Rotacion de logs |
| `setup_cron.sh` | Configuracion de cron jobs |

---

## Tests (`tests/`)

```
tests/
  conftest.py                          # Fixtures de pytest
  test_serve.py                        # Tests de integracion del servidor
  data/
    test_eventlog_schema.py
    test_rlset_schema.py
  unit/
    test_action_mask.py
    test_build_mdp.py
    test_build_mdp_real.py
    test_encode_prefixes.py
    test_encode_prefixes_deterministic.py
    test_ope_dr.py
    test_preprocess.py
    test_preprocess_parser.py
    test_tdqn_smoke.py
    test_train_smoke.py
    test_validate_split.py
```

Tests lentos marcados con `@pytest.mark.slow`, excluidos por defecto via `addopts` en pytest config.

---

## Flujo Completo del Pipeline

```
Event Log (CSV/XES)
    |
    v
[01] preprocess_log --> clean.parquet
    |
    v
[02] encode_prefixes --> prefixes.npz + vocab_activity.json
    |
    v
[03] build_mdp_dataset --> D_offline.npz
    |
    v
[01b] validate_and_split --> splits.json
    |
    v
[04] train_tdqn_offline --> Q_theta.ckpt + target_Q.ckpt
    |
    +---> [05] run_ope_dr --> ope_dr.json
    |
    +---> [06] explain_policy --> risk/deltaQ/ig/policy_summary
    |         |
    |         v
    |     [07] fidelity_tests --> fidelity.csv
    |
    +---> [08] distill_policy --> tree.pkl + rules.sql
              |
              v
          [09] export_schema --> schema.json
              |
              v
          [10] build_deploy_bundle --> artifacts/deploy/v1/
              |
              v
          [11] test_deployment (smoke test)
              |
              v
          policy_server.py (FastAPI)
              |
              v
          decisions.jsonl
              |
              +---> [13] compute_monitoring_metrics
              |         |
              |         v
              |     [14] detect_drift --> drift_report.json
              |         |
              |         v
              |     [15] send_alerts
              |
              +---> [16] consolidate_feedback
              |         |
              |         v
              |     [17] check_retraining_triggers
              |
              +---> [18] generate_dashboard
              |
              +---> [19] generate_results_figures
```
