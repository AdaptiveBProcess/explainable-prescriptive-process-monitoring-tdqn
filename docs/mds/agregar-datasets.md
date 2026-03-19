El sistema ya tiene todo el mecanismo. Son 3 pasos por dataset:

  ---
  Paso 1 — Crear el overlay YAML

  Por cada dataset nuevo, crea configs/datasets/<nombre>.yaml copiando el template:

  cp configs/datasets/template.yaml configs/datasets/bpic2017.yaml
  cp configs/datasets/template.yaml configs/datasets/bpic2012.yaml
  cp configs/datasets/template.yaml configs/datasets/sepsis.yaml
  cp configs/datasets/template.yaml configs/datasets/traffic.yaml

  Luego editar cada uno. Lo mínimo que cambia por dataset:

  # configs/datasets/bpic2017.yaml
  data:
    raw_path: "data/raw/bpic2017/BPI_Challenge_2017.csv"
    format: "csv"

  schema:
    case_id: "Case ID"          # nombre real de la columna en el log
    activity: "Activity"
    timestamp: "time:timestamp"

  mdp:
    behavior_trigger_activity: "O_Accepted"   # actividad que = intervención
    actions:
      id2name: ["do_nothing", "O_Accepted"]
      noop_action: "do_nothing"
    reward:
      terminal_column: "outcome"    # columna con el resultado final

  Solo pones las claves que difieren de configs/config.yaml. El deep_merge lo maneja.

  ---
  Paso 2 — Poner los logs en data/raw/

  data/raw/
    bpic2017/BPI_Challenge_2017.csv
    bpic2012/BPI_Challenge_2012.csv
    sepsis/Sepsis_Cases.csv
    traffic/Road_Traffic_Fine.csv

  ---
  Paso 3 — Correr el pipeline completo por dataset

  # Cada dataset es completamente independiente y namespaceado
  python scripts/01_preprocess_log.py   --dataset bpic2017
  python scripts/01b_validate_and_split.py --dataset bpic2017 --overwrite
  python scripts/02_encode_prefixes.py  --dataset bpic2017
  python scripts/03_build_mdp_dataset.py --dataset bpic2017
  python scripts/04_train_tdqn_offline.py --dataset bpic2017 --algo tdqn
  python scripts/05_run_ope_dr.py       --dataset bpic2017
  # ... etc

  Los artefactos quedan separados automáticamente porque {dataset_name} se resuelve en todas las rutas:
  data/bpic2017/interim/clean.parquet
  data/bpic2017/interim/vocab_activity.json
  data/bpic2017/processed/D_offline.npz
  artifacts/models/tdqn/<run_id>/Q_theta.ckpt   # ← este es global, no namespaceado

  ---
  Lo que probablemente necesites ajustar por dataset

  ┌─────────────────────────────────┬──────────────────────────────────────────┬─────────────────────┐
  │              Clave              │                  Qué es                  │        Dónde        │
  ├─────────────────────────────────┼──────────────────────────────────────────┼─────────────────────┤
  │ schema.case_id                  │ Nombre real de la columna de caso        │ overlay YAML        │
  ├─────────────────────────────────┼──────────────────────────────────────────┼─────────────────────┤
  │ schema.activity                 │ Nombre real de la columna de actividad   │ overlay YAML        │
  ├─────────────────────────────────┼──────────────────────────────────────────┼─────────────────────┤
  │ mdp.behavior_trigger_activity   │ Actividad que representa la intervención │ overlay YAML        │
  ├─────────────────────────────────┼──────────────────────────────────────────┼─────────────────────┤
  │ mdp.actions.id2name             │ Nombres de las acciones                  │ overlay YAML        │
  ├─────────────────────────────────┼──────────────────────────────────────────┼─────────────────────┤
  │ mdp.reward.terminal_column      │ Columna con el outcome final             │ overlay YAML        │
  ├─────────────────────────────────┼──────────────────────────────────────────┼─────────────────────┤
  │ numeric_cols                    │ Features numéricas del log               │ overlay si difieren │
  ├─────────────────────────────────┼──────────────────────────────────────────┼─────────────────────┤
  │ training.batch_size / max_steps │ Si el dataset es muy pequeño o grande    │ overlay YAML        │
  └─────────────────────────────────┴──────────────────────────────────────────┴─────────────────────┘

  Los datasets BPIC suelen tener columnas XES (case:concept:name, concept:name, time:timestamp) — el template ya las
  tiene como default.
