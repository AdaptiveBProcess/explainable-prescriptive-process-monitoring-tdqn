Déjame revisar el estado real de cada script antes de responder.

● Read 1 file (ctrl+o to expand)

● Explore(Audit scripts 12-18 and policy_server for dataset support)
  ⎿  Done (8 tool uses · 36.4k tokens · 16s)
  (ctrl+o to expand)

● Aqui va la respuesta honesta, script por script:

  ---
  Estado actual de cada script

  Script: 11
  Estado: Recien arreglado — necesita servidor activo
  Por que: Hace HTTP requests a localhost:8000. Para cada dataset hay que
    levantar el server con ese bundle, luego correr el test
  ────────────────────────────────────────
  Script: 12
  Estado: Ops-level — OK as-is
  Por que: Solo manda curl a la API con archivos example_*.json. No es
    per-dataset
  ────────────────────────────────────────
  Script: 13
  Estado: Ops-level — OK as-is
  Por que: Lee decisions.jsonl del server en produccion; no es per-dataset
  ────────────────────────────────────────
  Script: 14
  Estado: Ops-level — OK as-is
  Por que: Analiza artifacts/monitoring/ en agregado
  ────────────────────────────────────────
  Script: 15
  Estado: Ops-level — OK as-is
  Por que: Despacha alertas desde el reporte de drift global
  ────────────────────────────────────────
  Script: 16
  Estado: Ops-level — OK as-is
  Por que: Consolida feedback de un log compartido del servidor
  ────────────────────────────────────────
  Script: 17
  Estado: Ops-level — OK as-is
  Por que: Verifica triggers de reentrenamiento global
  ────────────────────────────────────────
  Script: 18
  Estado: Ops-level — OK as-is
  Por que: Genera un dashboard HTML con metricas agregadas
  ────────────────────────────────────────
  Script: 19
  Estado: Recien arreglado
  Por que: Lee artifacts, genera figuras. No necesita servidor
  ────────────────────────────────────────
  Script: generate_fidelity_plots
  Estado: Recien arreglado
  Por que: Lee artifacts, genera plots. No necesita servidor
  ────────────────────────────────────────
  Script: policy_server.py
  Estado: Ops-level — OK as-is
  Por que: Levanta una instancia de FastAPI; se configura por bundle, no por
    --dataset

  ---
  El problema real con el script 11

  El script 11 no puede correr sin un servidor levantado. Para el paper, el
  flujo correcto seria:

  # Para cada dataset:
  # 1. Levantar el server apuntando al bundle de ese dataset
  python scripts/policy_server.py  # pero necesita saber qué bundle cargar

  # 2. Correr el smoke test
  python scripts/11_test_deployment.py \
    --bundle-dir artifacts/deploy/${ds}/v1 \
    --dataset $ds --n-cases 30

  El problema es que policy_server.py no tiene --bundle-dir. Si quieres correr
  el script 11 para los 4 datasets BPI de forma automatizada, necesitaria:
  - O bien que policy_server.py acepte --bundle-dir como argumento CLI
  - O simplemente no correr el 11 en batch y usarlo solo cuando tengas el
  servidor levantado manualmente

  Para el paper, los scripts 19 y generate_fidelity_plots (que generan las
  figuras) son los que realmente importan en batch. Los scripts 13-18 son
  infraestructura de produccion que no aplica al pipeline academico.

  Quieres que arregle policy_server.py para que acepte --bundle-dir y asi poder
   correr el 11 en loop por dataset?
