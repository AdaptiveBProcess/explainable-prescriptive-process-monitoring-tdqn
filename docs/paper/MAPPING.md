# Paper ↔ repository mapping

Where every number in the ICPM paper ("One Step Is Not Enough: Policy-Level
Explainability for Prescriptive Process Monitoring") comes from. All artifacts
are produced by the v2 (positional) encoder chain; run
`python scripts/34_audit_chain_versions.py --expect 2` to verify no stale
checkpoint is referenced.

## Configuration names

| Paper name | Config / dataset name | Dataset yaml |
|---|---|---|
| SimBank (synth.) | `simbank` (root artifact layout) | `configs/datasets/simbank.yaml` |
| SimBank-IR3 (w=3) | `simbank-ir3` | `configs/datasets/simbank-ir3.yaml` |
| BPI 2012 | `bpi2012` | `configs/datasets/bpi2012.yaml` |
| BPI 2012-Off | `bpi2012-offertes` | `configs/datasets/bpi2012-offertes.yaml` |
| BPI 2017 | `bpi2017` | `configs/datasets/bpi2017.yaml` |
| BPI 2017-SLA | `bpi2017ct` | `configs/datasets/bpi2017ct.yaml` |
| BPI 2020 RFP | `bpi2020-rfp` | `configs/datasets/bpi2020-rfp.yaml` |
| BPI 2020 Int-Decl | `bpi2020-int-decl` | `configs/datasets/bpi2020-int-decl.yaml` |
| BPI 2020 Travel | `bpi2020-travel` | `configs/datasets/bpi2020-travel.yaml` |
| Sepsis | `sepsis` | `configs/datasets/sepsis.yaml` |
| — (CQL run, discarded; Threats) | `bpi2012cql` | `configs/datasets/bpi2012cql.yaml` |

SimBank keeps the historical root layout (`artifacts/ope/ope_dr.json`,
`artifacts/xai/*.json`, `artifacts/fidelity/fidelity.csv`); every other
configuration nests under its name (`artifacts/ope/<name>/`,
`artifacts/xai/<name>/`, `artifacts/fidelity/<name>/`).

## Table 1a (OPE) — `scripts/05_run_ope_dr.py`

- **Primary estimator (the paper's Table): `ope_dr_boa.json`** — behavior
  policy = bag-of-activities logistic regression (`--behavior boa_logreg`,
  the script default), independent of Q_theta.
- **Robustness variant: `ope_dr_encoder.json`** — behavior policy fitted on
  frozen Q_theta embeddings (`--behavior tdqn_encoder`), encoded with the
  network's own deployed forward (`pooled_state`). Each file records which
  estimator produced it in `metadata.behavior_estimator`; the default output
  filename is derived from `--behavior`, so the two can never overwrite each
  other.
- **Superseded: `ope_dr.json`** — pre-fix tdqn_encoder runs whose state
  encoding omitted positional embeddings and the attention mask and used the
  v1 pooling rule (bug fixed 2026-08-09). Kept for provenance; never citable.
- Per-arm WIS/CI/ESS: `results.tdqn_wis_mean`, `results.tdqn_wis_ci95`,
  `diagnostics.ess_fraction` (policy arm), `results.baselines.noop.*`.
  Paired case-level differences: `paired_diff.vs_noop`.

## Table 1b (fidelity + evaluability)

Every random-deletion null uses the single shared convention
`evaluability.DEFAULT_N_RANDOM = 20`, `DEFAULT_SEED = 123`, with identical
per-(item, repetition) position draws (`evaluability._random_positions`) in
scripts 23/26/32, so each configuration has one value of the random null
across all artifacts. `configs/config.yaml → fidelity.{n_random,seed}`
declares the same values (guarded by `tests/test_evaluability.py`).

| Column | Artifact | Generator |
|---|---|---|
| n_dQ, Margin, flips | `artifacts/fidelity/baselines/margin_drop_compare.json` (`ig` entry per config) | `scripts/23_margin_drop_compare.py` |
| Q-drop gap (guided vs random, paired SE) | `artifacts/fidelity/baselines/absgap_final_ig.json` | `scripts/25_absgap_final.py` |
| band (IQR(V) / E\|Δ_random\|) | `artifacts/fidelity/evaluability.json` | `scripts/32_evaluability.py` |
| Baseline methods (saliency / attention rollout) | same two files, `saliency` / `attention` entries | `scripts/21_baseline_attributions.py` + 22/23/24 |
| SLA / IR3 per-config test detail | `artifacts/fidelity/{bpi2017ct,simbank-ir3}/fidelity_ct.json` | `scripts/26_fidelity_bpi2017ct.py` |
| Level cross-matrix (each attribution under both tests) | `artifacts/fidelity/baselines/cross_level_tests.json` | `scripts/28_cross_level_tests.py` |

Explanations themselves: `scripts/06_explain_policy.py` →
`artifacts/xai[/<name>]/{risk,deltaQ}_explanations.json`. IG completeness
per config: `risk_explanations.json → metadata.ig_completeness_{risk,deltaq}`
(`n_steps_ig` 128; 512 on `simbank` and `bpi2017ct`, set in their yamls).

## Other paper numbers

| Claim | Artifact | Generator |
|---|---|---|
| Order sensitivity (Threats) | `artifacts/reports/order_sensitivity/<name>.json` | `scripts/33_check_order_sensitivity.py --out` |
| Dossier-call selection effect (use sketch) | `artifacts/reports/selection_effect_bpi2012.json` | `scripts/29_selection_effect_bpi2012.py` |
| Case 552 card + dossier numbers | `artifacts/interp/pairs.parquet` + `artifacts/explanation_example.pdf` | `scripts/27_build_interp_pairs.py`, `scripts/generate_explanation_card.py` |
| Prose statistics (l-bar per config, V ranges/median, ties, case-552 percentiles, argmax off-support fractions, V(s0), IG latency) | `artifacts/reports/paper_stats.json` | `scripts/39_paper_stats.py` |
| Generality (Sec. 4.5, PPO transplant) | `artifacts/generality/{rl-prescriptive-monitoring,when-to-treat}.json` | `scripts/37_ppo_transplant.py` (see `experiments/generality_ppo/README.md`) |
| Ranking overlap between levels | `artifacts/reports/ranking_separation.json` | `scripts/ranking_separation.py` |

## Historical / non-citable

- `artifacts/reports/fase0/` — exploratory phase-0 comparisons kept for
  provenance only (incl. `diag_results.json` and `diag_fase0_results_b.json`,
  moved out of `scripts/` 2026-08-09). Paper numbers must trace to the
  current-pipeline artifacts listed above.
- `artifacts/_archive_v1/` — artifacts produced by the discarded v1
  (permutation-invariant) encoder. Never cited. Includes
  `diag_paper_20260804.json` (pre-v2 diagnostics, moved here 2026-08-09).
- `artifacts/ope/**/ope_dr.json` — pre-fix tdqn_encoder variant runs (see the
  Table 1a section). Never cited.
- `artifacts/reports/{ingest,split,mdp_build,prefix_encoding}_report.json` —
  single-slot files overwritten by whichever dataset ran last; not traceable
  per configuration. Per-config prose statistics (e.g. Table 1a's mean prefix
  length) come from `artifacts/reports/paper_stats.json` instead.
- `artifacts/fidelity/fidelity.csv` (from `07_fidelity_tests.py`) — signed-drop
  convention that predates the |displacement| convention of Defs. 3–4; the
  paper's fidelity numbers come from scripts 23–26/28, never from this file.
