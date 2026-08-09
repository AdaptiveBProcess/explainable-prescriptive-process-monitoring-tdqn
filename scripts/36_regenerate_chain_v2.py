"""Fase 3: regenerate the full artifact chain on the v2 checkpoints.

Every subprocess runs under XPPM_EXPECT_ENCODER_VERSION=2, so any stage that
silently resolves a stale v1 checkpoint aborts instead of mixing versions.

Stages:
  A. Per dataset (simbank root layout + 9 per-dataset configs):
     05_run_ope_dr -> 06_explain_policy -> 07_fidelity_tests
     (simbank additionally regenerates the boa_logreg OPE sensitivity report)
  B. 32_evaluability (Def. 4) over all datasets.
  C. Paper scripts: 21-27, fase0/*, ranking_separation, diagnostics.
  D. Figures: 19, generate_fidelity_plots, generate_explanation_card.
  E. 34_audit_chain_versions --expect 2 (must report zero v1 artifacts).

Resumable at stage granularity via artifacts/reports/fase3_chain_summary.json;
failures are logged and the chain continues (audit runs regardless).
Per-stage logs: artifacts/logs/fase3/.
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
LOGDIR = REPO / "artifacts/logs/fase3"
SUMMARY = REPO / "artifacts/reports/fase3_chain_summary.json"

# simbank keeps the historical root layout (artifacts/ope/ope_dr.json,
# artifacts/xai, artifacts/fidelity/fidelity.csv) that 32_evaluability and the
# paper scripts read; it runs without --dataset but with an explicit --ckpt.
SIMBANK_CKPT = "artifacts/models/tdqn/20260807_170819/Q_theta.ckpt"

DATASETS = [
    "simbank",
    "simbank-ir3",
    "bpi2012",
    "bpi2017",
    "bpi2017ct",
    "bpi2020-rfp",
    "bpi2020-int-decl",
    "bpi2020-travel",
    "bpi2012-offertes",
    "sepsis",
]

# (stage key, argv) — keys must be unique; they gate resume skipping.
PAPER_SCRIPTS = [
    ("21_baseline_attributions", ["scripts/21_baseline_attributions.py"]),
    ("22_fidelity_baselines", ["scripts/22_fidelity_baselines.py"]),
    ("23_margin_drop_compare", ["scripts/23_margin_drop_compare.py"]),
    ("24_absgap_compare", ["scripts/24_absgap_compare.py"]),
    ("25_absgap_final", ["scripts/25_absgap_final.py"]),
    ("26_fidelity_bpi2017ct", ["scripts/26_fidelity_bpi2017ct.py"]),
    ("26_fidelity_simbank-ir3", ["scripts/26_fidelity_bpi2017ct.py", "simbank-ir3"]),
    (
        "27_build_interp_pairs",
        [
            "scripts/27_build_interp_pairs.py",
            "--dataset",
            "simbank",
            "--ckpt",
            SIMBANK_CKPT,
            "--npz",
            "data/simbank/processed/D_offline.npz",
            "--vocab",
            "data/simbank/interim/vocab_activity.json",
            "--overwrite",
        ],
    ),
    ("f0_21", ["scripts/fase0/f0_21.py"]),
    ("f0_23", ["scripts/fase0/f0_23.py"]),
    ("f0_24", ["scripts/fase0/f0_24.py"]),
    ("f0_24_ir3", ["scripts/fase0/f0_24_ir3.py"]),
    ("t13_repr", ["scripts/fase0/t13_repr.py"]),
    ("t23_ablation", ["scripts/fase0/t23_ablation.py"]),
    ("t25_cross", ["scripts/fase0/t25_cross.py"]),
    ("diag_fase0", ["scripts/fase0/diag_fase0.py"]),
    ("ranking_separation", ["scripts/ranking_separation.py"]),
    ("diagnostics_paper_20260804", ["scripts/diagnostics_paper_20260804.py"]),
]

# 19_generate_results_figures and generate_fidelity_plots expect a pre-refactor
# single-target fidelity.csv schema and produce PNGs the paper does not include
# (its only figure is artifacts/explanation_example.pdf) — left out as legacy.
FIGURE_SCRIPTS = [
    ["scripts/generate_explanation_card.py"],
]


def sh(args, log_path):
    import os

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src")
    env["XPPM_EXPECT_ENCODER_VERSION"] = "2"
    with open(log_path, "a") as log:
        log.write(f"\n$ {' '.join(map(str, args))}\n")
        log.flush()
        proc = subprocess.run(args, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    return proc.returncode


def main() -> None:
    LOGDIR.mkdir(parents=True, exist_ok=True)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    summary = json.loads(SUMMARY.read_text()) if SUMMARY.exists() else {}
    failures = []

    def run_stage(key: str, args, log_name: str):
        if summary.get(key, {}).get("status") == "ok":
            print(f"[{key}] ya completado, salto", flush=True)
            return
        log = LOGDIR / log_name
        entry = {"started": datetime.now().isoformat(timespec="seconds")}
        summary[key] = entry
        rc = sh([PY, *args], log)
        entry["rc"] = rc
        entry["status"] = "ok" if rc == 0 else "FAILED"
        entry["finished"] = datetime.now().isoformat(timespec="seconds")
        SUMMARY.write_text(json.dumps(summary, indent=1))
        if rc == 0:
            print(f"[{key}] OK", flush=True)
        else:
            failures.append(key)
            print(f"[{key}] FALLO rc={rc} (ver {log})", flush=True)

    # A. OPE -> XAI -> fidelity per dataset
    for ds in DATASETS:
        # Always pass --dataset: the root config's {dataset_name} placeholders
        # are only substituted through Config.for_dataset. simbank.yaml does not
        # override the xai/fidelity output paths, so the historical root layout
        # (artifacts/xai, artifacts/fidelity/fidelity.csv) is preserved; only
        # the OPE report needs the explicit root --output.
        flag = ["--dataset", ds]
        if ds == "simbank":
            run_stage(
                "simbank/05",
                [
                    "scripts/05_run_ope_dr.py",
                    *flag,
                    "--ckpt",
                    SIMBANK_CKPT,
                    "--output",
                    "artifacts/ope/ope_dr.json",
                ],
                "simbank.log",
            )
            run_stage(
                "simbank/05-boa",
                [
                    "scripts/05_run_ope_dr.py",
                    *flag,
                    "--ckpt",
                    SIMBANK_CKPT,
                    "--behavior",
                    "boa_logreg",
                    "--output",
                    "artifacts/ope/ope_dr_boa.json",
                ],
                "simbank.log",
            )
        else:
            run_stage(f"{ds}/05", ["scripts/05_run_ope_dr.py", *flag], f"{ds}.log")
            if ds in ("bpi2017ct", "simbank-ir3"):
                # 26_fidelity_bpi2017ct (also run for simbank-ir3) reads the
                # boa_logreg OPE variant.
                run_stage(
                    f"{ds}/05-boa",
                    [
                        "scripts/05_run_ope_dr.py",
                        *flag,
                        "--behavior",
                        "boa_logreg",
                        "--output",
                        f"artifacts/ope/{ds}/ope_dr_boa.json",
                    ],
                    f"{ds}.log",
                )
        run_stage(f"{ds}/06", ["scripts/06_explain_policy.py", *flag], f"{ds}.log")
        run_stage(f"{ds}/07", ["scripts/07_fidelity_tests.py", *flag], f"{ds}.log")

    # B. Evaluability (Def. 4)
    run_stage("global/32_evaluability", ["scripts/32_evaluability.py"], "global.log")

    # C. Paper scripts
    for name, script in PAPER_SCRIPTS:
        run_stage(f"paper/{name}", script, "paper.log")

    # D. Figures
    for script in FIGURE_SCRIPTS:
        name = Path(script[0]).stem
        run_stage(f"figs/{name}", script, "figs.log")

    # E. Audit — always rerun (cheap, and it must see the final state)
    summary.pop("global/34_audit", None)
    run_stage(
        "global/34_audit",
        [
            "scripts/34_audit_chain_versions.py",
            "--expect",
            "2",
            "--json",
            "artifacts/reports/chain_version_audit.json",
        ],
        "global.log",
    )

    print("\n=== Fase 3 cadena completa ===", flush=True)
    print(f"  etapas: {len(summary)}, fallos: {len(failures)}", flush=True)
    for k in failures:
        print(f"  FALLO: {k}", flush=True)


if __name__ == "__main__":
    main()
