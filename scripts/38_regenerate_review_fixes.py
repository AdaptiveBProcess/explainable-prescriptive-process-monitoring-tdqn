"""Regeneration chain for the 2026-08-09 review fixes (criticas-revision-2026-08-09.md).

Stages:
  A. bpi2017ct XAI at n_steps_ig=512 (G-completeness; simbank's 512 run was
     already produced by the trial), then fidelity for both.
  B. BoA (boa_logreg) OPE for the 6 configs that only had the tdqn_encoder
     run (G3): bpi2012, bpi2012-offertes, bpi2017, bpi2020-rfp,
     bpi2020-int-decl, bpi2020-travel.
  C. Paper scripts over the refreshed artifacts, including
     23_margin_drop_compare now covering sepsis + bpi2012-offertes (G6).
  D. Order-sensitivity JSON artifact per config (G4) via 33 --out.
  E. Explanation card figure + 34_audit_chain_versions --expect 2.

Resumable at stage granularity via artifacts/reports/fase6_chain_summary.json;
logs in artifacts/logs/fase6/.
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
LOGDIR = REPO / "artifacts/logs/fase6"
SUMMARY = REPO / "artifacts/reports/fase6_chain_summary.json"

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
BOA_MISSING = [
    "bpi2012",
    "bpi2012-offertes",
    "bpi2017",
    "bpi2020-rfp",
    "bpi2020-int-decl",
    "bpi2020-travel",
]

PAPER_SCRIPTS = [
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
    ("ranking_separation", ["scripts/ranking_separation.py"]),
    ("32_evaluability", ["scripts/32_evaluability.py"]),
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


def ope_meta(ds: str) -> dict:
    path = (
        REPO / "artifacts/ope/ope_dr.json"
        if ds == "simbank"
        else REPO / f"artifacts/ope/{ds}/ope_dr.json"
    )
    return json.loads(path.read_text())["metadata"]


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

    # A. 512-step XAI for bpi2017ct + fidelity for both 512 configs
    run_stage(
        "bpi2017ct/06-512",
        ["scripts/06_explain_policy.py", "--dataset", "bpi2017ct"],
        "bpi2017ct.log",
    )
    run_stage("simbank/07", ["scripts/07_fidelity_tests.py", "--dataset", "simbank"], "simbank.log")
    run_stage(
        "bpi2017ct/07", ["scripts/07_fidelity_tests.py", "--dataset", "bpi2017ct"], "bpi2017ct.log"
    )

    # B. BoA OPE for the six configs missing it
    for ds in BOA_MISSING:
        run_stage(
            f"{ds}/05-boa",
            [
                "scripts/05_run_ope_dr.py",
                "--dataset",
                ds,
                "--behavior",
                "boa_logreg",
                "--output",
                f"artifacts/ope/{ds}/ope_dr_boa.json",
            ],
            f"{ds}.log",
        )

    # C. Paper scripts
    for name, script in PAPER_SCRIPTS:
        run_stage(f"paper/{name}", script, "paper.log")

    # D. Order-sensitivity artifacts (G4)
    for ds in DATASETS:
        meta = ope_meta(ds)
        run_stage(
            f"order_sens/{ds}",
            [
                "scripts/33_check_order_sensitivity.py",
                "--ckpt",
                meta["ckpt_path"],
                "--data",
                f"data/{ds}/processed",
                "--vocab",
                meta["vocab_path"],
                "--out",
                f"artifacts/reports/order_sensitivity/{ds}.json",
            ],
            "order_sens.log",
        )

    # E. Figure + audit (audit always reruns)
    run_stage("figs/explanation_card", ["scripts/generate_explanation_card.py"], "figs.log")
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

    print("\n=== Fase 6 (review fixes) completa ===", flush=True)
    print(f"  etapas: {len(summary)}, fallos: {len(failures)}", flush=True)
    for k in failures:
        print(f"  FALLO: {k}", flush=True)


if __name__ == "__main__":
    main()
