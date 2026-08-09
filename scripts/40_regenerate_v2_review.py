"""Regeneration chain for the 2nd-round review fixes (criticas-2026-08-09-v2.md).

Stages:
  A. tdqn_encoder OPE variant on all 10 configs with the FIXED encoder path
     (S2: _encode_states now delegates to q_net.pooled_state). Written to
     ope_dr_encoder.json; the old ope_dr.json files are the pre-fix runs and
     become non-citable.
  B. Paper fidelity scripts under the unified random null (n_random=20,
     seed=123 shared via evaluability.DEFAULT_*): 23, 24 (now 7 configs),
     25, 26 (SLA + IR3), 28, 32.
  C. 34_audit_chain_versions --expect 2.

Resumable at stage granularity via artifacts/reports/fase7_chain_summary.json;
logs in artifacts/logs/fase7/.
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
LOGDIR = REPO / "artifacts/logs/fase7"
SUMMARY = REPO / "artifacts/reports/fase7_chain_summary.json"

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

PAPER_SCRIPTS = [
    ("23_margin_drop_compare", ["scripts/23_margin_drop_compare.py"]),
    ("24_absgap_compare", ["scripts/24_absgap_compare.py"]),
    ("25_absgap_final", ["scripts/25_absgap_final.py"]),
    ("26_fidelity_bpi2017ct", ["scripts/26_fidelity_bpi2017ct.py"]),
    ("26_fidelity_simbank-ir3", ["scripts/26_fidelity_bpi2017ct.py", "simbank-ir3"]),
    ("28_cross_level_tests", ["scripts/28_cross_level_tests.py"]),
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

    # A. tdqn_encoder variant (fixed forward) on every config
    for ds in DATASETS:
        out = (
            "artifacts/ope/ope_dr_encoder.json"
            if ds == "simbank"
            else f"artifacts/ope/{ds}/ope_dr_encoder.json"
        )
        run_stage(
            f"{ds}/05-encoder",
            [
                "scripts/05_run_ope_dr.py",
                "--dataset",
                ds,
                "--behavior",
                "tdqn_encoder",
                "--output",
                out,
            ],
            f"{ds}.log",
        )

    # B. Fidelity/paper scripts under the unified null
    for name, script in PAPER_SCRIPTS:
        run_stage(f"paper/{name}", script, "paper.log")

    # C. Version audit
    run_stage(
        "audit/34",
        ["scripts/34_audit_chain_versions.py", "--expect", "2"],
        "audit.log",
    )

    print(json.dumps({"failures": failures}, indent=1), flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
