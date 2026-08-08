"""Fase 2: retrain every dataset with the v2 encoder, sequentially.

For each dataset in QUEUE:
  1. If its D_offline.npz is missing, run the data pipeline (01, 01b, 02, 03).
  2. Train (04_train_tdqn_offline.py).
  3. Locate the new run dir, verify encoder_version == 2 and vocab consistency.
  4. Update the dataset yaml checkpoint pins (experiment/xai checkpoint_path,
     distill.teacher_checkpoint) to the new run.
  5. Run 33_check_order_sensitivity.py under XPPM_EXPECT_ENCODER_VERSION=2.

A failure in one dataset is logged and the chain continues. Summary at
artifacts/reports/fase2_retrain_summary.json, per-dataset logs under
artifacts/logs/fase2/.
"""

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
LOGDIR = REPO / "artifacts/logs/fase2"
SUMMARY = REPO / "artifacts/reports/fase2_retrain_summary.json"
RUNS = REPO / "artifacts/models/tdqn"

QUEUE = [
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


def sh(args, log_path, env_extra=None):
    import os

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src")
    if env_extra:
        env.update(env_extra)
    with open(log_path, "a") as log:
        log.write(f"\n$ {' '.join(map(str, args))}\n")
        log.flush()
        proc = subprocess.run(args, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    return proc.returncode


def verify_ckpt(run_dir: Path, dataset: str) -> dict:
    import torch

    c = torch.load(run_dir / "Q_theta.ckpt", map_location="cpu", weights_only=False)
    sd = c.get("model_state_dict", c)
    vocab = json.loads((REPO / f"data/{dataset}/interim/vocab_activity.json").read_text())
    n_vocab = len(vocab["token2id"])
    info = {
        "encoder_version": c.get("encoder_version"),
        "vocab_size": c.get("vocab_size"),
        "emb_rows": int(sd["embedding.weight"].shape[0]),
        "expected_vocab": n_vocab,
    }
    info["ok"] = info["encoder_version"] == 2 and info["vocab_size"] == n_vocab
    return info


def update_pins(dataset: str, ckpt_rel: str) -> None:
    yaml_path = REPO / f"configs/datasets/{dataset}.yaml"
    text = yaml_path.read_text()
    pin = f'"{ckpt_rel}"  # v2 (fase2 auto)'
    text = re.sub(r"^(\s*checkpoint_path:).*$", rf"\1 {pin}", text, flags=re.M)
    text = re.sub(r"^(\s*teacher_checkpoint:).*$", rf"\1 {pin}", text, flags=re.M)
    yaml_path.write_text(text)


def main() -> None:
    LOGDIR.mkdir(parents=True, exist_ok=True)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    summary = json.loads(SUMMARY.read_text()) if SUMMARY.exists() else {}

    for ds in QUEUE:
        if summary.get(ds, {}).get("status") == "ok":
            print(f"[{ds}] ya completado, salto", flush=True)
            continue
        log = LOGDIR / f"{ds}.log"
        entry = {"started": datetime.now().isoformat(timespec="seconds")}
        summary[ds] = entry
        print(f"[{ds}] inicio -> {log}", flush=True)

        try:
            # 1. Data pipeline if needed
            if not (REPO / f"data/{ds}/processed/D_offline.npz").exists():
                for script in (
                    ["scripts/01_preprocess_log.py"],
                    ["scripts/01b_validate_and_split.py", "--overwrite"],
                    ["scripts/02_encode_prefixes.py"],
                    ["scripts/03_build_mdp_dataset.py", "--overwrite"],
                ):
                    rc = sh([PY, *script, "--dataset", ds], log)
                    if rc != 0:
                        raise RuntimeError(f"{script[0]} rc={rc}")
                entry["data_pipeline"] = "ran"

            # 2. Train
            before = {p.name for p in RUNS.iterdir() if p.is_dir()}
            rc = sh([PY, "scripts/04_train_tdqn_offline.py", "--dataset", ds], log)
            if rc != 0:
                raise RuntimeError(f"train rc={rc}")
            new = sorted({p.name for p in RUNS.iterdir() if p.is_dir()} - before)
            if not new:
                raise RuntimeError("train ok pero no aparecio run dir nuevo")
            run_id = new[-1]
            entry["run_id"] = run_id

            # 3. Verify checkpoint
            info = verify_ckpt(RUNS / run_id, ds)
            entry["ckpt"] = info
            if not info["ok"]:
                raise RuntimeError(f"checkpoint no valido: {info}")

            # 4. Update yaml pins
            ckpt_rel = f"artifacts/models/tdqn/{run_id}/Q_theta.ckpt"
            update_pins(ds, ckpt_rel)
            entry["pins"] = ckpt_rel

            # 5. Order sensitivity under the v2 guard
            rc = sh(
                [
                    PY,
                    "scripts/33_check_order_sensitivity.py",
                    "--ckpt",
                    ckpt_rel,
                    "--data",
                    f"data/{ds}/processed",
                    "--vocab",
                    f"data/{ds}/interim/vocab_activity.json",
                ],
                log,
                env_extra={"XPPM_EXPECT_ENCODER_VERSION": "2"},
            )
            entry["order_sensitivity_rc"] = rc

            entry["status"] = "ok"
            print(f"[{ds}] OK run={run_id}", flush=True)
        except Exception as exc:
            entry["status"] = "FAILED"
            entry["error"] = str(exc)
            print(f"[{ds}] FALLO: {exc} (ver {log})", flush=True)
        finally:
            entry["finished"] = datetime.now().isoformat(timespec="seconds")
            SUMMARY.write_text(json.dumps(summary, indent=1))

    print("\n=== Fase 2 cadena completa ===", flush=True)
    for ds, e in summary.items():
        print(f"  {ds}: {e.get('status')} run={e.get('run_id', '-')}", flush=True)


if __name__ == "__main__":
    main()
