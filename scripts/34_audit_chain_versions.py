"""Audit which checkpoint (and encoder version) produced each chain artifact.

Walks every checkpoint reference in the pipeline — config pins
(configs/config.yaml + configs/datasets/*.yaml) and any string mentioning a
checkpoint inside artifacts/**/*.json (OPE metadata, XAI outputs, reports) —
resolves each referenced checkpoint's encoder_version and vocab_size, and
prints a grouped report. This is the Fase 1 guard against silently mixing v1
and v2 checkpoints across the paper chain: run it after every regeneration.

Usage:
    python scripts/34_audit_chain_versions.py                 # report only
    python scripts/34_audit_chain_versions.py --expect 2      # exit 1 on any
                                                              # other version
    python scripts/34_audit_chain_versions.py --json out.json # also write JSON

Exit code 1 with --expect if any reference resolves to a checkpoint whose
encoder version differs, or whose file is missing.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

CKPT_MARKERS = ("Q_theta.ckpt", "target_Q.ckpt", "artifacts/models/tdqn/", "artifacts/checkpoints/")
MAX_JSON_BYTES = 5_000_000  # skip training histories and other bulk logs


def looks_like_ckpt(value: str) -> bool:
    return any(m in value for m in CKPT_MARKERS) and value.endswith(".ckpt")


def walk_json(node, path, hits):
    if isinstance(node, dict):
        for k, v in node.items():
            walk_json(v, f"{path}.{k}", hits)
    elif isinstance(node, list):
        for i, v in enumerate(node[:200]):
            walk_json(v, f"{path}[{i}]", hits)
    elif isinstance(node, str) and looks_like_ckpt(node):
        hits.append((path, node))


def collect_references() -> list[dict]:
    refs = []
    # 1) Config pins
    for cfg_path in [
        REPO / "configs/config.yaml",
        *sorted((REPO / "configs/datasets").glob("*.yaml")),
    ]:
        try:
            cfg = yaml.safe_load(cfg_path.read_text()) or {}
        except Exception as exc:
            refs.append(
                {"source": str(cfg_path), "where": "(parse error)", "ckpt": None, "error": str(exc)}
            )
            continue
        hits = []
        walk_json(cfg, "", hits)
        for where, ckpt in hits:
            refs.append({"source": str(cfg_path.relative_to(REPO)), "where": where, "ckpt": ckpt})
    # 2) Artifact JSONs (OPE metadata, XAI outputs, reports)
    for jf in sorted((REPO / "artifacts").rglob("*.json")):
        try:
            if jf.stat().st_size > MAX_JSON_BYTES:
                continue
            doc = json.loads(jf.read_text())
        except Exception:
            continue
        hits = []
        walk_json(doc, "", hits)
        for where, ckpt in hits:
            refs.append({"source": str(jf.relative_to(REPO)), "where": where, "ckpt": ckpt})
    return refs


def resolve_ckpt(ckpt: str) -> dict:
    """encoder_version / vocab_size of a referenced checkpoint (metadata only)."""
    path = Path(ckpt)
    if not path.is_absolute():
        path = REPO / path
    if not path.exists():
        return {"exists": False}
    try:
        c = torch.load(path, map_location="cpu", weights_only=False)
        sd = c.get("model_state_dict", c)
        if "encoder_version" in c:
            version = int(c["encoder_version"])
        else:
            version = 2 if any(k.startswith("pos_embedding.") for k in sd) else 1
        emb = sd.get("embedding.weight")
        return {
            "exists": True,
            "encoder_version": version,
            "vocab_size": int(c.get("vocab_size", emb.shape[0] if emb is not None else -1)),
        }
    except Exception as exc:
        return {"exists": True, "error": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--expect", type=int, default=None, help="fail unless every ref is this version"
    )
    parser.add_argument("--json", type=str, default=None, help="write full report to this path")
    args = parser.parse_args()

    refs = collect_references()
    unique_ckpts = sorted({r["ckpt"] for r in refs if r.get("ckpt")})
    resolved = {ck: resolve_ckpt(ck) for ck in unique_ckpts}

    by_ckpt: dict[str, list[dict]] = defaultdict(list)
    for r in refs:
        if r.get("ckpt"):
            by_ckpt[r["ckpt"]].append(r)

    bad = []
    print(f"{len(refs)} referencias a checkpoint en {len(by_ckpt)} checkpoints distintos\n")
    for ck in unique_ckpts:
        info = resolved[ck]
        if not info.get("exists"):
            tag = "MISSING"
        elif "error" in info:
            tag = f"ERROR: {info['error']}"
        else:
            tag = f"v{info['encoder_version']} vocab={info['vocab_size']}"
        marker = ""
        if args.expect is not None and info.get("encoder_version") != args.expect:
            marker = "  <<< FUERA DE VERSION ESPERADA"
            bad.append(ck)
        print(f"[{tag}] {ck}{marker}")
        for r in sorted(by_ckpt[ck], key=lambda x: x["source"]):
            print(f"    - {r['source']}  ({r['where'].lstrip('.')})")
        print()

    versions = {info.get("encoder_version") for info in resolved.values() if info.get("exists")}
    if len(versions) > 1:
        print(f"AVISO: la cadena mezcla versiones de encoder: {sorted(v for v in versions if v)}")

    if args.json:
        out = {"references": refs, "checkpoints": resolved}
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=1))
        print(f"reporte JSON: {args.json}")

    if args.expect is not None and bad:
        print(f"\nFALLO: {len(bad)} checkpoints fuera de la version esperada v{args.expect}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
