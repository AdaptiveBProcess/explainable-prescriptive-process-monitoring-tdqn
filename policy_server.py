"""CLI entry point for the XPPM policy server."""

from __future__ import annotations

from pathlib import Path

import uvicorn

from xppm.serve.server import app, set_bundle_dir

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default="artifacts/deploy/v1", help="Deploy bundle directory")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    set_bundle_dir(Path(args.bundle))
    uvicorn.run(app, host=args.host, port=args.port)
