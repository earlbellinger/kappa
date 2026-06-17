from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_WORKSPACE = ROOT / "rsp_batch_runs"
DEFAULT_PYTHON = Path(r"C:\Program Files\Python311\python.exe")
RUN_SCRIPT = ROOT / "rsp_batch_run.py"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate batch visualization products after a scaling change.")
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable))
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--skip-verify", action="store_true")
    return parser.parse_args()


def run_stage(args: argparse.Namespace, model: str, stage: str) -> int:
    command = [
        str(args.python),
        str(RUN_SCRIPT),
        "--workspace",
        str(args.workspace),
        "--model",
        model,
        "--stage",
        stage,
        "--force",
        "--allow-unconverged-products",
    ]
    print(f"[{now_iso()}] {stage} {model}", flush=True)
    completed = subprocess.run(command, cwd=ROOT)
    print(f"[{now_iso()}] {stage} {model} exit={completed.returncode}", flush=True)
    return int(completed.returncode)


def main() -> None:
    args = parse_args()
    worst_returncode = 0
    for model in args.models:
        worst_returncode = max(worst_returncode, run_stage(args, model, "plot"))
        if not args.skip_verify:
            worst_returncode = max(worst_returncode, run_stage(args, model, "verify"))
    print(f"[{now_iso()}] done exit={worst_returncode}", flush=True)
    raise SystemExit(worst_returncode)


if __name__ == "__main__":
    main()
