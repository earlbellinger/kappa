from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_WORKSPACE = ROOT / "rsp_batch_runs"
DEFAULT_PYTHON = Path(r"C:\Program Files\Python311\python.exe")
RUNNER = ROOT / "rsp_batch_run.py"
AUTO_RESUME_STAGES = {"create", "continue_saturation"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run several prepared RSP batch models sequentially.")
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--models", nargs="+", default=["model_006", "model_007", "model_008", "model_009"])
    parser.add_argument("--stage", default="all", choices=["create", "continue_saturation", "restart", "deep2cycles", "final_cycle", "plot", "verify", "mesa", "all"])
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable))
    parser.add_argument("--bash", default="bash")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument(
        "--no-auto-resume-continuation",
        action="store_true",
        help="Disable retrying a failed create/continuation stage from the latest saved photo.",
    )
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--status", type=Path, default=None)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def latest_failed_stage(workspace: Path, model: str) -> str | None:
    manifest = json.loads((workspace / "inputs" / "manifest.json").read_text())
    record = next((row for row in manifest if row["model_id"] == model or row["run_name"] == model), None)
    if record is None:
        return None
    status_path = Path(record["output_dir"]) / "run_status.json"
    status = read_json(status_path)
    stages = status.get("stages", {})
    for stage in ("create", "continue_saturation", "restart", "deep2cycles", "final_cycle", "plot", "verify"):
        if stages.get(stage, {}).get("status") == "failed":
            return stage
    return None


def run_command(command: list[str], log_path: Path) -> int:
    with log_path.open("ab") as log:
        header = f"\n\n[{now_iso()}] $ {' '.join(command)}\n".encode("utf-8", "replace")
        log.write(header)
        log.flush()
        completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        footer = f"[{now_iso()}] returncode={completed.returncode}\n".encode("utf-8", "replace")
        log.write(footer)
        log.flush()
    return completed.returncode


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    output_dir = workspace / "output"
    log_path = (args.log or output_dir / "batch_remaining_006_009.log").resolve()
    status_path = (args.status or output_dir / "batch_remaining_006_009_status.json").resolve()

    batch_status = {
        "driver_pid": os.getpid(),
        "started_at": now_iso(),
        "ended_at": None,
        "workspace": str(workspace),
        "models": args.models,
        "stage": args.stage,
        "log": str(log_path),
        "results": {},
        "status": "running",
    }
    write_json(status_path, batch_status)

    overall_code = 0
    for model in args.models:
        batch_status["current_model"] = model
        batch_status["results"].setdefault(model, {})
        batch_status["results"][model].update({"status": "running", "started_at": now_iso()})
        write_json(status_path, batch_status)

        command = [
            str(args.python),
            str(RUNNER),
            "--workspace",
            str(workspace),
            "--model",
            model,
            "--stage",
            args.stage,
            "--bash",
            args.bash,
        ]
        returncode = run_command(command, log_path)

        retry_code = None
        retry_stage = None
        failed_stage = latest_failed_stage(workspace, model) if returncode != 0 else None
        if returncode != 0 and not args.no_auto_resume_continuation and failed_stage in AUTO_RESUME_STAGES:
            retry_stage = failed_stage
            resume_command = [
                str(args.python),
                str(RUNNER),
                "--workspace",
                str(workspace),
                "--model",
                model,
                "--stage",
                retry_stage,
                "--bash",
                args.bash,
                "--resume-from-latest-photo",
            ]
            retry_code = run_command(resume_command, log_path)
            if retry_code == 0:
                returncode = run_command(command, log_path)

        result_status = "complete" if returncode == 0 else "failed"
        batch_status["results"][model].update(
            {
                "status": result_status,
                "ended_at": now_iso(),
                "returncode": returncode,
                "resume_stage": retry_stage,
                "resume_returncode": retry_code,
                "continuation_resume_returncode": retry_code if retry_stage == "continue_saturation" else None,
            }
        )
        write_json(status_path, batch_status)

        if returncode != 0:
            overall_code = returncode
            if args.stop_on_failure:
                break

    batch_status["ended_at"] = now_iso()
    batch_status.pop("current_model", None)
    batch_status["status"] = "complete" if overall_code == 0 else "failed"
    write_json(status_path, batch_status)
    return overall_code


if __name__ == "__main__":
    raise SystemExit(main())
