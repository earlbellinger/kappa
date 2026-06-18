from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_WORKSPACE = ROOT / "rsp_batch_runs"
DEFAULT_RUNNER = ROOT / "rsp_batch_run.py"


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def write_status(path: Path, status: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run rsp_batch_run continue_saturation resumes sequentially for selected models."
    )
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--bash", default="bash")
    parser.add_argument("--resume-max-num-periods", type=int, default=5000)
    parser.add_argument("--status-file", type=Path)
    parser.add_argument("--models", nargs="+", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.workspace / "output"
    status_file = args.status_file or output_dir / "batch_strict_continue_status.json"
    status: dict[str, object] = {
        "status": "running",
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "workspace": str(args.workspace),
        "models": args.models,
        "results": {},
    }
    write_status(status_file, status)

    for model in args.models:
        log_path = output_dir / f"strict_continue_{model}.stdout.log"
        command = [
            str(args.python),
            str(args.runner),
            "--workspace",
            str(args.workspace),
            "--model",
            model,
            "--stage",
            "continue_saturation",
            "--bash",
            args.bash,
            "--force",
            "--resume-from-latest-photo",
            "--resume-max-num-periods",
            str(args.resume_max_num_periods),
        ]
        result_record = {
            "status": "running",
            "started_at": utc_now(),
            "log": str(log_path),
            "command": command,
        }
        status["current_model"] = model
        status["results"][model] = result_record
        status["updated_at"] = utc_now()
        write_status(status_file, status)

        with log_path.open("a", encoding="utf-8", errors="replace") as log:
            log.write(f"\n[{utc_now()}] starting {' '.join(command)}\n")
            log.flush()
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
            log.write(f"\n[{utc_now()}] finished with returncode {completed.returncode}\n")

        result_record["status"] = "complete" if completed.returncode == 0 else "failed"
        result_record["ended_at"] = utc_now()
        result_record["returncode"] = completed.returncode
        status["updated_at"] = utc_now()
        write_status(status_file, status)
        if completed.returncode != 0:
            status["status"] = "failed"
            status["ended_at"] = utc_now()
            status["updated_at"] = utc_now()
            write_status(status_file, status)
            raise SystemExit(completed.returncode)

    status["status"] = "complete"
    status["ended_at"] = utc_now()
    status["updated_at"] = utc_now()
    status.pop("current_model", None)
    write_status(status_file, status)


if __name__ == "__main__":
    main()
