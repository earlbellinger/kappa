from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_WORKSPACE = ROOT / "rsp_batch_runs"
DEFAULT_PYTHON = Path(r"C:\Program Files\Python311\python.exe")
RUNNER = ROOT / "rsp_batch_run.py"
BATCH_RUNNER = ROOT / "rsp_batch_run_many.py"
RESUMABLE_STAGES = {"create", "continue_saturation"}
STAGE_ORDER = ("create", "continue_saturation", "restart", "deep2cycles", "final_cycle", "plot", "verify")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conservative recovery supervisor for the remaining RSP batch.")
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--models", nargs="+", default=["model_006", "model_007", "model_008", "model_009"])
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable))
    parser.add_argument("--bash", default="bash")
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--batch-status", type=Path, default=None)
    parser.add_argument("--batch-log", type=Path, default=None)
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--once", action="store_true", help="Check once and exit.")
    return parser.parse_args()


def read_json(path: Path) -> dict | list:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{now_iso()}] {message.rstrip()}\n")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def manifest_by_model(workspace: Path) -> dict[str, dict]:
    manifest = read_json(workspace / "inputs" / "manifest.json")
    if not isinstance(manifest, list):
        return {}
    return {str(row["model_id"]): row for row in manifest}


def run_status_for(record: dict) -> dict:
    output_dir = Path(str(record["output_dir"]))
    status = read_json(output_dir / "run_status.json")
    return status if isinstance(status, dict) else {}


def first_incomplete_model(workspace: Path, models: list[str]) -> str | None:
    records = manifest_by_model(workspace)
    for model in models:
        record = records.get(model)
        if record is None:
            return model
        status = run_status_for(record)
        stages = status.get("stages", {}) if isinstance(status, dict) else {}
        verify = stages.get("verify", {}) if isinstance(stages, dict) else {}
        if verify.get("status") != "complete":
            return model
    return None


def failed_stage(workspace: Path, model: str) -> str | None:
    records = manifest_by_model(workspace)
    record = records.get(model)
    if record is None:
        return None
    status = run_status_for(record)
    stages = status.get("stages", {}) if isinstance(status, dict) else {}
    if not isinstance(stages, dict):
        return None
    for stage in STAGE_ORDER:
        stage_payload = stages.get(stage, {})
        if isinstance(stage_payload, dict) and stage_payload.get("status") == "failed":
            return stage
    return None


def verified_stage_complete(record: dict) -> bool:
    status = run_status_for(record)
    stages = status.get("stages", {}) if isinstance(status, dict) else {}
    verify = stages.get("verify", {}) if isinstance(stages, dict) else {}
    return isinstance(verify, dict) and verify.get("status") == "complete"


def verification_needs_refresh(record: dict) -> bool:
    if not verified_stage_complete(record):
        return False
    verification_path = Path(str(record["output_dir"])) / "verification_summary.json"
    verification = read_json(verification_path)
    if not isinstance(verification, dict):
        return True
    return verification.get("phase_seam_ok") is not True


def run_command(command: list[str], cwd: Path, log_path: Path) -> int:
    append_log(log_path, "$ " + " ".join(command))
    with log_path.open("ab") as handle:
        completed = subprocess.run(command, cwd=str(cwd), stdout=handle, stderr=subprocess.STDOUT)
    append_log(log_path, f"returncode={completed.returncode}")
    return completed.returncode


def refresh_stale_verifications(args: argparse.Namespace, log_path: Path) -> int:
    workspace = args.workspace.resolve()
    records = manifest_by_model(workspace)
    refreshed: list[str] = []
    for model in args.models:
        record = records.get(model)
        if record is None or not verification_needs_refresh(record):
            continue
        command = [
            str(args.python),
            str(RUNNER),
            "--workspace",
            str(workspace),
            "--model",
            model,
            "--stage",
            "verify",
            "--force",
        ]
        append_log(log_path, f"refresh stale verification for {model}")
        code = run_command(command, ROOT, log_path)
        if code != 0:
            return code
        refreshed.append(model)
    if refreshed:
        append_log(log_path, "refreshed stale verifications: " + ", ".join(refreshed))
    return 0


def recover(args: argparse.Namespace, status_path: Path, batch_log_path: Path, log_path: Path) -> int:
    workspace = args.workspace.resolve()
    models = list(args.models)
    start_model = first_incomplete_model(workspace, models)
    if start_model is None:
        append_log(log_path, "all supervised models already verify complete")
        return 0
    remaining = models[models.index(start_model) :] if start_model in models else [start_model]
    stage = failed_stage(workspace, start_model)
    append_log(log_path, f"recovery target={start_model}; failed_stage={stage}; remaining={remaining}")

    if stage in RESUMABLE_STAGES:
        resume_command = [
            str(args.python),
            str(RUNNER),
            "--workspace",
            str(workspace),
            "--model",
            start_model,
            "--stage",
            stage,
            "--bash",
            args.bash,
            "--resume-from-latest-photo",
        ]
        resume_code = run_command(resume_command, ROOT, log_path)
        if resume_code != 0:
            return resume_code

    batch_command = [
        str(args.python),
        str(BATCH_RUNNER),
        "--workspace",
        str(workspace),
        "--models",
        *remaining,
        "--stage",
        "all",
        "--bash",
        args.bash,
        "--log",
        str(batch_log_path),
        "--status",
        str(status_path),
    ]
    return run_command(batch_command, ROOT, log_path)


def supervise_once(args: argparse.Namespace, status_path: Path, batch_log_path: Path, log_path: Path) -> str:
    refresh_code = refresh_stale_verifications(args, log_path)
    if refresh_code != 0:
        return f"verification_refresh_failed:{refresh_code}"

    batch_status = read_json(status_path)
    if not isinstance(batch_status, dict) or not batch_status:
        append_log(log_path, f"batch status missing at {status_path}")
        return "missing"
    status_text = str(batch_status.get("status", "unknown"))
    current = batch_status.get("current_model")
    append_log(log_path, f"batch status={status_text}; current_model={current}")
    if status_text == "failed":
        code = recover(args, status_path, batch_log_path, log_path)
        return "recovered" if code == 0 else f"recovery_failed:{code}"
    return status_text


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    output_dir = workspace / "output"
    status_path = (args.batch_status or output_dir / "batch_remaining_006_009_status.json").resolve()
    batch_log_path = (args.batch_log or output_dir / "batch_remaining_006_009.log").resolve()
    log_path = (args.log or output_dir / "batch_supervisor.log").resolve()
    supervisor_status_path = output_dir / "batch_supervisor_status.json"

    append_log(log_path, f"supervisor started; status={status_path}")
    while True:
        result = supervise_once(args, status_path, batch_log_path, log_path)
        write_json(
            supervisor_status_path,
            {
                "updated_at": now_iso(),
                "result": result,
                "workspace": str(workspace),
                "batch_status": str(status_path),
                "log": str(log_path),
            },
        )
        if args.once or result in {"complete", "recovered"} or result.startswith(("recovery_failed:", "verification_refresh_failed:")):
            return 0 if result in {"complete", "running", "recovered"} else 1
        time.sleep(max(30, args.interval_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
