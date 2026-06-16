from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_WORKSPACE = ROOT / "rsp_batch_runs"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> object:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{now_iso()}] {message.rstrip()}\n")


def run_wsl(command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["wsl", "-e", "bash", "-lc", command],
        capture_output=True,
        text=True,
        timeout=30,
    )


def wsl_process_exists(pid: int) -> bool:
    result = run_wsl(f"kill -0 {pid}")
    return result.returncode == 0


def signal_wsl_process(pid: int, signal: str) -> subprocess.CompletedProcess[str]:
    return run_wsl(f"kill -{signal} {pid}")


def model_stage_status(workspace: Path, model_id: str, stage: str) -> str | None:
    manifest = read_json(workspace / "inputs" / "manifest.json")
    if not isinstance(manifest, list):
        return None
    for record in manifest:
        if not isinstance(record, dict) or record.get("model_id") != model_id:
            continue
        output_dir = Path(str(record.get("output_dir", "")))
        status = read_json(output_dir / "run_status.json")
        stages = status.get("stages", {}) if isinstance(status, dict) else {}
        stage_payload = stages.get(stage, {}) if isinstance(stages, dict) else {}
        if isinstance(stage_payload, dict):
            return stage_payload.get("status")
    return None


def should_resume(stage_status: str | None, mode: str) -> bool:
    if mode == "complete":
        return stage_status == "complete"
    if mode == "not_running":
        return stage_status is not None and stage_status != "running"
    raise ValueError(f"Unknown resume mode: {mode}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Temporarily pause one WSL process and resume it after a batch stage changes.")
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--pid", type=int, required=True, help="WSL PID to pause and later resume.")
    parser.add_argument("--resume-model", default="model_006")
    parser.add_argument("--resume-after-stage", default="continue_saturation")
    parser.add_argument(
        "--resume-when-stage-status",
        choices=("complete", "not_running"),
        default="not_running",
        help="Resume when the watched stage is complete, or when it has any known non-running status.",
    )
    parser.add_argument("--interval-seconds", type=int, default=120)
    parser.add_argument("--status", type=Path, default=None)
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    status_path = args.status or workspace / "output" / "pause_resume_status.json"
    log_path = args.log or workspace / "output" / "pause_resume.log"

    append_log(log_path, f"pause watcher started for WSL pid={args.pid}")
    if not wsl_process_exists(args.pid):
        payload = {
            "updated_at": now_iso(),
            "status": "missing_before_pause",
            "paused_pid": args.pid,
            "resume_model": args.resume_model,
            "resume_after_stage": args.resume_after_stage,
            "resume_when_stage_status": args.resume_when_stage_status,
        }
        write_json(status_path, payload)
        append_log(log_path, f"pid {args.pid} missing before pause")
        return 1

    pause_result = signal_wsl_process(args.pid, "STOP")
    payload = {
        "updated_at": now_iso(),
        "status": "paused" if pause_result.returncode == 0 else "pause_failed",
        "paused_pid": args.pid,
        "pause_returncode": pause_result.returncode,
        "pause_stderr": pause_result.stderr.strip(),
        "resume_model": args.resume_model,
        "resume_after_stage": args.resume_after_stage,
        "resume_when_stage_status": args.resume_when_stage_status,
    }
    write_json(status_path, payload)
    append_log(log_path, f"sent STOP to pid={args.pid}; returncode={pause_result.returncode}")
    if pause_result.returncode != 0:
        return pause_result.returncode

    while True:
        stage_status = model_stage_status(workspace, args.resume_model, args.resume_after_stage)
        exists = wsl_process_exists(args.pid)
        payload.update(
            {
                "updated_at": now_iso(),
                "status": "paused_waiting" if exists else "paused_process_missing",
                "paused_pid_exists": exists,
                "resume_stage_status": stage_status,
                "resume_when_stage_status": args.resume_when_stage_status,
            }
        )
        write_json(status_path, payload)

        if not exists:
            append_log(log_path, f"pid {args.pid} disappeared before resume")
            return 1
        if should_resume(stage_status, args.resume_when_stage_status):
            resume_result = signal_wsl_process(args.pid, "CONT")
            payload.update(
                {
                    "updated_at": now_iso(),
                    "status": "resumed" if resume_result.returncode == 0 else "resume_failed",
                    "resume_returncode": resume_result.returncode,
                    "resume_stderr": resume_result.stderr.strip(),
                    "resume_stage_status": stage_status,
                }
            )
            write_json(status_path, payload)
            append_log(
                log_path,
                f"sent CONT to pid={args.pid}; stage_status={stage_status}; returncode={resume_result.returncode}",
            )
            return resume_result.returncode

        time.sleep(max(30, args.interval_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
