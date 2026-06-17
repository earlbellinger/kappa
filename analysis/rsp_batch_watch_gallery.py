from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_WORKSPACE = ROOT / "rsp_batch_runs"
DEFAULT_PYTHON = Path(r"C:\Program Files\Python311\python.exe")
GALLERY_SCRIPT = ROOT / "rsp_batch_make_gallery.py"
FINISHED_VIEWER_SCRIPT = ROOT / "rsp_batch_make_finished_viewer.py"
LIVE_STATUS_SCRIPT = ROOT / "rsp_batch_live_status.py"
AUDIT_SCRIPT = ROOT / "rsp_batch_audit.py"
CYCLE_DIAGNOSTICS_SCRIPT = ROOT / "rsp_batch_cycle_diagnostics.py"
CONVERGENCE_SCRIPT = ROOT / "rsp_batch_convergence.py"
CONVERGENCE_TRENDS_SCRIPT = ROOT / "rsp_batch_convergence_trends.py"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh the local RSP batch gallery while the batch is active.")
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable))
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--refresh-seconds", type=int, default=180)
    parser.add_argument("--batch-status", type=Path, default=None)
    parser.add_argument("--log", type=Path, default=None)
    return parser.parse_args()


def read_status(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def read_active_batch_status(output_dir: Path) -> tuple[dict, Path | None]:
    candidates: list[tuple[float, Path, dict]] = []
    for path in output_dir.glob("batch*_status.json"):
        data = read_status(path)
        if isinstance(data, dict):
            candidates.append((path.stat().st_mtime, path, data))
    if not candidates:
        return {}, None
    running = [item for item in candidates if item[2].get("status") == "running"]
    selected = max(running, key=lambda item: item[0]) if running else max(candidates, key=lambda item: item[0])
    return selected[2], selected[1]


def append_log(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip() + "\n")


def write_refresh_marker(workspace: Path, log_path: Path, status: str) -> Path:
    marker = workspace / "output" / "gallery_refresh_in_progress.json"
    marker.write_text(
        json.dumps(
            {
                "status": status,
                "updated_at": now_iso(),
                "pid": os.getpid(),
                "log": str(log_path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return marker


def rebuild(args: argparse.Namespace, log_path: Path) -> int:
    marker = write_refresh_marker(args.workspace, log_path, "running")
    status_command = [
        str(args.python),
        str(LIVE_STATUS_SCRIPT),
        "--workspace",
        str(args.workspace),
    ]
    try:
        status_completed = subprocess.run(status_command, cwd=ROOT, capture_output=True, text=True)
        append_log(log_path, f"[{now_iso()}] live status returncode={status_completed.returncode}")
        if status_completed.stdout.strip():
            append_log(log_path, status_completed.stdout.strip())
        if status_completed.stderr.strip():
            append_log(log_path, status_completed.stderr.strip())

        audit_command = [
            str(args.python),
            str(AUDIT_SCRIPT),
            "--workspace",
            str(args.workspace),
            "--allow-incomplete",
        ]
        audit_completed = subprocess.run(audit_command, cwd=ROOT, capture_output=True, text=True)
        append_log(log_path, f"[{now_iso()}] audit returncode={audit_completed.returncode}")
        if audit_completed.stdout.strip():
            append_log(log_path, audit_completed.stdout.strip())
        if audit_completed.stderr.strip():
            append_log(log_path, audit_completed.stderr.strip())

        cycle_command = [
            str(args.python),
            str(CYCLE_DIAGNOSTICS_SCRIPT),
            "--workspace",
            str(args.workspace),
        ]
        cycle_completed = subprocess.run(cycle_command, cwd=ROOT, capture_output=True, text=True)
        append_log(log_path, f"[{now_iso()}] cycle diagnostics returncode={cycle_completed.returncode}")
        if cycle_completed.stdout.strip():
            append_log(log_path, cycle_completed.stdout.strip())
        if cycle_completed.stderr.strip():
            append_log(log_path, cycle_completed.stderr.strip())

        convergence_command = [
            str(args.python),
            str(CONVERGENCE_SCRIPT),
            "--workspace",
            str(args.workspace),
        ]
        convergence_completed = subprocess.run(convergence_command, cwd=ROOT, capture_output=True, text=True)
        append_log(log_path, f"[{now_iso()}] convergence returncode={convergence_completed.returncode}")
        if convergence_completed.stdout.strip():
            append_log(log_path, convergence_completed.stdout.strip())
        if convergence_completed.stderr.strip():
            append_log(log_path, convergence_completed.stderr.strip())

        convergence_trends_command = [
            str(args.python),
            str(CONVERGENCE_TRENDS_SCRIPT),
            "--workspace",
            str(args.workspace),
        ]
        convergence_trends_completed = subprocess.run(convergence_trends_command, cwd=ROOT, capture_output=True, text=True)
        append_log(log_path, f"[{now_iso()}] convergence trends returncode={convergence_trends_completed.returncode}")
        if convergence_trends_completed.stdout.strip():
            append_log(log_path, convergence_trends_completed.stdout.strip())
        if convergence_trends_completed.stderr.strip():
            append_log(log_path, convergence_trends_completed.stderr.strip())

        command = [
            str(args.python),
            str(GALLERY_SCRIPT),
            "--workspace",
            str(args.workspace),
            "--refresh-seconds",
            str(args.refresh_seconds),
        ]
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        append_log(log_path, f"[{now_iso()}] gallery rebuild returncode={completed.returncode}")
        if completed.stdout.strip():
            append_log(log_path, completed.stdout.strip())
        if completed.stderr.strip():
            append_log(log_path, completed.stderr.strip())

        viewer_command = [
            str(args.python),
            str(FINISHED_VIEWER_SCRIPT),
            "--workspace",
            str(args.workspace),
        ]
        viewer_completed = subprocess.run(viewer_command, cwd=ROOT, capture_output=True, text=True)
        append_log(log_path, f"[{now_iso()}] finished viewer rebuild returncode={viewer_completed.returncode}")
        if viewer_completed.stdout.strip():
            append_log(log_path, viewer_completed.stdout.strip())
        if viewer_completed.stderr.strip():
            append_log(log_path, viewer_completed.stderr.strip())
        return (
            status_completed.returncode
            or audit_completed.returncode
            or cycle_completed.returncode
            or convergence_completed.returncode
            or convergence_trends_completed.returncode
            or completed.returncode
            or viewer_completed.returncode
        )
    finally:
        try:
            marker.unlink(missing_ok=True)
        except OSError as exc:
            append_log(log_path, f"[{now_iso()}] could not remove refresh marker: {exc!r}")


def strict_audit(args: argparse.Namespace, log_path: Path) -> int:
    command = [
        str(args.python),
        str(AUDIT_SCRIPT),
        "--workspace",
        str(args.workspace),
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    append_log(log_path, f"[{now_iso()}] strict audit returncode={completed.returncode}")
    if completed.stdout.strip():
        append_log(log_path, completed.stdout.strip())
    if completed.stderr.strip():
        append_log(log_path, completed.stderr.strip())
    return completed.returncode


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    output_dir = workspace / "output"
    explicit_status_path = args.batch_status.resolve() if args.batch_status else None
    initial_status_path = explicit_status_path or read_active_batch_status(output_dir)[1] or output_dir / "batch_remaining_006_009_status.json"
    status_path = initial_status_path.resolve()
    log_path = (args.log or output_dir / "gallery_watch.log").resolve()

    append_log(log_path, f"[{now_iso()}] watcher started; status={status_path}")
    last_status = None
    while True:
        rebuild(args, log_path)
        if explicit_status_path is None:
            status, selected_path = read_active_batch_status(output_dir)
            if selected_path is not None and selected_path.resolve() != status_path:
                status_path = selected_path.resolve()
                append_log(log_path, f"[{now_iso()}] watcher switched status={status_path}")
        else:
            status = read_status(status_path)
        status_text = status.get("status") if isinstance(status, dict) else None
        if status_text != last_status:
            append_log(log_path, f"[{now_iso()}] batch status={status_text}")
            last_status = status_text
        if status_text == "complete":
            strict_audit(args, log_path)
            rebuild(args, log_path)
            append_log(log_path, f"[{now_iso()}] watcher exiting")
            return 0
        time.sleep(max(10, args.interval_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
