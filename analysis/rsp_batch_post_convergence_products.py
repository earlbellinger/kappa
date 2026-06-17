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
LIVE_STATUS = ROOT / "rsp_batch_live_status.py"
CONVERGENCE = ROOT / "rsp_batch_convergence.py"
AUDIT = ROOT / "rsp_batch_audit.py"
GALLERY = ROOT / "rsp_batch_make_gallery.py"
FINISHED_VIEWER = ROOT / "rsp_batch_make_finished_viewer.py"
PRODUCT_STAGES = ("restart", "deep2cycles", "final_cycle", "plot", "verify")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Watch converged RSP batch models and build canonical products only "
            "after strict convergence passes and the model lock is free."
        )
    )
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--models", nargs="+", default=["model_007", "model_008", "model_009"])
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable))
    parser.add_argument("--bash", default="bash")
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--status", type=Path, default=None)
    return parser.parse_args()


def read_json(path: Path) -> dict | list:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{now_iso()}] {message.rstrip()}\n")


def manifest_by_model(workspace: Path) -> dict[str, dict]:
    manifest = read_json(workspace / "inputs" / "manifest.json")
    if not isinstance(manifest, list):
        return {}
    return {str(row.get("model_id")): row for row in manifest if isinstance(row, dict)}


def convergence_by_model(workspace: Path) -> dict[str, dict]:
    convergence = read_json(workspace / "output" / "convergence_summary_last100.json")
    rows = convergence.get("models", []) if isinstance(convergence, dict) else []
    return {
        str(row.get("model_id")): row
        for row in rows
        if isinstance(row, dict) and row.get("model_id")
    }


def run_status_for(record: dict) -> dict:
    output_dir = Path(str(record["output_dir"]))
    status = read_json(output_dir / "run_status.json")
    return status if isinstance(status, dict) else {}


def stage_complete(record: dict, stage: str) -> bool:
    status = run_status_for(record)
    stages = status.get("stages", {}) if isinstance(status, dict) else {}
    payload = stages.get(stage, {}) if isinstance(stages, dict) else {}
    return isinstance(payload, dict) and payload.get("status") == "complete"


def model_lock_path(record: dict) -> Path:
    return Path(str(record["output_dir"])) / ".model_run.lock"


def run_command(command: list[str], log_path: Path) -> int:
    append_log(log_path, "$ " + " ".join(command))
    with log_path.open("ab") as handle:
        completed = subprocess.run(command, cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT)
    append_log(log_path, f"returncode={completed.returncode}")
    return completed.returncode


def refresh_analysis(args: argparse.Namespace, log_path: Path) -> int:
    workspace = args.workspace.resolve()
    commands = [
        [str(args.python), str(LIVE_STATUS), "--workspace", str(workspace)],
        [str(args.python), str(CONVERGENCE), "--workspace", str(workspace), "--models", *args.models, "--merge-existing"],
        [str(args.python), str(AUDIT), "--workspace", str(workspace), "--allow-incomplete"],
        [str(args.python), str(GALLERY), "--workspace", str(workspace)],
        [str(args.python), str(FINISHED_VIEWER), "--workspace", str(workspace)],
    ]
    for command in commands:
        code = run_command(command, log_path)
        if code != 0:
            return code
    return 0


def build_products(args: argparse.Namespace, model_id: str, log_path: Path) -> int:
    workspace = args.workspace.resolve()
    for stage in PRODUCT_STAGES:
        command = [
            str(args.python),
            str(RUNNER),
            "--workspace",
            str(workspace),
            "--model",
            model_id,
            "--stage",
            stage,
            "--bash",
            args.bash,
            "--force",
        ]
        code = run_command(command, log_path)
        if code != 0:
            return code
    return refresh_analysis(args, log_path)


def supervise_once(args: argparse.Namespace, log_path: Path) -> dict[str, object]:
    refresh_code = refresh_analysis(args, log_path)
    if refresh_code != 0:
        return {"result": "analysis_refresh_failed", "returncode": refresh_code}

    records = manifest_by_model(args.workspace)
    convergence = convergence_by_model(args.workspace)
    actions: list[dict[str, object]] = []
    for model_id in args.models:
        record = records.get(model_id)
        row = convergence.get(model_id, {})
        if record is None:
            actions.append({"model_id": model_id, "action": "missing_manifest_record"})
            continue
        if row.get("converged_exact") is not True:
            actions.append(
                {
                    "model_id": model_id,
                    "action": "waiting_for_convergence",
                    "last_period_number": row.get("last_period_number"),
                    "delta_r_fractional_peak_to_peak_last_window": row.get(
                        "delta_r_fractional_peak_to_peak_last_window"
                    ),
                }
            )
            continue
        if model_lock_path(record).exists():
            actions.append({"model_id": model_id, "action": "waiting_for_model_lock"})
            continue
        if stage_complete(record, "verify"):
            actions.append({"model_id": model_id, "action": "already_verified"})
            continue

        append_log(log_path, f"{model_id} converged and unlocked; building products")
        code = build_products(args, model_id, log_path)
        actions.append({"model_id": model_id, "action": "built_products", "returncode": code})
        if code != 0:
            return {"result": "product_build_failed", "returncode": code, "actions": actions}

    return {"result": "ok", "actions": actions}


def main() -> int:
    args = parse_args()
    args.workspace = args.workspace.resolve()
    output_dir = args.workspace / "output"
    log_path = (args.log or output_dir / "post_convergence_products.log").resolve()
    status_path = (args.status or output_dir / "post_convergence_products_status.json").resolve()

    append_log(log_path, "post-convergence product watcher started")
    while True:
        result = supervise_once(args, log_path)
        write_json(
            status_path,
            {
                "updated_at": now_iso(),
                "workspace": str(args.workspace),
                "models": args.models,
                "log": str(log_path),
                **result,
            },
        )
        if args.once or result.get("result") not in {"ok"}:
            return 0 if result.get("result") == "ok" else int(result.get("returncode") or 1)
        time.sleep(max(30, args.interval_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
