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
RUNNER = ROOT / "rsp_batch_run.py"
LIVE_STATUS = ROOT / "rsp_batch_live_status.py"
CONVERGENCE = ROOT / "rsp_batch_convergence.py"
CONVERGENCE_TRENDS = ROOT / "rsp_batch_convergence_trends.py"
CONVERGENCE_FORECAST = ROOT / "rsp_batch_convergence_forecast.py"
CONVERGENCE_GATE_AUDIT = ROOT / "rsp_batch_convergence_gate_audit.py"
AUDIT = ROOT / "rsp_batch_audit.py"
PHASE_SEAM_AUDIT = ROOT / "rsp_batch_phase_seam_audit.py"
CYCLE_BOUNDARY_AUDIT = ROOT / "rsp_batch_cycle_boundary_audit.py"
GALLERY = ROOT / "rsp_batch_make_gallery.py"
FINISHED_VIEWER = ROOT / "rsp_batch_make_finished_viewer.py"
FOURIER_FIXED = ROOT / "plot_fourier_fixed_cells_vs_logT.py"
PRODUCT_STAGES = ("restart", "deep2cycles", "final_cycle", "plot", "verify")
FOURIER_FIXED_SCHEMA_VERSION = "amplitude-plus-thermodynamic-peak-lags-v1"

from rsp_batch_run import animation_product_is_current


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
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Models to monitor. Defaults to the active model set reported by live_status.json.",
    )
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


def active_model_ids(workspace: Path) -> list[str]:
    live = read_json(workspace / "output" / "live_status.json")
    if not isinstance(live, dict):
        return []
    ids: list[str] = []
    batch = live.get("batch_status")
    if isinstance(batch, dict):
        for model_id in batch.get("models", []):
            text = str(model_id)
            if text not in ids:
                ids.append(text)
        for model_id in batch.get("current_models", []):
            text = str(model_id)
            if text not in ids:
                ids.append(text)
    for row in live.get("models", []):
        if not isinstance(row, dict):
            continue
        model_id = row.get("model_id")
        if model_id and row.get("active_stage"):
            text = str(model_id)
            if text not in ids:
                ids.append(text)
    return ids


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


def fixed_fourier_product_is_current(record: dict) -> tuple[bool, str]:
    output_dir = Path(str(record["output_dir"]))
    prefix = str(record["prefix"])
    required = {
        "png": output_dir / f"{prefix}_fourier_fixed_cells_vs_logT.png",
        "csv": output_dir / f"{prefix}_fourier_fixed_cells_vs_logT.csv",
        "summary": output_dir / f"{prefix}_fourier_fixed_cells_vs_logT_summary.json",
    }
    missing = [label for label, path in required.items() if not path.exists()]
    if missing:
        return False, "fixed-cell Fourier product is missing " + ", ".join(missing)
    summary = read_json(required["summary"])
    if not isinstance(summary, dict):
        return False, "fixed-cell Fourier summary could not be parsed"
    if summary.get("fourier_fixed_schema_version") != FOURIER_FIXED_SCHEMA_VERSION:
        return False, "fixed-cell Fourier product predates the amplitude/thermodynamic phase-lag schema"
    if not summary.get("thermodynamic_peak_phase_panel"):
        return False, "fixed-cell Fourier product is missing thermodynamic peak phase-lag panel metadata"
    return True, "fixed-cell Fourier product exists"


def verified_product_current(record: dict) -> tuple[bool, str]:
    if not stage_complete(record, "verify"):
        return False, "verify stage is not complete"
    animation_current, animation_reason = animation_product_is_current(record, Path(str(record["output_dir"])))
    if not animation_current:
        return False, animation_reason
    fourier_current, fourier_reason = fixed_fourier_product_is_current(record)
    if not fourier_current:
        return False, fourier_reason
    return True, f"{animation_reason}; {fourier_reason}"


def model_lock_path(record: dict) -> Path:
    return Path(str(record["output_dir"])) / ".model_run.lock"


def run_command(command: list[str], log_path: Path) -> int:
    append_log(log_path, "$ " + " ".join(command))
    with log_path.open("ab") as handle:
        completed = subprocess.run(command, cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT)
    append_log(log_path, f"returncode={completed.returncode}")
    return completed.returncode


def build_fixed_fourier_product(args: argparse.Namespace, record: dict, log_path: Path) -> int:
    command = [
        str(args.python),
        str(FOURIER_FIXED),
        "--run-dir",
        str(Path(str(record["run_dir"]))),
        "--output-dir",
        str(Path(str(record["output_dir"]))),
        "--prefix",
        str(record["prefix"]),
        "--fit-harmonics",
        "14",
    ]
    return run_command(command, log_path)


def refresh_analysis(args: argparse.Namespace, log_path: Path, models: list[str]) -> int:
    workspace = args.workspace.resolve()
    commands = [
        [str(args.python), str(LIVE_STATUS), "--workspace", str(workspace)],
        [str(args.python), str(CONVERGENCE), "--workspace", str(workspace), "--models", *models, "--merge-existing"],
        [str(args.python), str(CONVERGENCE_TRENDS), "--workspace", str(workspace), "--models", *models, "--merge-existing"],
        [str(args.python), str(CONVERGENCE_FORECAST), "--workspace", str(workspace)],
        [str(args.python), str(CONVERGENCE_GATE_AUDIT), "--workspace", str(workspace), "--models", *models],
        [str(args.python), str(AUDIT), "--workspace", str(workspace), "--allow-incomplete"],
        [str(args.python), str(PHASE_SEAM_AUDIT), "--workspace", str(workspace)],
        [str(args.python), str(CYCLE_BOUNDARY_AUDIT), "--workspace", str(workspace)],
        [str(args.python), str(GALLERY), "--workspace", str(workspace)],
        [str(args.python), str(FINISHED_VIEWER), "--workspace", str(workspace)],
    ]
    for command in commands:
        code = run_command(command, log_path)
        if code != 0:
            return code
    return 0


def build_products(args: argparse.Namespace, model_id: str, log_path: Path, models: list[str]) -> int:
    workspace = args.workspace.resolve()
    record = manifest_by_model(workspace).get(model_id)
    if record is None:
        append_log(log_path, f"{model_id} missing manifest record; cannot build fixed-cell Fourier product")
        return 1
    animation_current = False
    if stage_complete(record, "verify"):
        animation_current, animation_reason = animation_product_is_current(record, Path(str(record["output_dir"])))
        append_log(log_path, f"{model_id} animation current check: {animation_current} ({animation_reason})")
    if not animation_current:
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
    fourier_current, fourier_reason = fixed_fourier_product_is_current(record)
    append_log(log_path, f"{model_id} fixed-cell Fourier current check: {fourier_current} ({fourier_reason})")
    if not fourier_current:
        code = build_fixed_fourier_product(args, record, log_path)
        if code != 0:
            return code
    return refresh_analysis(args, log_path, models)


def supervise_once(args: argparse.Namespace, log_path: Path) -> dict[str, object]:
    live_code = run_command(
        [str(args.python), str(LIVE_STATUS), "--workspace", str(args.workspace.resolve())],
        log_path,
    )
    if live_code != 0:
        return {"result": "live_status_failed", "returncode": live_code, "models": []}

    models = list(args.models) if args.models else active_model_ids(args.workspace)
    if not models:
        return {"result": "no_active_models", "models": []}

    refresh_code = refresh_analysis(args, log_path, models)
    if refresh_code != 0:
        return {"result": "analysis_refresh_failed", "returncode": refresh_code, "models": models}

    records = manifest_by_model(args.workspace)
    convergence = convergence_by_model(args.workspace)
    actions: list[dict[str, object]] = []
    for model_id in models:
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
        verified_current, verified_reason = verified_product_current(record)
        if verified_current:
            actions.append(
                {
                    "model_id": model_id,
                    "action": "already_verified",
                    "reason": verified_reason,
                }
            )
            continue

        append_log(
            log_path,
            f"{model_id} converged and unlocked; building products ({verified_reason})",
        )
        code = build_products(args, model_id, log_path, models)
        actions.append({"model_id": model_id, "action": "built_products", "returncode": code})
        if code != 0:
            return {"result": "product_build_failed", "returncode": code, "actions": actions, "models": models}

    return {"result": "ok", "actions": actions, "models": models}


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
                "models": result.get("models", args.models or []),
                "log": str(log_path),
                "pid": os.getpid(),
                "interval_seconds": int(args.interval_seconds),
                **result,
            },
        )
        if args.once or result.get("result") not in {"ok"}:
            return 0 if result.get("result") == "ok" else int(result.get("returncode") or 1)
        time.sleep(max(30, args.interval_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
