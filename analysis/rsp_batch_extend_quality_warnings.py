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
AUDIT = ROOT / "rsp_batch_audit.py"
CYCLE_DIAGNOSTICS = ROOT / "rsp_batch_cycle_diagnostics.py"
CONVERGENCE = ROOT / "rsp_batch_convergence.py"
CONVERGENCE_TRENDS = ROOT / "rsp_batch_convergence_trends.py"
LIVE_STATUS = ROOT / "rsp_batch_live_status.py"
GALLERY = ROOT / "rsp_batch_make_gallery.py"
FINISHED_VIEWER = ROOT / "rsp_batch_make_finished_viewer.py"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extend completed RSP batch models that fail the clean-limit-cycle "
            "quality gate, then rebuild their canonical products."
        )
    )
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--models", nargs="*", default=None, help="Optional model ids to consider.")
    parser.add_argument("--caps", nargs="+", type=int, default=[5000], help="Resume RSP_max_num_periods caps to try.")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable))
    parser.add_argument("--bash", default="bash")
    parser.add_argument("--wait-for-batch", action="store_true")
    parser.add_argument("--wait-interval-seconds", type=int, default=300)
    parser.add_argument("--batch-status", type=Path, default=None)
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--status", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
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


def run_command(command: list[str], log_path: Path, dry_run: bool = False) -> int:
    append_log(log_path, "$ " + " ".join(command))
    if dry_run:
        append_log(log_path, "dry-run: command not executed")
        return 0
    with log_path.open("ab") as handle:
        completed = subprocess.run(command, cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT)
    append_log(log_path, f"returncode={completed.returncode}")
    return completed.returncode


def analysis_refresh(args: argparse.Namespace, log_path: Path, allow_incomplete: bool = True) -> int:
    commands = [
        [str(args.python), str(LIVE_STATUS), "--workspace", str(args.workspace)],
        [str(args.python), str(CYCLE_DIAGNOSTICS), "--workspace", str(args.workspace)],
        [str(args.python), str(CONVERGENCE), "--workspace", str(args.workspace)],
        [str(args.python), str(CONVERGENCE_TRENDS), "--workspace", str(args.workspace)],
        [
            str(args.python),
            str(AUDIT),
            "--workspace",
            str(args.workspace),
            *(["--allow-incomplete"] if allow_incomplete else []),
        ],
        [str(args.python), str(GALLERY), "--workspace", str(args.workspace)],
        [str(args.python), str(FINISHED_VIEWER), "--workspace", str(args.workspace)],
    ]
    for command in commands:
        code = run_command(command, log_path, args.dry_run)
        if code != 0:
            return code
    return 0


def strict_audit(args: argparse.Namespace, log_path: Path) -> int:
    return run_command(
        [str(args.python), str(AUDIT), "--workspace", str(args.workspace)],
        log_path,
        args.dry_run,
    )


def batch_status(args: argparse.Namespace) -> dict:
    status_path = args.batch_status or args.workspace / "output" / "batch_remaining_006_009_status.json"
    data = read_json(status_path)
    return data if isinstance(data, dict) else {}


def wait_for_batch(args: argparse.Namespace, log_path: Path, status_path: Path) -> int:
    while True:
        status = batch_status(args)
        status_text = status.get("status")
        current = status.get("current_model")
        ended_at = status.get("ended_at")
        waiting_status = "waiting_for_batch"
        terminal_failed = status_text == "failed" and ended_at
        if terminal_failed:
            waiting_status = "batch_terminal_failed"
        elif status_text in {"failed", "missing", None}:
            waiting_status = "waiting_for_batch_recovery"
        write_json(
            status_path,
            {
                "updated_at": now_iso(),
                "status": waiting_status,
                "batch_status": status_text,
                "current_model": current,
                "batch_ended_at": ended_at,
            },
        )
        append_log(log_path, f"batch status={status_text}; current_model={current}")
        if status_text == "complete":
            return 0
        if terminal_failed:
            append_log(log_path, "batch ended with failed status; proceeding to quality extension")
            return 0
        if status_text in {"failed", "missing", None}:
            append_log(log_path, "batch not complete; waiting for supervisor/recovery")
        time.sleep(max(30, args.wait_interval_seconds))


def convergence_by_model(output_dir: Path) -> dict[str, dict]:
    convergence = read_json(output_dir / "convergence_summary_last100.json")
    rows = convergence.get("models", []) if isinstance(convergence, dict) else []
    return {
        str(row.get("model_id")): row
        for row in rows
        if isinstance(row, dict) and row.get("model_id")
    }


def quality_warning_models(args: argparse.Namespace, log_path: Path) -> list[dict]:
    audit_path = args.workspace / "output" / "batch_audit_summary.json"
    output_dir = args.workspace / "output"
    audit = read_json(audit_path)
    if not isinstance(audit, dict):
        return []
    convergence = convergence_by_model(output_dir)
    selected = set(args.models or [])
    models = audit.get("models", [])
    result = []
    for model in models if isinstance(models, list) else []:
        if not isinstance(model, dict):
            continue
        model_id = str(model.get("model_id"))
        if selected and model_id not in selected:
            continue
        if model.get("registered_existing"):
            continue
        stages = model.get("stages", {})
        if isinstance(stages, dict) and any(
            isinstance(stage, dict) and stage.get("status") == "running"
            for stage in stages.values()
        ):
            append_log(log_path, f"skip {model_id}: model is still running")
            continue
        continue_stage = stages.get("continue_saturation") if isinstance(stages, dict) else None
        if isinstance(continue_stage, dict) and continue_stage.get("status") != "complete":
            append_log(log_path, f"skip {model_id}: continue_saturation is not complete")
            continue
        reasons = list(model.get("quality_warnings") or [])
        convergence_row = convergence.get(model_id)
        if not convergence_row:
            reasons.append("missing convergence summary")
        elif convergence_row.get("converged_exact") is not True:
            source_kind = convergence_row.get("source_kind")
            gamma = convergence_row.get("gamma_peak_to_peak_last_window")
            period = convergence_row.get("period_fractional_peak_to_peak_last_window")
            delta_r = convergence_row.get("delta_r_fractional_peak_to_peak_last_window")
            reasons.append(
                "not converged over final 100 cycles "
                f"(source={source_kind}, Gamma_ptp={gamma}, P_frac_ptp={period}, DeltaR_frac_ptp={delta_r})"
            )
        if not reasons:
            continue
        model = dict(model)
        model["extension_reasons"] = reasons
        result.append(model)
    return result


def extend_one_model(args: argparse.Namespace, model_id: str, cap: int, log_path: Path) -> int:
    commands = [
        [
            str(args.python),
            str(RUNNER),
            "--workspace",
            str(args.workspace),
            "--model",
            model_id,
            "--stage",
            "continue_saturation",
            "--bash",
            args.bash,
            "--force",
            "--resume-from-latest-photo",
            "--resume-max-num-periods",
            str(cap),
        ],
        [str(args.python), str(RUNNER), "--workspace", str(args.workspace), "--model", model_id, "--stage", "restart", "--bash", args.bash, "--force"],
        [str(args.python), str(RUNNER), "--workspace", str(args.workspace), "--model", model_id, "--stage", "deep2cycles", "--bash", args.bash, "--force"],
        [str(args.python), str(RUNNER), "--workspace", str(args.workspace), "--model", model_id, "--stage", "final_cycle", "--force"],
        [str(args.python), str(RUNNER), "--workspace", str(args.workspace), "--model", model_id, "--stage", "plot", "--force"],
        [str(args.python), str(RUNNER), "--workspace", str(args.workspace), "--model", model_id, "--stage", "verify", "--force"],
    ]
    for command in commands:
        code = run_command(command, log_path, args.dry_run)
        if code != 0:
            return code
    return 0


def main() -> int:
    args = parse_args()
    args.workspace = args.workspace.resolve()
    output_dir = args.workspace / "output"
    log_path = (args.log or output_dir / "quality_extension.log").resolve()
    status_path = (args.status or output_dir / "quality_extension_status.json").resolve()

    append_log(log_path, "quality-extension driver started")
    write_json(
        status_path,
        {
            "updated_at": now_iso(),
            "status": "starting",
            "workspace": str(args.workspace),
            "caps": args.caps,
            "models": args.models,
            "log": str(log_path),
            "dry_run": args.dry_run,
        },
    )

    if args.wait_for_batch:
        wait_code = wait_for_batch(args, log_path, status_path)
        if wait_code != 0:
            write_json(status_path, {"updated_at": now_iso(), "status": "postponed", "log": str(log_path)})
            return wait_code

    code = analysis_refresh(args, log_path, allow_incomplete=True)
    if code != 0:
        write_json(status_path, {"updated_at": now_iso(), "status": "analysis_refresh_failed", "returncode": code, "log": str(log_path)})
        return code

    attempts: list[dict[str, object]] = []
    for cap in args.caps:
        targets = quality_warning_models(args, log_path)
        if not targets:
            strict_code = strict_audit(args, log_path)
            status_text = "complete" if strict_code == 0 else "strict_audit_failed"
            write_json(
                status_path,
                {
                    "updated_at": now_iso(),
                    "status": status_text,
                    "strict_audit_returncode": strict_code,
                    "attempts": attempts,
                    "log": str(log_path),
                },
            )
            return strict_code
        append_log(log_path, f"cap={cap}; targets=" + ", ".join(str(item.get("model_id")) for item in targets))
        for target in targets:
            model_id = str(target["model_id"])
            write_json(
                status_path,
                {
                    "updated_at": now_iso(),
                    "status": "extending",
                    "current_model": model_id,
                    "cap": cap,
                    "attempts": attempts,
                    "log": str(log_path),
                },
            )
            returncode = extend_one_model(args, model_id, cap, log_path)
            attempts.append({"model_id": model_id, "cap": cap, "returncode": returncode, "ended_at": now_iso()})
            if returncode != 0:
                write_json(
                    status_path,
                    {
                        "updated_at": now_iso(),
                        "status": "failed",
                        "current_model": model_id,
                        "cap": cap,
                        "returncode": returncode,
                        "attempts": attempts,
                        "log": str(log_path),
                    },
                )
                return returncode
        code = analysis_refresh(args, log_path, allow_incomplete=True)
        if code != 0:
            write_json(status_path, {"updated_at": now_iso(), "status": "analysis_refresh_failed", "returncode": code, "attempts": attempts, "log": str(log_path)})
            return code

    remaining = quality_warning_models(args, log_path)
    strict_code = None
    if not remaining:
        strict_code = strict_audit(args, log_path)
    status_text = "complete" if not remaining and strict_code == 0 else "quality_warnings_remain" if remaining else "strict_audit_failed"
    write_json(
        status_path,
        {
            "updated_at": now_iso(),
            "status": status_text,
            "remaining_models": [item.get("model_id") for item in remaining],
            "strict_audit_returncode": strict_code,
            "attempts": attempts,
            "log": str(log_path),
        },
    )
    if remaining:
        return 1
    return int(strict_code or 0)


if __name__ == "__main__":
    raise SystemExit(main())
