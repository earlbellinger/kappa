from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_WORKSPACE = ROOT / "rsp_batch_runs"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a compact live status summary for the RSP batch workspace.")
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def read_json(path: Path) -> dict | list:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def read_active_batch_status(output_dir: Path) -> dict | list:
    candidates: list[tuple[float, Path, dict | list]] = []
    for path in output_dir.glob("batch*_status.json"):
        data = read_json(path)
        if isinstance(data, dict):
            candidates.append((path.stat().st_mtime, path, data))
    if not candidates:
        return {}
    running = [item for item in candidates if isinstance(item[2], dict) and item[2].get("status") == "running"]
    if running:
        return max(running, key=lambda item: item[0])[2]
    return max(candidates, key=lambda item: item[0])[2]


def parse_iso_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def parse_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(float(str(value).replace("d", "e").replace("D", "e")))
    except ValueError:
        return None


def latest_numeric_file(directory: Path) -> str | None:
    if not directory.exists() or not directory.is_dir():
        return None
    numeric = [path for path in directory.iterdir() if path.is_file() and path.name.isdigit()]
    if not numeric:
        return None
    return max(numeric, key=lambda path: int(path.name)).name


def latest_photo(run_dir: Path) -> str | None:
    if not run_dir.exists():
        return None
    candidates: list[tuple[float, Path]] = []
    for photo_dir in run_dir.glob("photos*"):
        photo = latest_numeric_file(photo_dir)
        if photo is not None:
            candidates.append((photo_dir.stat().st_mtime, photo_dir / photo))
    if not candidates:
        return None
    return str(max(candidates, key=lambda item: item[0])[1])


def latest_history_model(run_dir: Path) -> tuple[str | None, str | None]:
    history_files = sorted(run_dir.glob("LOGS*/history.data"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not history_files:
        return None, None
    history = history_files[0]
    try:
        lines = history.read_text(errors="ignore").splitlines()
    except OSError:
        return None, datetime.fromtimestamp(history.stat().st_mtime, timezone.utc).isoformat()
    for line in reversed(lines):
        fields = line.split()
        if fields and fields[0].lstrip("+-").isdigit():
            return fields[0], datetime.fromtimestamp(history.stat().st_mtime, timezone.utc).isoformat()
    return None, datetime.fromtimestamp(history.stat().st_mtime, timezone.utc).isoformat()


def latest_period(output_dir: Path) -> tuple[str | None, str | None]:
    logs_dir = output_dir / "logs"
    if not logs_dir.exists():
        return None, None
    for log_file in sorted(logs_dir.glob("*.log"), key=lambda path: path.stat().st_mtime, reverse=True):
        try:
            lines = log_file.read_text(errors="ignore").splitlines()
        except OSError:
            continue
        for line in reversed(lines):
            match = re.search(r"^\s*period\s+(\d+)\b", line)
            if match:
                return match.group(1), log_file.name
    return None, None


def stage_inlist_candidates(stage: str | None, period_log: str | None) -> tuple[str, ...]:
    stage_from_log = None
    if period_log:
        stage_from_log = period_log.removesuffix(".log")
    selected_stage = stage or stage_from_log
    preferred = {
        "create": ("inlist_create",),
        "continue_saturation": ("inlist_continue_saturation",),
        "restart": ("inlist_restart",),
        "deep2cycles": ("inlist_deep2cycles",),
    }.get(str(selected_stage), ())
    fallback = ("inlist_create", "inlist_continue_saturation", "inlist_restart", "inlist_deep2cycles")
    return tuple(dict.fromkeys((*preferred, *fallback)))


def max_periods(run_dir: Path, stage: str | None = None, period_log: str | None = None) -> str | None:
    if stage is None and period_log is None:
        return None
    for inlist_name in stage_inlist_candidates(stage, period_log):
        inlist = run_dir / inlist_name
        if not inlist.exists():
            continue
        try:
            text = inlist.read_text(errors="ignore")
        except OSError:
            continue
        match = re.search(r"^\s*RSP_max_num_periods\s*=\s*([^\s!]+)", text, flags=re.MULTILINE)
        if match:
            return match.group(1).strip()
    return None


def read_run_status(output_dir: Path) -> dict:
    status = read_json(output_dir / "run_status.json")
    if not isinstance(status, dict):
        return {}
    return status


def convergence_by_model(output_dir: Path) -> dict[str, dict[str, object]]:
    convergence = read_json(output_dir / "convergence_summary_last100.json")
    rows = convergence.get("models", []) if isinstance(convergence, dict) else []
    return {
        str(row.get("model_id")): row
        for row in rows
        if isinstance(row, dict) and row.get("model_id")
    }


def stage_summary(status: dict) -> dict[str, str]:
    stages = status.get("stages", {})
    if not isinstance(stages, dict):
        return {}
    return {name: str(data.get("status")) for name, data in stages.items() if isinstance(data, dict)}


def active_stage(status: dict) -> tuple[str | None, str | None]:
    stages = status.get("stages", {})
    if not isinstance(stages, dict):
        return None, None
    for stage in ("create", "continue_saturation", "restart", "deep2cycles", "final_cycle", "plot", "verify"):
        data = stages.get(stage)
        if isinstance(data, dict) and data.get("status") == "running":
            return stage, data.get("started_at") or data.get("updated_at")
    return None, None


def progress_estimate(period: str | None, max_period: str | None, started_at: str | None) -> dict:
    period_num = parse_int(period)
    max_period_num = parse_int(max_period)
    if period_num is None or max_period_num is None or period_num <= 0 or max_period_num <= 0:
        return {}
    percent = min(100.0, 100.0 * period_num / max_period_num)
    estimate = {
        "period_progress_fraction": period_num / max_period_num,
        "period_progress_percent": percent,
    }
    start_time = parse_iso_datetime(started_at)
    if start_time is not None:
        now = datetime.now(timezone.utc)
        elapsed = max(0.0, (now - start_time).total_seconds())
        if elapsed > 0:
            estimated_total = elapsed * max_period_num / period_num
            remaining = max(0.0, estimated_total - elapsed)
            estimate.update(
                {
                    "estimated_stage_seconds_elapsed": elapsed,
                    "estimated_stage_seconds_total": estimated_total,
                    "estimated_stage_seconds_remaining": remaining,
                    "estimated_stage_eta": (now + timedelta(seconds=remaining)).isoformat(),
                    "estimate_basis": "linear extrapolation from active stage start time and latest period count",
                }
            )
    return estimate


def model_summary(record: dict, convergence_by_id: dict[str, dict[str, object]] | None = None) -> dict:
    run_dir = Path(str(record["run_dir"]))
    output_dir = Path(str(record["output_dir"]))
    product_stem = str(record["product_stem"])
    gif = output_dir / f"{product_stem}.gif"
    verification = read_json(output_dir / "verification_summary.json")
    run_status = read_run_status(output_dir)
    hist_model, hist_mtime = latest_history_model(run_dir)
    period, period_log = latest_period(output_dir)
    running_stage, running_stage_started_at = active_stage(run_status)
    max_period = max_periods(run_dir, running_stage, period_log)
    convergence = (convergence_by_id or {}).get(str(record["model_id"]), {})
    summary = {
        "model_id": record["model_id"],
        "registered_existing": bool(record.get("registered_existing")),
        "run_name": record["run_name"],
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "stages": stage_summary(run_status),
        "active_stage": running_stage,
        "active_stage_started_at": running_stage_started_at,
        "latest_period": period,
        "max_periods": max_period,
        "latest_period_log": period_log,
        "latest_history_model": hist_model,
        "latest_history_mtime": hist_mtime,
        "latest_photo": latest_photo(run_dir),
        "gif_exists": gif.exists(),
        "gif_path": str(gif) if gif.exists() else None,
        "verification_passed": isinstance(verification, dict) and verification.get("passed") is True,
        "profile_count": verification.get("profile_count") if isinstance(verification, dict) else None,
    }
    if convergence:
        summary.update(
            {
                "convergence_source_kind": convergence.get("source_kind"),
                "convergence_cycle_count": convergence.get("cycle_count"),
                "convergence_last_period_number": convergence.get("last_period_number"),
                "convergence_window_start_period_number": convergence.get("window_start_period_number"),
                "convergence_window_end_period_number": convergence.get("window_end_period_number"),
                "gamma_peak_to_peak_last_window": convergence.get("gamma_peak_to_peak_last_window"),
                "period_fractional_peak_to_peak_last_window": convergence.get("period_fractional_peak_to_peak_last_window"),
                "delta_r_fractional_peak_to_peak_last_window": convergence.get("delta_r_fractional_peak_to_peak_last_window"),
                "delta_r_first_last_window": convergence.get("delta_r_first_last_window"),
                "delta_r_slope_per_cycle_last_window": convergence.get("delta_r_slope_per_cycle_last_window"),
                "steps_median_last_window": convergence.get("steps_median_last_window"),
                "converged_gamma": convergence.get("converged_gamma"),
                "converged_period": convergence.get("converged_period"),
                "converged_delta_r": convergence.get("converged_delta_r"),
                "converged_exact": convergence.get("converged_exact"),
                "limit_cycle_converged": convergence.get("limit_cycle_converged", convergence.get("converged_exact")),
            }
        )
    summary.update(progress_estimate(period, max_period, running_stage_started_at))
    return summary


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    output = (args.output or workspace / "output" / "live_status.json").resolve()
    manifest = read_json(workspace / "inputs" / "manifest.json")
    if not isinstance(manifest, list):
        raise RuntimeError(f"Missing manifest under {workspace}")
    workspace_output = workspace / "output"
    convergence = convergence_by_model(workspace_output)
    models = [model_summary(row, convergence) for row in manifest]
    registered_existing_gif_count = sum(1 for row in models if row["registered_existing"] and row["gif_exists"])
    new_batch_gif_count = sum(1 for row in models if not row["registered_existing"] and row["gif_exists"])
    new_batch_model_count = sum(1 for row in models if not row["registered_existing"])
    summary = {
        "generated_at": now_iso(),
        "workspace": str(workspace),
        "batch_status": read_active_batch_status(workspace_output),
        "total_gif_count": sum(1 for row in models if row["gif_exists"]),
        "completed_gif_count": sum(1 for row in models if row["gif_exists"]),
        "registered_existing_gif_count": registered_existing_gif_count,
        "new_batch_gif_count": new_batch_gif_count,
        "new_batch_model_count": new_batch_model_count,
        "verified_model_count": sum(1 for row in models if row["verification_passed"]),
        "limit_cycle_converged_model_count": sum(1 for row in models if row.get("limit_cycle_converged") is True),
        "models": models,
    }
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
