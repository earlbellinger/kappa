from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_WORKSPACE = ROOT / "rsp_batch_runs"
DEFAULT_PYTHON = Path(r"C:\Program Files\Python311\python.exe")
PLOT_SCRIPT = ROOT / "plot_work_logT_phase_cycle_gif.py"
BLACKBODY_TABLE = ROOT / "bbr_color.txt"
CONVERGENCE_SCRIPT = ROOT / "rsp_batch_convergence.py"

STAGE_ORDER = ("create", "continue_saturation", "restart", "deep2cycles", "final_cycle", "plot", "verify")
MESA_STAGES = ("create", "continue_saturation", "restart", "deep2cycles")
DOWNSTREAM_PRODUCT_STAGES = ("restart", "deep2cycles", "final_cycle", "plot", "verify")
RUN_SCRIPTS = {
    "create": "rn_create",
    "continue_saturation": "rn_continue_saturation",
    "restart": "rn_restart",
    "deep2cycles": "rn_deep_two_cycles",
}
RESUMABLE_PHOTO_DIRS = {
    "create": "photos_saturation",
    "continue_saturation": "photos_continue_saturation",
}
RESUME_BASE_INLISTS = {
    "create": "inlist_create",
    "continue_saturation": "inlist_continue_saturation",
}
RESUME_LOG_DIR_PREFIXES = {
    "create": "LOGS_saturation_resume",
    "continue_saturation": "LOGS_continue_saturation_resume",
}
RESUME_PHOTO_DIR_PREFIXES = {
    "create": "photos_saturation_resume",
    "continue_saturation": "photos_continue_saturation_resume",
}
EXPECTED_MODEL_FIELD = {
    "create": "create_model",
    "continue_saturation": "saturated_model",
    "restart": "restart_model",
    "deep2cycles": "deep_model",
}
EXPECTED_ANIMATION_SCALING_VERSION = "model000-visible-window-v8"
REQUIRED_PROFILE_COLUMNS = {
    "rsp_Pvsc",
    "rsp_src_snk",
    "rsp_Lr",
    "rsp_Lc",
    "rsp_Lt",
    "tau",
    "cp",
    "gamma1",
    "ionization_he4",
}
MAX_PHASE_SEAM_FRACTION = 0.025
MAX_PHASE_ADJACENT_FRACTION = 0.025
MODEL_LOCK_POLL_SECONDS = 30
MODEL_LOCK_STALE_SECONDS = 2 * 24 * 60 * 60
OUTPUT_FRESHNESS_TOLERANCE_SECONDS = 2.0


def expected_opacity_display_bounds(
    panel_bottom: float,
    panel_top: float,
    panel_bottom_fraction: float,
    panel_top_fraction: float,
    reference_display_max: object,
) -> tuple[float, float]:
    _ = reference_display_max
    y_min = float(panel_bottom)
    y_max = float(panel_top)
    y_span = y_max - y_min
    return (
        float(y_min + float(panel_bottom_fraction) * y_span),
        float(y_min + float(panel_top_fraction) * y_span),
    )


def parse_rsp_stop_reason(log_path: Path) -> dict[str, object]:
    result: dict[str, object] = {
        "log_path": str(log_path),
        "exists": log_path.exists(),
        "stop_reason": None,
        "termination_code": None,
        "saturated_by_grekm": None,
        "reached_max_periods": None,
    }
    if not log_path.exists():
        return result
    text = log_path.read_text(errors="replace")
    stop_matches = re.findall(r"stop because\s+(.+)", text)
    termination_matches = re.findall(r"termination code:\s+(.+)", text)
    stop_reason = stop_matches[-1].strip() if stop_matches else None
    termination_code = termination_matches[-1].strip() if termination_matches else None
    result["stop_reason"] = stop_reason
    result["termination_code"] = termination_code
    result["saturated_by_grekm"] = bool(stop_reason and "GREKM_avg_abs < RSP_GREKM_avg_abs_limit" in stop_reason)
    result["reached_max_periods"] = bool(
        (stop_reason and "period_number >= max_period_number" in stop_reason)
        or (termination_code and "reached max number of periods" in termination_code)
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or verify a prepared congruent RSP batch model.")
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--model", required=True, help="Model id or run name, e.g. model_001")
    parser.add_argument(
        "--stage",
        choices=(*STAGE_ORDER, "all", "mesa"),
        default="all",
        help="Stage to run. 'mesa' runs create through deep2cycles; 'all' also plots and verifies.",
    )
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable))
    parser.add_argument("--bash", default="bash", help="Bash executable used for MESA run scripts.")
    parser.add_argument("--force", action="store_true", help="Run stage even if its expected output exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands and update nothing.")
    parser.add_argument(
        "--resume-from-latest-photo",
        action="store_true",
        help="For create or continue_saturation, resume from the latest saved photo for that stage.",
    )
    parser.add_argument(
        "--resume-max-num-periods",
        type=int,
        default=None,
        help="When resuming an RSP stage, override RSP_max_num_periods in the generated resume inlist.",
    )
    parser.add_argument(
        "--lock-wait-seconds",
        type=int,
        default=7 * 24 * 60 * 60,
        help="Maximum time to wait for another runner's model lock before failing.",
    )
    parser.add_argument(
        "--allow-unconverged-products",
        action="store_true",
        help="Allow restart/deep-profile/plot products even when strict limit-cycle convergence has not passed.",
    )
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_iso_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            completed = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return True
        return str(pid) in completed.stdout
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def read_model_lock(lock_path: Path) -> dict[str, object]:
    try:
        payload = json.loads(lock_path.read_text())
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def lock_is_stale(lock_path: Path) -> tuple[bool, str]:
    payload = read_model_lock(lock_path)
    pid_raw = payload.get("pid")
    try:
        pid = int(pid_raw)
    except (TypeError, ValueError):
        pid = -1
    if pid > 0 and not process_exists(pid):
        return True, f"pid {pid} is no longer running"

    started = parse_iso_datetime(payload.get("started_at"))
    if started is not None:
        age_seconds = (datetime.now(timezone.utc) - started).total_seconds()
        if age_seconds > MODEL_LOCK_STALE_SECONDS:
            return True, f"lock age {age_seconds:.0f}s exceeds stale threshold"
    return False, "lock owner still appears active"


@contextmanager
def model_run_lock(record: dict[str, object], wait_seconds: int, dry_run: bool):
    if dry_run:
        yield
        return

    output_dir = Path(str(record["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".model_run.lock"
    started_waiting = time.monotonic()
    lock_payload = {
        "pid": os.getpid(),
        "model_id": record.get("model_id"),
        "run_name": record.get("run_name"),
        "started_at": now_iso(),
    }
    lock_text = json.dumps(lock_payload, indent=2, sort_keys=True) + "\n"

    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            stale, reason = lock_is_stale(lock_path)
            if stale:
                print(f"Removing stale model lock {lock_path}: {reason}")
                try:
                    lock_path.unlink()
                    continue
                except OSError:
                    pass

            elapsed = time.monotonic() - started_waiting
            if elapsed > max(0, wait_seconds):
                raise TimeoutError(f"Timed out waiting for model lock {lock_path}: {reason}")
            print(f"Waiting for model lock {lock_path}: {reason}")
            time.sleep(MODEL_LOCK_POLL_SECONDS)
            continue
        else:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(lock_text)
            break

    try:
        yield
    finally:
        current_payload = read_model_lock(lock_path)
        if current_payload.get("pid") == os.getpid():
            try:
                lock_path.unlink()
            except OSError:
                pass


def load_manifest(workspace: Path) -> list[dict[str, object]]:
    manifest = workspace / "inputs" / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest}. Run rsp_batch_prepare.py first.")
    return json.loads(manifest.read_text())


def find_record(records: list[dict[str, object]], model_selector: str) -> dict[str, object]:
    for record in records:
        if model_selector in {str(record["model_id"]), str(record["run_name"])}:
            return record
    raise KeyError(f"No model matching {model_selector!r}")


def load_status(output_dir: Path, record: dict[str, object]) -> dict[str, object]:
    path = output_dir / "run_status.json"
    if path.exists():
        return json.loads(path.read_text())
    return {
        "model_id": record["model_id"],
        "run_name": record["run_name"],
        "registered_existing": bool(record["registered_existing"]),
        "run_dir": record["run_dir"],
        "output_dir": record["output_dir"],
        "prefix": record["prefix"],
        "product_stem": record["product_stem"],
        "stages": {},
    }


def save_status(output_dir: Path, status: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    status["updated_at"] = now_iso()
    (output_dir / "run_status.json").write_text(json.dumps(status, indent=2) + "\n")


def stage_list(stage: str) -> tuple[str, ...]:
    if stage == "all":
        return STAGE_ORDER
    if stage == "mesa":
        return MESA_STAGES
    return (stage,)


def expected_path(record: dict[str, object], stage: str) -> Path:
    run_dir = Path(str(record["run_dir"]))
    output_dir = Path(str(record["output_dir"]))
    if stage in EXPECTED_MODEL_FIELD:
        return run_dir / str(record[EXPECTED_MODEL_FIELD[stage]])
    if stage == "final_cycle":
        return output_dir / f"{record['prefix']}_final_cycle_summary.json"
    if stage == "plot":
        return output_dir / f"{record['product_stem']}.gif"
    if stage == "verify":
        return output_dir / "verification_summary.json"
    raise ValueError(stage)


def convergence_summary_path(workspace: Path) -> Path:
    return workspace / "output" / "convergence_summary_last100.json"


def refresh_convergence(
    workspace: Path,
    python_exe: Path,
    dry_run: bool,
    model_id: object | None = None,
) -> None:
    command = [str(python_exe), str(CONVERGENCE_SCRIPT), "--workspace", str(workspace)]
    if model_id is not None and convergence_summary_path(workspace).exists():
        command.extend(["--models", str(model_id), "--merge-existing"])
    print(" ".join(command))
    if dry_run:
        return
    subprocess.run(command, cwd=str(ROOT), check=True)


def convergence_row(workspace: Path, model_id: object) -> dict[str, object]:
    path = convergence_summary_path(workspace)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    rows = data.get("models", []) if isinstance(data, dict) else []
    for row in rows:
        if isinstance(row, dict) and row.get("model_id") == model_id:
            return row
    return {}


def convergence_gate_reason(row: dict[str, object]) -> str:
    if not row:
        return "strict convergence summary is missing"
    return (
        "strict convergence pending "
        f"(source={row.get('source_kind')}, "
        f"cycles={row.get('cycle_count')}, "
        f"Gamma_ptp={row.get('gamma_peak_to_peak_last_window')}, "
        f"P_frac_ptp={row.get('period_fractional_peak_to_peak_last_window')}, "
        f"DeltaR_frac_ptp={row.get('delta_r_fractional_peak_to_peak_last_window')})"
    )


def mark_downstream_pending_convergence(
    record: dict[str, object],
    status: dict[str, object],
    stages: tuple[str, ...],
    start_index: int,
    reason: str,
    convergence: dict[str, object],
) -> None:
    for pending_stage in stages[start_index:]:
        if pending_stage not in DOWNSTREAM_PRODUCT_STAGES:
            continue
        mark_stage(
            status,
            pending_stage,
            "skipped_pending_convergence",
            expected_output=str(expected_path(record, pending_stage)),
            reason=reason,
            convergence=convergence,
        )


def mark_stage(
    status: dict[str, object],
    stage: str,
    stage_status: str,
    started_at: str | None = None,
    **extra: object,
) -> None:
    stages = status.setdefault("stages", {})
    assert isinstance(stages, dict)
    payload = {
        "status": stage_status,
        "updated_at": now_iso(),
        **extra,
    }
    if started_at is not None:
        payload["started_at"] = started_at
    stages[stage] = payload


def stage_payload(status: dict[str, object], stage: str) -> dict[str, object]:
    stages = status.get("stages", {})
    if not isinstance(stages, dict):
        return {}
    payload = stages.get(stage, {})
    return payload if isinstance(payload, dict) else {}


def output_is_newer_than(path: Path, started_at: datetime | None) -> bool:
    if started_at is None:
        return path.exists()
    if not path.exists():
        return False
    return path.stat().st_mtime >= started_at.timestamp() - OUTPUT_FRESHNESS_TOLERANCE_SECONDS


def existing_output_can_complete_stage(
    expected: Path,
    status: dict[str, object],
    stage: str,
) -> tuple[bool, str]:
    if not expected.exists():
        return False, f"expected output is missing: {expected}"
    payload = stage_payload(status, stage)
    stage_status = payload.get("status")
    started_at = parse_iso_datetime(payload.get("started_at"))
    if stage_status in {"running", "failed"} and started_at is not None:
        if not output_is_newer_than(expected, started_at):
            return (
                False,
                f"existing output predates {stage} attempt started at {started_at.isoformat()}: {expected}",
            )
    return True, f"expected output exists: {expected}"


def run_logged(command: list[str], cwd: Path, log_path: Path, dry_run: bool) -> None:
    print(" ".join(command))
    print(f"cwd: {cwd}")
    print(f"log: {log_path}")
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as handle:
        handle.write(f"\n===== {now_iso()} =====\n".encode())
        handle.write(("COMMAND: " + " ".join(command) + "\n").encode())
        handle.flush()
        proc = subprocess.Popen(command, cwd=str(cwd), stdout=handle, stderr=subprocess.STDOUT)
        return_code = proc.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def run_mesa_stage(
    record: dict[str, object],
    stage: str,
    bash_exe: str,
    status: dict[str, object],
    output_dir: Path,
    force: bool,
    dry_run: bool,
    resume_from_latest_photo: bool = False,
    resume_max_num_periods: int | None = None,
) -> None:
    run_dir = Path(str(record["run_dir"]))
    expected = expected_path(record, stage)
    script = RUN_SCRIPTS[stage]
    resume_photo: Path | None = None
    if expected.exists() and not force:
        output_usable, output_reason = existing_output_can_complete_stage(expected, status, stage)
        if output_usable:
            mark_stage(status, stage, "complete", skipped=True, expected_output=str(expected))
            if not dry_run:
                save_status(output_dir, status)
            print(f"{record['model_id']} {stage}: already complete")
            return
        print(f"{record['model_id']} {stage}: ignoring stale output ({output_reason})")
    if resume_from_latest_photo and stage in RESUMABLE_PHOTO_DIRS:
        try:
            resume_photo = latest_saved_photo(run_dir / RESUMABLE_PHOTO_DIRS[stage])
        except FileNotFoundError as exc:
            print(f"{record['model_id']} {stage}: no saved photo for resume ({exc}); retrying from stage start")
        else:
            if dry_run:
                script = f"{RUN_SCRIPTS[stage]}_resume_{resume_photo.name}"
            else:
                script = write_resume_files(run_dir, stage, resume_photo, resume_max_num_periods)
    if dry_run:
        if resume_photo is not None:
            print(f"would resume from photo: {resume_photo}")
        if resume_max_num_periods is not None:
            print(f"would set RSP_max_num_periods = {resume_max_num_periods}")
        run_logged([bash_exe, script], run_dir, output_dir / "logs" / f"{stage}.log", dry_run)
        return

    started_at = now_iso()
    extra = {"resume_photo": str(resume_photo)} if resume_photo is not None else {}
    if resume_max_num_periods is not None:
        extra["resume_max_num_periods"] = resume_max_num_periods
    mark_stage(status, stage, "running", started_at=started_at, expected_output=str(expected), **extra)
    save_status(output_dir, status)
    try:
        run_logged([bash_exe, script], run_dir, output_dir / "logs" / f"{stage}.log", dry_run)
        if not dry_run:
            started = parse_iso_datetime(started_at)
            if not output_is_newer_than(expected, started):
                raise FileNotFoundError(f"Expected fresh {expected} after {stage}")
    except Exception as exc:
        mark_stage(
            status,
            stage,
            "failed",
            started_at=started_at,
            error=repr(exc),
            expected_output=str(expected),
            **extra,
        )
        save_status(output_dir, status)
        raise
    mark_stage(status, stage, "complete", started_at=started_at, expected_output=str(expected), **extra)
    save_status(output_dir, status)


def latest_saved_photo(photo_dir: Path) -> Path:
    if not photo_dir.exists():
        raise FileNotFoundError(f"Missing photo directory: {photo_dir}")
    numeric_photos = []
    for path in photo_dir.iterdir():
        if path.is_file() and path.name.isdigit():
            numeric_photos.append((int(path.name), path))
    if not numeric_photos:
        raise FileNotFoundError(f"No numeric saved photos in {photo_dir}")
    return max(numeric_photos, key=lambda item: item[0])[1]


def set_or_insert_assignment(text: str, key: str, value: str, after_key: str | None = None) -> str:
    lines = text.splitlines()
    pattern = re_assignment_for_key(key)
    for idx, line in enumerate(lines):
        match = pattern.match(line)
        if match:
            lines[idx] = f"{match.group('indent')}{key} = {value}"
            return "\n".join(lines) + "\n"
    insert_at = 1
    if after_key is not None:
        after_pattern = re_assignment_for_key(after_key)
        for idx, line in enumerate(lines):
            if after_pattern.match(line):
                insert_at = idx + 1
                break
    lines.insert(insert_at, f"      {key} = {value}")
    return "\n".join(lines) + "\n"


def re_assignment_for_key(key: str) -> re.Pattern[str]:
    escaped = re.escape(key)
    return re.compile(rf"^(?P<indent>\s*){escaped}\s*=.*$")


def write_resume_files(
    run_dir: Path,
    stage: str,
    photo_path: Path,
    resume_max_num_periods: int | None = None,
) -> str:
    if stage not in RESUMABLE_PHOTO_DIRS:
        raise ValueError(f"Stage {stage!r} cannot be resumed from a saved photo")
    photo_id = photo_path.name
    inlist_name = f"{RESUME_BASE_INLISTS[stage]}_resume_{photo_id}"
    script_name = f"{RUN_SCRIPTS[stage]}_resume_{photo_id}"
    photo_dir = RESUMABLE_PHOTO_DIRS[stage]
    text = (run_dir / RESUME_BASE_INLISTS[stage]).read_text()
    text = set_or_insert_assignment(text, "load_saved_model", ".false.")
    text = set_or_insert_assignment(text, "load_saved_photo", ".true.", after_key="load_saved_model")
    text = set_or_insert_assignment(
        text,
        "saved_photo_name",
        f"'{photo_dir}/{photo_id}'",
        after_key="load_saved_photo",
    )
    if resume_max_num_periods is not None:
        text = set_or_insert_assignment(text, "RSP_max_num_periods", str(resume_max_num_periods))
        text = set_or_insert_assignment(text, "RSP_GREKM_avg_abs_limit", "-1")
        text = set_or_insert_assignment(text, "RSP_target_steps_per_cycle", "1000")
    text = set_or_insert_assignment(text, "log_directory", f"'{RESUME_LOG_DIR_PREFIXES[stage]}_{photo_id}'")
    text = set_or_insert_assignment(text, "photo_directory", f"'{RESUME_PHOTO_DIR_PREFIXES[stage]}_{photo_id}'")
    (run_dir / inlist_name).write_text(text, newline="\n")
    script = (
        "#!/bin/bash\n\n"
        f"cp -f {inlist_name} inlist || exit 1\n"
        "rm -f restart_photo\n\n"
        "date \"+DATE: %Y-%m-%d%nTIME: %H:%M:%S\"\n"
        "./star\n"
        "status=$?\n"
        "date \"+DATE: %Y-%m-%d%nTIME:%H:%M:%S\"\n"
        f"echo 'finished {stage} resume from photo {photo_id} with status '${{status}}\n"
        "exit ${status}\n"
    )
    script_path = run_dir / script_name
    script_path.write_text(script, newline="\n")
    try:
        script_path.chmod(script_path.stat().st_mode | 0o111)
    except OSError:
        pass
    return script_name


def run_plot_stage(
    record: dict[str, object],
    records: list[dict[str, object]],
    python_exe: Path,
    status: dict[str, object],
    output_dir: Path,
    force: bool,
    dry_run: bool,
) -> None:
    expected = expected_path(record, "plot")
    command = [
        str(python_exe),
        str(PLOT_SCRIPT),
        "--run-dir",
        str(record["run_dir"]),
        "--output-dir",
        str(record["output_dir"]),
        "--prefix",
        str(record["prefix"]),
        "--fps",
        "35",
        "--coordinate",
        "r_over_R",
        "--dark-mode",
        "--main-terms-only",
        "--pressure-work-mode",
        "gas_plus_pav",
        "--heating-mode",
        "gas_minus_c",
    ]
    reference_record = next((item for item in records if item.get("model_id") == "model_000"), None)
    if reference_record is not None:
        reference_summary = Path(str(reference_record["output_dir"])) / f"{reference_record['product_stem']}_summary.json"
        if reference_summary.exists():
            command.extend(["--scaling-reference-summary", str(reference_summary)])
    if BLACKBODY_TABLE.exists():
        command.extend(["--blackbody-color-file", str(BLACKBODY_TABLE)])
    if dry_run:
        run_logged(command, ROOT, output_dir / "logs" / "plot.log", dry_run)
        return
    if expected.exists() and not force:
        current, current_reason = animation_product_is_current(record, output_dir)
        if current:
            mark_stage(
                status,
                "plot",
                "complete",
                skipped=True,
                expected_output=str(expected),
                current_product_reason=current_reason,
            )
            save_status(output_dir, status)
            print(f"{record['model_id']} plot: already complete")
            return
        print(f"{record['model_id']} plot: regenerating stale product ({current_reason})")

    started_at = now_iso()
    mark_stage(status, "plot", "running", started_at=started_at, expected_output=str(expected))
    save_status(output_dir, status)
    try:
        run_logged(command, ROOT, output_dir / "logs" / "plot.log", dry_run)
        if not dry_run and not expected.exists():
            raise FileNotFoundError(f"Expected {expected} after plotting")
    except Exception as exc:
        mark_stage(status, "plot", "failed", started_at=started_at, error=repr(exc), expected_output=str(expected))
        save_status(output_dir, status)
        raise
    mark_stage(status, "plot", "complete", started_at=started_at, expected_output=str(expected))
    save_status(output_dir, status)


def read_history(path: Path) -> dict[str, list[float]]:
    lines = path.read_text(errors="replace").splitlines()
    header_idx = next(i for i, line in enumerate(lines) if line.strip().startswith("model_number"))
    names = lines[header_idx].split()
    rows: list[list[float]] = []
    for line in lines[header_idx + 1 :]:
        parts = line.split()
        if len(parts) != len(names):
            continue
        try:
            rows.append([float(part.replace("D", "E").replace("d", "e")) for part in parts])
        except ValueError:
            continue
    if not rows:
        raise RuntimeError(f"No numeric history rows found in {path}")
    columns: dict[str, list[float]] = {name: [] for name in names}
    for row in rows:
        for name, value in zip(names, row):
            columns[name].append(value)
    return columns


def preferred_history_candidates(run_dir: Path) -> list[Path]:
    resume_histories = sorted(
        run_dir.glob("LOGS_continue_saturation_resume_*/history.data"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    candidates = [
        run_dir / "LOGS" / "history.data",
        *resume_histories,
        run_dir / "LOGS_continue_saturation" / "history.data",
        run_dir / "LOGS_saturation" / "history.data",
    ]
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def local_maxima(values: list[float]) -> list[int]:
    return [
        idx
        for idx in range(1, len(values) - 1)
        if values[idx] > values[idx - 1] and values[idx] >= values[idx + 1]
    ]


def generate_final_cycle_summary(record: dict[str, object], output_dir: Path) -> dict[str, object]:
    run_dir = Path(str(record["run_dir"]))
    history_candidates = preferred_history_candidates(run_dir)
    history_path = next((path for path in history_candidates if path.exists()), None)
    if history_path is None:
        raise RuntimeError("Could not find history.data in deep, continuation, or saturation LOGS directories.")
    history = read_history(history_path)
    required = ("star_age", "log_R", "log_Teff", "log_L", "abs_mag_V", "abs_mag_I")
    missing = [name for name in required if name not in history]
    if missing:
        raise RuntimeError(f"Missing history column(s) in {history_path}: {', '.join(missing)}")

    age_days = [value * 365.25 for value in history["star_age"]]
    radius_rsun = [10.0 ** value for value in history["log_R"]]
    teff_k = [10.0 ** value for value in history["log_Teff"]]
    log_l = history["log_L"]
    mag_v = history["abs_mag_V"]
    mag_i = history["abs_mag_I"]

    maxima = local_maxima(radius_rsun)
    if len(radius_rsun) >= 2 and radius_rsun[-1] > radius_rsun[-2]:
        maxima.append(len(radius_rsun) - 1)
    if len(maxima) < 2:
        raise RuntimeError("Could not identify two radius maxima for the final pulsation cycle.")
    start_idx = maxima[-2]
    end_idx = maxima[-1]
    if end_idx <= start_idx:
        raise RuntimeError("Final-cycle radius maxima are not ordered in time.")

    boundary_end_age_days = age_days[end_idx]
    indices = list(range(start_idx, end_idx))
    if len(indices) < 2:
        raise RuntimeError("Final-cycle radius maxima do not enclose enough samples for a half-open cycle.")
    cycle_age_days = [age_days[idx] for idx in indices]
    cycle_radius_rsun = [radius_rsun[idx] for idx in indices]
    cycle_teff_k = [teff_k[idx] for idx in indices]
    cycle_log_l = [log_l[idx] for idx in indices]
    cycle_mag_v = [mag_v[idx] for idx in indices]
    cycle_mag_i = [mag_i[idx] for idx in indices]

    period_days = boundary_end_age_days - cycle_age_days[0]
    if period_days <= 0.0:
        raise RuntimeError(f"Non-positive final-cycle period from {history_path}: {period_days}")
    max_light_local_idx = min(range(len(cycle_mag_v)), key=lambda idx: cycle_mag_v[idx])
    max_light_age_days = cycle_age_days[max_light_local_idx]
    max_light_phase = (max_light_age_days - cycle_age_days[0]) / period_days

    # Keep the displayed cycle chronological.  Rephasing a radius-bounded cycle
    # to max light moves the cycle boundary into the middle of the plotted
    # lightcurve when a model has not settled into a clean limit cycle.
    phase = [(age - cycle_age_days[0]) / period_days for age in cycle_age_days]
    order = list(range(len(phase)))

    csv_path = output_dir / f"{record['prefix']}_final_cycle_lightcurve.csv"
    json_path = output_dir / f"{record['prefix']}_final_cycle_summary.json"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["phase", "age_days", "abs_mag_V", "abs_mag_I", "radius_rsun", "teff_k", "log_L"])
        for idx in order:
            writer.writerow(
                [
                    f"{phase[idx]:.12g}",
                    f"{cycle_age_days[idx]:.12g}",
                    f"{cycle_mag_v[idx]:.12g}",
                    f"{cycle_mag_i[idx]:.12g}",
                    f"{cycle_radius_rsun[idx]:.12g}",
                    f"{cycle_teff_k[idx]:.12g}",
                    f"{cycle_log_l[idx]:.12g}",
                ]
            )

    summary = {
        "history_file": str(history_path),
        "period_days": float(period_days),
        "max_light_age_days": float(max_light_age_days),
        "max_light_phase": float(max_light_phase),
        "phase_reference_age_days": float(cycle_age_days[0]),
        "phase_reference_kind": "cycle-start radius maximum",
        "cycle_selection": "radius-maximum to radius-maximum, chronological phase",
        "cycle_max_light_local_index": int(max_light_local_idx),
        "radius_min_rsun": float(min(cycle_radius_rsun)),
        "radius_max_rsun": float(max(cycle_radius_rsun)),
        "radius_peak_to_peak_rsun": float(max(cycle_radius_rsun) - min(cycle_radius_rsun)),
        "teff_min_k": float(min(cycle_teff_k)),
        "teff_max_k": float(max(cycle_teff_k)),
        "abs_mag_V_min": float(min(cycle_mag_v)),
        "abs_mag_V_max": float(max(cycle_mag_v)),
        "abs_mag_V_peak_to_peak": float(max(cycle_mag_v) - min(cycle_mag_v)),
        "abs_mag_I_min": float(min(cycle_mag_i)),
        "abs_mag_I_max": float(max(cycle_mag_i)),
        "abs_mag_I_peak_to_peak": float(max(cycle_mag_i) - min(cycle_mag_i)),
        "cycle_start_age_days": float(cycle_age_days[0]),
        "cycle_end_age_days": float(boundary_end_age_days),
        "cycle_last_sample_age_days": float(cycle_age_days[-1]),
        "cycle_interval": "half-open [radius maximum, next radius maximum)",
        "models_in_cycle": int(len(cycle_age_days)),
        "lightcurve_csv": str(csv_path),
    }
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def run_final_cycle_stage(
    record: dict[str, object],
    status: dict[str, object],
    output_dir: Path,
    force: bool,
    dry_run: bool,
) -> None:
    expected = expected_path(record, "final_cycle")
    if dry_run:
        print(f"generate final-cycle summary {record['model_id']} -> {expected}")
        return
    if expected.exists() and not force:
        mark_stage(status, "final_cycle", "complete", skipped=True, expected_output=str(expected))
        save_status(output_dir, status)
        print(f"{record['model_id']} final_cycle: already complete")
        return
    started_at = now_iso()
    mark_stage(status, "final_cycle", "running", started_at=started_at, expected_output=str(expected))
    save_status(output_dir, status)
    try:
        summary = generate_final_cycle_summary(record, output_dir)
    except Exception as exc:
        mark_stage(status, "final_cycle", "failed", started_at=started_at, error=repr(exc), expected_output=str(expected))
        save_status(output_dir, status)
        raise
    mark_stage(status, "final_cycle", "complete", started_at=started_at, expected_output=str(expected), summary=summary)
    save_status(output_dir, status)


def animation_summary_scaling_status(summary: dict[str, object]) -> tuple[bool, str]:
    try:
        scaling_version = str(summary["scaling_method_version"])
        main_radius_xlim = summary["main_radius_xlim_used"]
        scaling_x_limits = summary["left_panel_x_limits_for_scaling"]
        visible_power_bounds = summary["left_power_visible_data_bounds"]
        left_power_ylim = summary["left_power_ylim"]
        opacity_scaling = summary["opacity_scaling"]
        panel_y_ranges = summary["panel_y_ranges"]
        left_power_panel = panel_y_ranges["left_power"]
        scale_left = float(scaling_x_limits[0])
        scale_right = float(scaling_x_limits[1])
        radius_left = float(main_radius_xlim[0])
        radius_right = float(main_radius_xlim[1])
        visible_min = float(visible_power_bounds[0])
        visible_max = float(visible_power_bounds[1])
        panel_bottom = float(left_power_ylim[0])
        panel_top = float(left_power_ylim[1])
        opacity_units = float(opacity_scaling["display_units_per_opacity_unit"])
        opacity_max_display = float(opacity_scaling["opacity_max_display_value"])
        opacity_min_display = float(opacity_scaling["display_min_value"])
        opacity_panel_bottom_fraction = float(opacity_scaling["panel_bottom_fraction"])
        opacity_panel_top_fraction = float(opacity_scaling["panel_top_fraction"])
        reference_opacity_max_display = opacity_scaling.get("reference_opacity_max_display_value")
        opacity_scaled_visible = opacity_scaling["scaled_visible_display_bounds"]
        opacity_x_limits = opacity_scaling["x_limits_for_scaling"]
        opacity_visible = opacity_scaling["visible_data_bounds"]
        left_power_limits = left_power_panel["limits"]
        left_power_visible = left_power_panel["visible_data_bounds"]
    except (KeyError, TypeError, ValueError, IndexError):
        return False, "visible-window scaling metadata is missing"

    if scaling_version != EXPECTED_ANIMATION_SCALING_VERSION:
        return False, (
            f"animation scaling version is {scaling_version!r}, "
            f"expected {EXPECTED_ANIMATION_SCALING_VERSION!r}"
        )
    finite_values = (
        scale_left,
        scale_right,
        radius_left,
        radius_right,
        visible_min,
        visible_max,
        panel_bottom,
        panel_top,
        opacity_units,
        opacity_max_display,
        opacity_min_display,
        opacity_panel_bottom_fraction,
        opacity_panel_top_fraction,
        float(opacity_scaled_visible[0]),
        float(opacity_scaled_visible[1]),
        float(opacity_x_limits[0]),
        float(opacity_x_limits[1]),
        float(opacity_visible[0]),
        float(opacity_visible[1]),
    )
    if not all(math.isfinite(value) for value in finite_values):
        return False, "visible-window scaling metadata contains non-finite values"
    if abs(scale_left - radius_left) > 1.0e-6 or abs(scale_right - radius_right) > 1.0e-6:
        return False, "left-panel scaling limits do not match the displayed radius range"
    if visible_min < panel_bottom - 1.0e-8 or visible_max > panel_top + 1.0e-8:
        return False, "visible power extrema fall outside the plotted y-limits"
    if (
        abs(float(left_power_limits[0]) - panel_bottom) > 1.0e-8
        or abs(float(left_power_limits[1]) - panel_top) > 1.0e-8
        or abs(float(left_power_visible[0]) - visible_min) > 1.0e-8
        or abs(float(left_power_visible[1]) - visible_max) > 1.0e-8
    ):
        return False, "left-power panel y-range metadata does not match the plotted limits"
    if opacity_units <= 0.0 or opacity_max_display <= 0.0:
        return False, "opacity display scale is not positive"
    expected_opacity_min_display, expected_opacity_max_display = expected_opacity_display_bounds(
        panel_bottom,
        panel_top,
        opacity_panel_bottom_fraction,
        opacity_panel_top_fraction,
        reference_opacity_max_display,
    )
    if not math.isclose(
        opacity_min_display,
        expected_opacity_min_display,
        rel_tol=1.0e-8,
        abs_tol=1.0e-8,
    ):
        return False, "opacity scaling is not tied to this panel's lower opacity band"
    if not math.isclose(
        opacity_max_display,
        expected_opacity_max_display,
        rel_tol=1.0e-8,
        abs_tol=1.0e-8,
    ):
        return False, "opacity scaling is not tied to this panel's upper opacity band"
    if not math.isclose(float(opacity_scaled_visible[0]), opacity_min_display, abs_tol=1.0e-12) or not math.isclose(
        float(opacity_scaled_visible[1]), opacity_max_display, rel_tol=1.0e-8, abs_tol=1.0e-8
    ):
        return False, "opacity scaled visible-display bounds do not match the plotted opacity range"
    if opacity_min_display < panel_bottom - 1.0e-8 or opacity_max_display > panel_top + 1.0e-8:
        return False, "opacity display band falls outside the plotted y-limit"
    if (
        opacity_scaling.get("x_field_for_scaling") != summary["left_panel_x_field_for_scaling"]
        or abs(float(opacity_x_limits[0]) - scale_left) > 1.0e-8
        or abs(float(opacity_x_limits[1]) - scale_right) > 1.0e-8
    ):
        return False, "opacity scaling limits do not match the displayed left-panel radius range"
    if float(opacity_visible[1]) <= float(opacity_visible[0]):
        return False, "opacity visible-window extrema are degenerate"
    return True, "per-panel visible-window scaling metadata is current"


def load_animation_summary(record: dict[str, object], output_dir: Path) -> dict[str, object]:
    summary_path = output_dir / f"{record['product_stem']}_summary.json"
    if not summary_path.exists():
        return {}
    try:
        return json.loads(summary_path.read_text())
    except json.JSONDecodeError:
        return {}


def animation_product_is_current(record: dict[str, object], output_dir: Path) -> tuple[bool, str]:
    summary = load_animation_summary(record, output_dir)
    if not summary:
        return False, "animation summary is missing or unreadable"
    return animation_summary_scaling_status(summary)


def profile_header_columns(profile_path: Path) -> set[str]:
    lines = profile_path.read_text(errors="replace").splitlines()
    if len(lines) < 6:
        raise ValueError(f"Profile is too short to parse: {profile_path}")
    return set(lines[5].split())


def phase_seam_summary(lightcurve_path: Path) -> dict[str, object]:
    if not lightcurve_path.exists():
        return {
            "path": str(lightcurve_path),
            "exists": False,
            "ok": False,
            "failures": [f"final-cycle lightcurve missing: {lightcurve_path}"],
        }

    with lightcurve_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 2:
        return {
            "path": str(lightcurve_path),
            "exists": True,
            "ok": False,
            "row_count": len(rows),
            "failures": ["final-cycle lightcurve has fewer than two rows"],
        }

    metrics = {
        "luminosity_lsun": "log_L",
        "radius_rsun": "radius_rsun",
        "teff_k": "teff_k",
    }
    parsed: list[dict[str, float]] = []
    for row in rows:
        try:
            phase = float(row["phase"])
            parsed_row = {"phase": phase}
            if "age_days" in row and row["age_days"] != "":
                parsed_row["age_days"] = float(row["age_days"])
            if "log_L" in row and row["log_L"] != "":
                parsed_row["luminosity_lsun"] = 10.0 ** float(row["log_L"])
            for key, column in metrics.items():
                if key == "luminosity_lsun":
                    continue
                if column in row and row[column] != "":
                    parsed_row[key] = float(row[column])
            parsed.append(parsed_row)
        except (KeyError, TypeError, ValueError):
            continue

    parsed = sorted(parsed, key=lambda item: item["phase"])
    if len(parsed) < 2:
        return {
            "path": str(lightcurve_path),
            "exists": True,
            "ok": False,
            "row_count": len(rows),
            "parsed_row_count": len(parsed),
            "failures": ["could not parse final-cycle phase seam"],
        }

    result: dict[str, object] = {
        "path": str(lightcurve_path),
        "exists": True,
        "row_count": len(rows),
        "parsed_row_count": len(parsed),
        "phase_first": parsed[0]["phase"],
        "phase_last": parsed[-1]["phase"],
        "max_allowed_fraction_of_amplitude": MAX_PHASE_SEAM_FRACTION,
        "max_allowed_adjacent_fraction_of_amplitude": MAX_PHASE_ADJACENT_FRACTION,
        "metrics": {},
        "failures": [],
    }
    failures: list[str] = []
    phase_values = [item["phase"] for item in parsed]
    if phase_values[0] > 1.0e-6:
        failures.append(f"phase starts at {phase_values[0]:.6g}, expected 0")
    if phase_values[-1] < 0.95:
        failures.append(f"phase ends at {phase_values[-1]:.6g}, expected close to 1")
    age_values = [item["age_days"] for item in parsed if "age_days" in item]
    if len(age_values) == len(parsed):
        age_diffs = [age_values[idx + 1] - age_values[idx] for idx in range(len(age_values) - 1)]
        min_age_diff = min(age_diffs) if age_diffs else 0.0
        result["age_monotonic_by_phase"] = bool(min_age_diff >= -1.0e-9)
        result["minimum_age_step_days_by_phase"] = float(min_age_diff)
        if min_age_diff < -1.0e-9:
            failures.append(
                "age is not monotonic in phase order; this indicates an internal phase wrap in the displayed cycle"
            )
    else:
        result["age_monotonic_by_phase"] = None
    metric_results: dict[str, dict[str, float | None]] = {}
    for key in metrics:
        values = [item[key] for item in parsed if key in item]
        if len(values) < 2:
            failures.append(f"could not parse {key} for phase-seam check")
            metric_results[key] = {"seam": None, "amplitude": None, "fraction_of_amplitude": None}
            continue
        seam = abs(values[-1] - values[0])
        amplitude = max(values) - min(values)
        fraction = seam / amplitude if amplitude > 0.0 else 0.0
        adjacent = max((abs(values[idx + 1] - values[idx]) for idx in range(len(values) - 1)), default=0.0)
        adjacent_fraction = adjacent / amplitude if amplitude > 0.0 else 0.0
        metric_results[key] = {
            "seam": seam,
            "amplitude": amplitude,
            "fraction_of_amplitude": fraction,
            "max_adjacent_step": adjacent,
            "max_adjacent_fraction_of_amplitude": adjacent_fraction,
        }
        if fraction > MAX_PHASE_SEAM_FRACTION:
            failures.append(
                f"{key} phase seam is {fraction:.4g} of amplitude, "
                f"expected <= {MAX_PHASE_SEAM_FRACTION:.4g}"
            )
        if adjacent_fraction > MAX_PHASE_ADJACENT_FRACTION:
            failures.append(
                f"{key} adjacent phase step is {adjacent_fraction:.4g} of amplitude, "
                f"expected <= {MAX_PHASE_ADJACENT_FRACTION:.4g}"
            )

    result["metrics"] = metric_results
    result["failures"] = failures
    result["ok"] = not failures
    return result


def verify_model(record: dict[str, object], output_dir: Path) -> dict[str, object]:
    run_dir = Path(str(record["run_dir"]))
    logs_dir = run_dir / "LOGS"
    profiles = sorted(logs_dir.glob("profile*.data"))
    if not profiles:
        raise FileNotFoundError(f"No deep-cycle profile*.data files in {logs_dir}")
    columns = profile_header_columns(profiles[0])
    missing_columns = sorted(REQUIRED_PROFILE_COLUMNS - columns)

    summary_path = output_dir / f"{record['product_stem']}_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    pressure_mode = summary.get("pdv_generation", {}).get("pressure_work_mode")
    heating_mode = summary.get("heating_mode")
    cycle_source = summary.get("cycle_source")
    main_radius_xlim = summary.get("main_radius_xlim_used")
    scaling_x_limits = summary.get("left_panel_x_limits_for_scaling")
    visible_power_bounds = summary.get("left_power_visible_data_bounds")
    left_power_ylim = summary.get("left_power_ylim")
    opacity_scaling = summary.get("opacity_scaling")
    scaling_version = summary.get("scaling_method_version")
    panel_y_ranges = summary.get("panel_y_ranges")
    photosphere_visualization = summary.get("photosphere_visualization", {})
    photosphere_radius_min = photosphere_visualization.get("sphere_radius_min_rsun")
    photosphere_radius_max = photosphere_visualization.get("sphere_radius_max_rsun")
    seam = phase_seam_summary(output_dir / f"{record['prefix']}_final_cycle_lightcurve.csv")
    continue_stop = parse_rsp_stop_reason(output_dir / "logs" / "continue_saturation.log")
    radius_window_contains_photosphere = None
    failures: list[str] = []
    if missing_columns:
        failures.append(f"missing profile columns: {', '.join(missing_columns)}")
    if pressure_mode != "gas_plus_pav":
        failures.append(f"pressure_work_mode is {pressure_mode!r}, expected 'gas_plus_pav'")
    if heating_mode != "gas_minus_c":
        failures.append(f"heating_mode is {heating_mode!r}, expected 'gas_minus_c'")
    if cycle_source != "final-cycle summary age window":
        failures.append(f"cycle_source is {cycle_source!r}, expected 'final-cycle summary age window'")
    if scaling_version != EXPECTED_ANIMATION_SCALING_VERSION:
        failures.append(
            f"scaling_method_version is {scaling_version!r}, expected {EXPECTED_ANIMATION_SCALING_VERSION!r}"
        )
    try:
        scale_left = float(scaling_x_limits[0])
        scale_right = float(scaling_x_limits[1])
        radius_left_for_scaling = float(main_radius_xlim[0])
        radius_right_for_scaling = float(main_radius_xlim[1])
        if (
            abs(scale_left - radius_left_for_scaling) > 1.0e-6
            or abs(scale_right - radius_right_for_scaling) > 1.0e-6
        ):
            failures.append(
                "left-panel scaling limits do not match the displayed radius range: "
                f"scaling=[{scale_left:.6g}, {scale_right:.6g}], "
                f"displayed=[{radius_left_for_scaling:.6g}, {radius_right_for_scaling:.6g}]"
            )
    except (TypeError, ValueError, IndexError):
        failures.append("left-panel visible-window scaling metadata is missing")
    try:
        visible_min = float(visible_power_bounds[0])
        visible_max = float(visible_power_bounds[1])
        panel_bottom = float(left_power_ylim[0])
        panel_top = float(left_power_ylim[1])
        if not all(math.isfinite(value) for value in (visible_min, visible_max, panel_bottom, panel_top)):
            raise ValueError("non-finite power scaling values")
        if visible_min < panel_bottom - 1.0e-8 or visible_max > panel_top + 1.0e-8:
            failures.append(
                "visible power extrema fall outside the plotted y-limits: "
                f"visible=[{visible_min:.6g}, {visible_max:.6g}], "
                f"ylim=[{panel_bottom:.6g}, {panel_top:.6g}]"
            )
        try:
            left_power_panel = panel_y_ranges["left_power"]  # type: ignore[index]
            left_panel_limits = left_power_panel["limits"]
            left_panel_visible = left_power_panel["visible_data_bounds"]
            if (
                abs(float(left_panel_limits[0]) - panel_bottom) > 1.0e-8
                or abs(float(left_panel_limits[1]) - panel_top) > 1.0e-8
                or abs(float(left_panel_visible[0]) - visible_min) > 1.0e-8
                or abs(float(left_panel_visible[1]) - visible_max) > 1.0e-8
            ):
                failures.append("left-power panel y-range metadata does not match the plotted limits")
        except (TypeError, ValueError, KeyError, IndexError):
            failures.append("panel-local y-range metadata is missing")
    except (TypeError, ValueError, IndexError):
        failures.append("visible power extrema metadata is missing")
    try:
        opacity_units = float(opacity_scaling["display_units_per_opacity_unit"])
        opacity_max_display = float(opacity_scaling["opacity_max_display_value"])
        opacity_min_display = float(opacity_scaling["display_min_value"])
        opacity_panel_bottom_fraction = float(opacity_scaling["panel_bottom_fraction"])
        opacity_panel_top_fraction = float(opacity_scaling["panel_top_fraction"])
        reference_opacity_max_display = opacity_scaling.get("reference_opacity_max_display_value")
        opacity_scaled_visible = opacity_scaling["scaled_visible_display_bounds"]
        opacity_x_limits = opacity_scaling["x_limits_for_scaling"]
        opacity_visible = opacity_scaling["visible_data_bounds"]
        if not math.isfinite(opacity_units) or opacity_units <= 0.0:
            failures.append(f"opacity display scale is {opacity_units!r}, expected a positive finite value")
        if not math.isfinite(opacity_max_display) or opacity_max_display <= 0.0:
            failures.append(
                f"opacity_max_display_value is {opacity_max_display!r}, expected a positive finite value"
            )
        expected_opacity_min_display, expected_opacity_max_display = expected_opacity_display_bounds(
            float(left_power_ylim[0]),
            float(left_power_ylim[1]),
            opacity_panel_bottom_fraction,
            opacity_panel_top_fraction,
            reference_opacity_max_display,
        )
        if not math.isclose(
            opacity_min_display,
            expected_opacity_min_display,
            rel_tol=1.0e-8,
            abs_tol=1.0e-8,
        ):
            failures.append(
                "opacity display minimum is not tied to the panel-local opacity band: "
                f"opacity_min_display={opacity_min_display:.6g}, "
                f"expected={expected_opacity_min_display:.6g}"
            )
        if not math.isclose(
            opacity_max_display,
            expected_opacity_max_display,
            rel_tol=1.0e-8,
            abs_tol=1.0e-8,
        ):
            failures.append(
                "opacity display maximum is not tied to the panel-local opacity band: "
                f"opacity_max_display={opacity_max_display:.6g}, "
                f"expected={expected_opacity_max_display:.6g}"
            )
        if not math.isclose(float(opacity_scaled_visible[0]), opacity_min_display, abs_tol=1.0e-12) or not math.isclose(
            float(opacity_scaled_visible[1]), opacity_max_display, rel_tol=1.0e-8, abs_tol=1.0e-8
        ):
            failures.append("opacity scaled visible-display bounds do not match the plotted opacity range")
        if (
            not math.isfinite(opacity_min_display)
            or not math.isfinite(opacity_max_display)
            or opacity_min_display < float(left_power_ylim[0]) - 1.0e-8
            or opacity_max_display > float(left_power_ylim[1]) + 1.0e-8
        ):
            failures.append("opacity display band falls outside the plotted y-limit")
        if opacity_scaling.get("x_field_for_scaling") != summary.get("left_panel_x_field_for_scaling"):
            failures.append("opacity scaling x-field does not match the left-panel scaling x-field")
        if (
            abs(float(opacity_x_limits[0]) - float(scaling_x_limits[0])) > 1.0e-8
            or abs(float(opacity_x_limits[1]) - float(scaling_x_limits[1])) > 1.0e-8
        ):
            failures.append("opacity scaling x-limits do not match the displayed left-panel radius range")
        if float(opacity_visible[1]) <= float(opacity_visible[0]):
            failures.append("opacity visible-window extrema are degenerate")
    except (TypeError, ValueError, KeyError):
        failures.append("visible-window opacity scaling metadata is missing")
    try:
        radius_left = float(main_radius_xlim[0])
        radius_right = float(main_radius_xlim[1])
        photosphere_left = float(photosphere_radius_min)
        photosphere_right = float(photosphere_radius_max)
        radius_window_contains_photosphere = (
            radius_left <= photosphere_left + 1.0e-6
            and radius_right >= photosphere_right - 1.0e-6
        )
        if not radius_window_contains_photosphere:
            failures.append(
                "main radius xlim does not contain photosphere range: "
                f"xlim=[{radius_left:.6g}, {radius_right:.6g}], "
                f"photosphere=[{photosphere_left:.6g}, {photosphere_right:.6g}]"
            )
    except (TypeError, ValueError, IndexError):
        failures.append("could not verify main radius xlim against photosphere range")
    if not seam.get("ok"):
        failures.extend(str(item) for item in seam.get("failures", []))

    result = {
        "model_id": record["model_id"],
        "run_name": record["run_name"],
        "profile_count": len(profiles),
        "first_profile": str(profiles[0]),
        "required_profile_columns_present": not missing_columns,
        "missing_profile_columns": missing_columns,
        "summary_path": str(summary_path),
        "pressure_work_mode": pressure_mode,
        "heating_mode": heating_mode,
        "scaling_method_version": scaling_version,
        "expected_scaling_method_version": EXPECTED_ANIMATION_SCALING_VERSION,
        "panel_y_ranges": panel_y_ranges,
        "cycle_source": cycle_source,
        "main_radius_xlim_used": main_radius_xlim,
        "left_panel_x_limits_for_scaling": scaling_x_limits,
        "left_power_visible_data_bounds": visible_power_bounds,
        "left_power_ylim": left_power_ylim,
        "opacity_scaling": opacity_scaling,
        "photosphere_radius_rsun_range": [photosphere_radius_min, photosphere_radius_max],
        "radius_window_contains_photosphere": radius_window_contains_photosphere,
        "continue_saturation_stop": continue_stop,
        "saturated_by_grekm": continue_stop.get("saturated_by_grekm"),
        "reached_max_periods": continue_stop.get("reached_max_periods"),
        "phase_seam": seam,
        "phase_seam_ok": seam.get("ok") is True,
        "passed": not failures,
        "failures": failures,
        "verified_at": now_iso(),
    }
    (output_dir / "verification_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    if failures:
        raise RuntimeError("; ".join(failures))
    return result


def run_verify_stage(
    record: dict[str, object],
    status: dict[str, object],
    output_dir: Path,
    force: bool,
    dry_run: bool,
) -> None:
    expected = expected_path(record, "verify")
    if expected.exists() and not force:
        current, current_reason = animation_product_is_current(record, output_dir)
        if current:
            mark_stage(
                status,
                "verify",
                "complete",
                skipped=True,
                expected_output=str(expected),
                current_product_reason=current_reason,
            )
            if not dry_run:
                save_status(output_dir, status)
            print(f"{record['model_id']} verify: already complete")
            return
        print(f"{record['model_id']} verify: regenerating stale verification ({current_reason})")
    if dry_run:
        print(f"verify {record['model_id']}")
        return
    started_at = now_iso()
    mark_stage(status, "verify", "running", started_at=started_at, expected_output=str(expected))
    save_status(output_dir, status)
    try:
        result = verify_model(record, output_dir)
    except Exception as exc:
        mark_stage(status, "verify", "failed", started_at=started_at, error=repr(exc), expected_output=str(expected))
        save_status(output_dir, status)
        raise
    mark_stage(status, "verify", "complete", started_at=started_at, expected_output=str(expected), result=result)
    save_status(output_dir, status)


def ensure_runner_available(bash_exe: str) -> None:
    if shutil.which(bash_exe) is None:
        raise FileNotFoundError(
            f"Could not find bash executable {bash_exe!r}. Pass --bash with the shell used to run MESA scripts."
        )


def main() -> None:
    args = parse_args()
    records = load_manifest(args.workspace)
    record = find_record(records, args.model)
    if bool(record["registered_existing"]) and args.stage in {"all", "mesa", *MESA_STAGES}:
        raise SystemExit(
            f"{record['model_id']} is registered from an existing run and is not rerun by default. "
            "Use --stage plot or --stage verify if needed."
        )

    if any(stage in MESA_STAGES for stage in stage_list(args.stage)) and not args.dry_run:
        ensure_runner_available(args.bash)

    with model_run_lock(record, args.lock_wait_seconds, args.dry_run):
        output_dir = Path(str(record["output_dir"]))
        status = load_status(output_dir, record)
        stages_to_run = stage_list(args.stage)
        for stage_index, stage in enumerate(stages_to_run):
            if (
                stage in DOWNSTREAM_PRODUCT_STAGES
                and not args.allow_unconverged_products
                and not bool(record["registered_existing"])
            ):
                refresh_convergence(args.workspace, args.python, args.dry_run, record["model_id"])
                row = convergence_row(args.workspace, record["model_id"])
                if row.get("converged_exact") is not True:
                    reason = convergence_gate_reason(row)
                    mark_downstream_pending_convergence(
                        record,
                        status,
                        stages_to_run,
                        stage_index,
                        reason,
                        row,
                    )
                    if not args.dry_run:
                        save_status(output_dir, status)
                    print(f"{record['model_id']} products skipped: {reason}")
                    break
            if stage in MESA_STAGES:
                run_mesa_stage(
                    record,
                    stage,
                    args.bash,
                    status,
                    output_dir,
                    args.force,
                    args.dry_run,
                    resume_from_latest_photo=args.resume_from_latest_photo,
                    resume_max_num_periods=args.resume_max_num_periods,
                )
            elif stage == "final_cycle":
                run_final_cycle_stage(record, status, output_dir, args.force, args.dry_run)
            elif stage == "plot":
                run_plot_stage(record, records, args.python, status, output_dir, args.force, args.dry_run)
            elif stage == "verify":
                run_verify_stage(record, status, output_dir, args.force, args.dry_run)
            else:
                raise ValueError(stage)


if __name__ == "__main__":
    main()
