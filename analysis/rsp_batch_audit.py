from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_WORKSPACE = ROOT / "rsp_batch_runs"
STAGE_ORDER = ("create", "continue_saturation", "restart", "deep2cycles", "final_cycle", "plot", "verify")
EXPECTED_PRESSURE_WORK_MODE = "gas_plus_pav"
EXPECTED_HEATING_MODE = "gas_minus_c"
EXPECTED_CYCLE_SOURCE = "final-cycle summary age window"
MAX_PHASE_SEAM_FRACTION = 0.025
STRONG_MAX_L_MODULATION_FRACTION = 0.05
STRONG_MIN_V_MODULATION_MAG = 0.05
STRONG_PERIOD_MODULATION_FRACTION = 0.05
CONVERGENCE_TOLERANCE = 1.0e-3
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


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the congruent RSP batch workspace.")
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Exit successfully even when some models are still running or pending.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict | list:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return {"_json_error": repr(exc)}


def read_active_batch_status_path(output_dir: Path) -> Path | None:
    candidates: list[tuple[float, Path, dict]] = []
    for path in output_dir.glob("batch*_status.json"):
        data = read_json(path)
        if isinstance(data, dict):
            candidates.append((path.stat().st_mtime, path, data))
    if not candidates:
        return None
    running = [item for item in candidates if item[2].get("status") == "running"]
    selected = max(running, key=lambda item: item[0]) if running else max(candidates, key=lambda item: item[0])
    return selected[1]


def file_check(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() and path.is_file() else None,
    }


def cycle_modulation_by_model(output_dir: Path) -> dict[str, dict]:
    path = output_dir / "cycle_modulation_summary.json"
    data = read_json(path)
    rows = data.get("models", []) if isinstance(data, dict) else []
    return {
        str(row.get("model_id")): row
        for row in rows
        if isinstance(row, dict) and row.get("model_id")
    }


def convergence_by_model(output_dir: Path) -> dict[str, dict]:
    path = output_dir / "convergence_summary_last100.json"
    data = read_json(path)
    rows = data.get("models", []) if isinstance(data, dict) else []
    return {
        str(row.get("model_id")): row
        for row in rows
        if isinstance(row, dict) and row.get("model_id")
    }


def as_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def cycle_modulation_warnings(modulation: dict | None) -> list[str]:
    if not modulation:
        return []
    warnings: list[str] = []
    max_l = as_float(modulation.get("max_l_modulation_fraction"))
    min_v = as_float(modulation.get("min_v_modulation_mag"))
    period = as_float(modulation.get("period_modulation_fraction"))
    if max_l is not None and max_l > STRONG_MAX_L_MODULATION_FRACTION:
        warnings.append(
            f"strong max-L cycle modulation {max_l:.4g} "
            f"> {STRONG_MAX_L_MODULATION_FRACTION:.4g}"
        )
    if min_v is not None and min_v > STRONG_MIN_V_MODULATION_MAG:
        warnings.append(
            f"strong min-V cycle modulation {min_v:.4g} mag "
            f"> {STRONG_MIN_V_MODULATION_MAG:.4g} mag"
        )
    if period is not None and period > STRONG_PERIOD_MODULATION_FRACTION:
        warnings.append(
            f"strong period modulation {period:.4g} "
            f"> {STRONG_PERIOD_MODULATION_FRACTION:.4g}"
        )
    return warnings


def add_quality_warnings(model: dict[str, object], warnings: list[str]) -> None:
    existing = list(model.get("quality_warnings") or [])
    existing.extend(warnings)
    model["quality_warnings"] = existing
    if warnings and model.get("status") == "complete":
        model["status"] = "quality_warning"


def attach_cycle_modulation_quality(model: dict[str, object], modulation_by_model: dict[str, dict]) -> None:
    modulation = modulation_by_model.get(str(model.get("model_id")))
    if modulation:
        model["cycle_modulation"] = modulation
    warnings = cycle_modulation_warnings(modulation)
    add_quality_warnings(model, warnings)


def convergence_warnings(convergence: dict | None) -> list[str]:
    if not convergence:
        return ["convergence summary missing"]
    if convergence.get("converged_exact") is True:
        return []
    warnings: list[str] = []
    source_kind = convergence.get("source_kind")
    if source_kind != "history_exact_rsp_columns":
        warnings.append(f"exact Gamma/P/DeltaR convergence unavailable from source={source_kind!r}")
    used = as_float(convergence.get("last_cycle_count_used"))
    required = as_float(convergence.get("required_last_cycles"))
    if required is not None and (used is None or used < required):
        warnings.append(f"only {used or 0:g}/{required:g} cycles available for convergence window")
    gamma_ptp = as_float(convergence.get("gamma_peak_to_peak_last_window"))
    period_ptp = as_float(convergence.get("period_fractional_peak_to_peak_last_window"))
    delta_r_ptp = as_float(convergence.get("delta_r_fractional_peak_to_peak_last_window"))
    if convergence.get("converged_gamma") is not True:
        warnings.append(f"Gamma peak-to-peak {gamma_ptp!r} > {CONVERGENCE_TOLERANCE:g}")
    if convergence.get("converged_period") is not True:
        warnings.append(f"P fractional peak-to-peak {period_ptp!r} > {CONVERGENCE_TOLERANCE:g}")
    if convergence.get("converged_delta_r") is not True:
        warnings.append(f"DeltaR fractional peak-to-peak {delta_r_ptp!r} > {CONVERGENCE_TOLERANCE:g}")
    return warnings


def attach_convergence_quality(model: dict[str, object], convergence_by_id: dict[str, dict]) -> None:
    convergence = convergence_by_id.get(str(model.get("model_id")))
    if convergence:
        model["convergence"] = convergence
    if model.get("registered_existing"):
        return
    add_quality_warnings(model, convergence_warnings(convergence))


def stage_statuses(run_status: dict) -> dict[str, str | None]:
    stages = run_status.get("stages", {}) if isinstance(run_status, dict) else {}
    if not isinstance(stages, dict):
        return {stage: None for stage in STAGE_ORDER}
    result: dict[str, str | None] = {}
    for stage in STAGE_ORDER:
        payload = stages.get(stage, {})
        result[stage] = payload.get("status") if isinstance(payload, dict) else None
    return result


def expected_files(record: dict) -> dict[str, Path]:
    run_dir = Path(str(record["run_dir"]))
    output_dir = Path(str(record["output_dir"]))
    prefix = str(record["prefix"])
    product_stem = str(record["product_stem"])
    return {
        "create_model": run_dir / str(record.get("create_model", "")),
        "saturated_model": run_dir / str(record.get("saturated_model", "")),
        "restart_model": run_dir / str(record.get("restart_model", "")),
        "deep_model": run_dir / str(record.get("deep_model", "")),
        "final_cycle_summary": output_dir / f"{prefix}_final_cycle_summary.json",
        "final_cycle_lightcurve": output_dir / f"{prefix}_final_cycle_lightcurve.csv",
        "gif": output_dir / f"{product_stem}.gif",
        "png": output_dir / f"{product_stem}.png",
        "animation_summary": output_dir / f"{product_stem}_summary.json",
        "verification_summary": output_dir / "verification_summary.json",
    }


def audit_registered_model(record: dict) -> dict[str, object]:
    output_dir = Path(str(record["output_dir"]))
    product_stem = str(record["product_stem"])
    summary_path = output_dir / f"{product_stem}_summary.json"
    verification_path = output_dir / "verification_summary.json"
    verification = read_json(verification_path)
    animation_summary = read_json(summary_path)
    verification_passed = isinstance(verification, dict) and verification.get("passed") is True
    pressure_mode = verification.get("pressure_work_mode") if isinstance(verification, dict) else None
    heating_mode = verification.get("heating_mode") if isinstance(verification, dict) else None
    cycle_source = verification.get("cycle_source") if isinstance(verification, dict) else None
    radius_ok = verification.get("radius_window_contains_photosphere") if isinstance(verification, dict) else None
    phase_seam_ok = verification.get("phase_seam_ok") if isinstance(verification, dict) else None
    phase_seam = verification.get("phase_seam") if isinstance(verification, dict) else None
    continue_stop = verification.get("continue_saturation_stop") if isinstance(verification, dict) else None
    if cycle_source is None and isinstance(animation_summary, dict):
        cycle_source = animation_summary.get("cycle_source")
    checks = {
        "gif": file_check(output_dir / f"{product_stem}.gif"),
        "png": file_check(output_dir / f"{product_stem}.png"),
        "animation_summary": file_check(summary_path),
        "verification_summary": file_check(verification_path),
    }
    failures = [f"{name} missing" for name, check in checks.items() if not check["exists"]]
    if not verification_passed:
        failures.append("verification_summary did not pass")
    if pressure_mode != EXPECTED_PRESSURE_WORK_MODE:
        failures.append(f"pressure_work_mode={pressure_mode!r}")
    if heating_mode != EXPECTED_HEATING_MODE:
        failures.append(f"heating_mode={heating_mode!r}")
    if cycle_source != EXPECTED_CYCLE_SOURCE:
        failures.append(f"cycle_source={cycle_source!r}")
    if radius_ok is not True:
        failures.append(f"radius_window_contains_photosphere={radius_ok!r}")
    if phase_seam_ok is not True:
        failures.append(f"phase_seam_ok={phase_seam_ok!r}")
    return {
        "model_id": record["model_id"],
        "run_name": record["run_name"],
        "registered_existing": True,
        "status": "complete" if not failures else "failed",
        "checks": checks,
        "verification_passed": verification_passed,
        "pressure_work_mode": pressure_mode,
        "heating_mode": heating_mode,
        "cycle_source": cycle_source,
        "radius_window_contains_photosphere": radius_ok,
        "phase_seam_ok": phase_seam_ok,
        "phase_seam": phase_seam,
        "continue_saturation_stop": continue_stop,
        "saturated_by_grekm": verification.get("saturated_by_grekm") if isinstance(verification, dict) else None,
        "reached_max_periods": verification.get("reached_max_periods") if isinstance(verification, dict) else None,
        "failures": failures,
    }


def audit_batch_model(record: dict) -> dict[str, object]:
    output_dir = Path(str(record["output_dir"]))
    run_status_path = output_dir / "run_status.json"
    run_status = read_json(run_status_path)
    validation = run_status.get("validation", {}) if isinstance(run_status, dict) else {}
    validation_passed = validation.get("passed") is True
    stages = stage_statuses(run_status if isinstance(run_status, dict) else {})
    pending_convergence = any(value == "skipped_pending_convergence" for value in stages.values())
    downstream_stages = {"restart", "deep2cycles", "final_cycle", "plot", "verify"}
    missing_or_bad_stages = [
        stage
        for stage, value in stages.items()
        if value != "complete"
        and not (pending_convergence and stage in downstream_stages and value == "skipped_pending_convergence")
    ]

    files = expected_files(record)
    file_checks = {name: file_check(path) for name, path in files.items()}
    if pending_convergence:
        required_file_names = ("create_model", "saturated_model")
    else:
        required_file_names = (
            "create_model",
            "saturated_model",
            "restart_model",
            "deep_model",
            "final_cycle_summary",
            "final_cycle_lightcurve",
            "gif",
            "png",
            "animation_summary",
            "verification_summary",
        )
    missing_files = [name for name in required_file_names if not file_checks[name]["exists"]]

    verification = read_json(files["verification_summary"])
    verification_passed = isinstance(verification, dict) and verification.get("passed") is True
    pressure_mode = verification.get("pressure_work_mode") if isinstance(verification, dict) else None
    heating_mode = verification.get("heating_mode") if isinstance(verification, dict) else None
    cycle_source = verification.get("cycle_source") if isinstance(verification, dict) else None
    radius_ok = verification.get("radius_window_contains_photosphere") if isinstance(verification, dict) else None
    phase_seam_ok = verification.get("phase_seam_ok") if isinstance(verification, dict) else None
    phase_seam = verification.get("phase_seam") if isinstance(verification, dict) else None
    continue_stop = verification.get("continue_saturation_stop") if isinstance(verification, dict) else None
    missing_columns = verification.get("missing_profile_columns") if isinstance(verification, dict) else None
    if missing_columns is None:
        missing_columns = sorted(REQUIRED_PROFILE_COLUMNS)

    failures: list[str] = []
    if not validation_passed:
        failures.append("validation did not pass")
    if missing_or_bad_stages:
        failures.append("stages not complete: " + ", ".join(missing_or_bad_stages))
    if missing_files:
        failures.append("missing files: " + ", ".join(missing_files))
    if not verification_passed and not pending_convergence:
        failures.append("verification_summary did not pass")
    if pressure_mode != EXPECTED_PRESSURE_WORK_MODE and not pending_convergence:
        failures.append(f"pressure_work_mode={pressure_mode!r}")
    if heating_mode != EXPECTED_HEATING_MODE and not pending_convergence:
        failures.append(f"heating_mode={heating_mode!r}")
    if cycle_source != EXPECTED_CYCLE_SOURCE and not pending_convergence:
        failures.append(f"cycle_source={cycle_source!r}")
    if radius_ok is not True and not pending_convergence:
        failures.append(f"radius_window_contains_photosphere={radius_ok!r}")
    if phase_seam_ok is not True and not pending_convergence:
        failures.append(f"phase_seam_ok={phase_seam_ok!r}")
    if missing_columns and not pending_convergence:
        failures.append("missing profile columns: " + ", ".join(map(str, missing_columns)))

    is_running = any(value == "running" for value in stages.values())
    if failures:
        status = "running" if is_running else "incomplete"
    elif pending_convergence:
        status = "awaiting_convergence"
    else:
        status = "complete"

    return {
        "model_id": record["model_id"],
        "run_name": record["run_name"],
        "registered_existing": False,
        "status": status,
        "validation_passed": validation_passed,
        "stages": stages,
        "files": file_checks,
        "verification_passed": verification_passed,
        "pressure_work_mode": pressure_mode,
        "heating_mode": heating_mode,
        "cycle_source": cycle_source,
        "radius_window_contains_photosphere": radius_ok,
        "phase_seam_ok": phase_seam_ok,
        "phase_seam": phase_seam,
        "pending_convergence": pending_convergence,
        "continue_saturation_stop": continue_stop,
        "saturated_by_grekm": verification.get("saturated_by_grekm") if isinstance(verification, dict) else None,
        "reached_max_periods": verification.get("reached_max_periods") if isinstance(verification, dict) else None,
        "missing_profile_columns": missing_columns,
        "profile_count": verification.get("profile_count") if isinstance(verification, dict) else None,
        "failures": failures,
    }


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    output_dir = workspace / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = (args.output or output_dir / "batch_audit_summary.json").resolve()

    manifest_path = workspace / "inputs" / "manifest.json"
    manifest = read_json(manifest_path)
    if not isinstance(manifest, list):
        summary = {
            "generated_at": now_iso(),
            "workspace": str(workspace),
            "manifest": str(manifest_path),
            "status": "failed",
            "failures": ["manifest missing or invalid"],
            "models": [],
        }
        audit_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(audit_path)
        return 1

    models = [
        audit_registered_model(record)
        if bool(record.get("registered_existing"))
        else audit_batch_model(record)
        for record in manifest
    ]
    modulation_by_model = cycle_modulation_by_model(output_dir)
    convergence_by_id = convergence_by_model(output_dir)
    for model in models:
        attach_cycle_modulation_quality(model, modulation_by_model)
        attach_convergence_quality(model, convergence_by_id)
    complete = all(model["status"] == "complete" for model in models)
    running = any(model["status"] == "running" for model in models)
    incomplete = [model["model_id"] for model in models if model["status"] != "complete"]
    failures = [
        f"{model['model_id']}: {'; '.join(model.get('failures', []))}"
        for model in models
        if model.get("failures")
    ]
    quality_warnings = [
        f"{model['model_id']}: {'; '.join(model.get('quality_warnings', []))}"
        for model in models
        if model.get("quality_warnings")
    ]
    live_status_path = output_dir / "live_status.json"
    batch_status_path = read_active_batch_status_path(output_dir) or output_dir / "batch_remaining_006_009_status.json"
    quality_extension_status_path = output_dir / "quality_extension_status.json"
    quality_extension_status = read_json(quality_extension_status_path)
    quality_extension_required = bool(quality_warnings) or quality_extension_status_path.exists()
    quality_extension_complete = (
        not quality_extension_required
        or (
            isinstance(quality_extension_status, dict)
            and quality_extension_status.get("status") == "complete"
            and int(quality_extension_status.get("strict_audit_returncode", 0) or 0) == 0
        )
    )
    if quality_extension_required and not quality_extension_complete and not args.allow_incomplete:
        status_text = quality_extension_status.get("status") if isinstance(quality_extension_status, dict) else None
        failures.append(f"quality extension status is {status_text!r}, expected 'complete'")
    complete = complete and quality_extension_complete
    summary = {
        "generated_at": now_iso(),
        "workspace": str(workspace),
        "manifest": str(manifest_path),
        "status": "complete" if complete else "running" if running else "incomplete",
        "complete": complete,
        "model_count": len(models),
        "complete_model_count": sum(1 for model in models if model["status"] == "complete"),
        "incomplete_models": incomplete,
        "failures": failures,
        "quality_warnings": quality_warnings,
        "quality_extension": quality_extension_status if isinstance(quality_extension_status, dict) else {},
        "quality_extension_required": quality_extension_required,
        "quality_extension_complete": quality_extension_complete,
        "live_status": str(live_status_path) if live_status_path.exists() else None,
        "batch_status": str(batch_status_path) if batch_status_path.exists() else None,
        "models": models,
    }
    audit_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(audit_path)
    if complete or args.allow_incomplete:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
