from __future__ import annotations

import argparse
import csv
import html
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


ANIMATION_SUFFIX = "_work_r_over_R_phase_cycle_dark_main_terms_gas_heating_pav_work"
PARAMETER_GROUPS = {
    "Stellar": (
        ("RSP_mass", "M", "M_sun", 4),
        ("RSP_Teff", "Teff", "K", 5),
        ("RSP_L", "L", "L_sun", 4),
        ("RSP_X", "X", "", 4),
        ("RSP_Z", "Z", "", 4),
    ),
    "Convection": (
        ("RSP_alfa", "alpha_MLT", "", 4),
        ("RSP_alfac", "alpha_c", "", 4),
        ("RSP_alfas", "alpha_s", "", 4),
        ("RSP_alfad", "alpha_d", "", 4),
        ("RSP_alfam", "alpha_m", "", 4),
        ("RSP_gammar", "gamma_r", "", 4),
        ("RSP_alfap", "alpha_p", "", 4),
        ("RSP_alfat", "alpha_t", "", 4),
    ),
}


def load_json(path: Path) -> object | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def copy_if_exists(source: Path, destination: Path, site_root: Path) -> str | None:
    if not source.exists():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination.relative_to(site_root).as_posix()


def sanitize_json_value(value: object, rre_root: Path) -> object:
    if isinstance(value, dict):
        return {key: sanitize_json_value(item, rre_root) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_json_value(item, rre_root) for item in value]
    if isinstance(value, str):
        root_text = str(rre_root)
        root_forward = rre_root.as_posix()
        root_escaped = root_text.replace("\\", "\\\\")
        sanitized = (
            value.replace(root_escaped, "<local-rre-root>")
            .replace(root_text, "<local-rre-root>")
            .replace(root_forward, "<local-rre-root>")
        )
        if re.match(r"^[A-Za-z]:[\\/]", sanitized):
            return "<local-path>"
        return sanitized.replace("\\", "/") if "<local-rre-root>" in sanitized else sanitized
    return value


def copy_json_if_exists(source: Path, destination: Path, site_root: Path, rre_root: Path) -> str | None:
    if not source.exists():
        return None
    value = load_json(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(sanitize_json_value(value, rre_root), indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return destination.relative_to(site_root).as_posix()


def sanitize_text_value(text: str, rre_root: Path) -> str:
    sanitized = (
        text.replace(str(rre_root), "<local-rre-root>")
        .replace(rre_root.as_posix(), "<local-rre-root>")
        .replace("\\", "/")
    )
    return re.sub(r"\b[A-Za-z]:/[^,\]\}\s]+", "<local-path>", sanitized)


def copy_text_if_exists(source: Path, destination: Path, site_root: Path, rre_root: Path) -> str | None:
    if not source.exists():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        sanitize_text_value(source.read_text(encoding="utf-8"), rre_root),
        encoding="utf-8",
        newline="\n",
    )
    return destination.relative_to(site_root).as_posix()


def parse_manifest_number(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace("d", "e").replace("D", "e")
    try:
        return float(text)
    except ValueError:
        return None


def normalized_trend_row(row: dict[str, object]) -> dict[str, object]:
    window_start = row.get("window_start_period")
    window_end = row.get("window_end_period")
    if isinstance(window_start, (float, int)) and float(window_start).is_integer():
        window_start = int(window_start)
    if isinstance(window_end, (float, int)) and float(window_end).is_integer():
        window_end = int(window_end)
    return {
        "source_kind": row.get("source_kind"),
        "source": row.get("source"),
        "cycle_count": window_end,
        "last_cycle_count_used": row.get("window_cycles"),
        "window_start_period_number": window_start,
        "window_end_period_number": window_end,
        "last_period_number": window_end,
        "gamma_peak_to_peak_last_window": row.get("gamma_peak_to_peak"),
        "period_fractional_peak_to_peak_last_window": row.get("period_fractional_peak_to_peak"),
        "delta_r_fractional_peak_to_peak_last_window": row.get("delta_r_fractional_peak_to_peak"),
        "steps_median_last_window": row.get("steps_median"),
        "steps_min_last_window": row.get("steps_min"),
        "steps_max_last_window": row.get("steps_max"),
        "max_vsurf_div_cs_median_last_window": row.get("max_vsurf_div_cs_median"),
        "max_vsurf_div_cs_first_last_window": [
            row.get("max_vsurf_div_cs_first"),
            row.get("max_vsurf_div_cs_last"),
        ],
        "max_vsurf_div_cs_slope_per_cycle_last_window": row.get("max_vsurf_div_cs_slope_per_cycle"),
        "max_vsurf_div_cs_min_last_window": row.get("max_vsurf_div_cs_min"),
        "max_vsurf_div_cs_max_last_window": row.get("max_vsurf_div_cs_max"),
        "has_full_window": row.get("window_cycles") is not None,
        "converged_gamma": row.get("converged_gamma"),
        "converged_period": row.get("converged_period"),
        "converged_delta_r": row.get("converged_delta_r"),
        "converged_exact": row.get("converged_exact"),
        "limit_cycle_converged": row.get("converged_exact"),
        "display_source": "convergence_trends_last100.latest_by_model",
    }


def convergence_window_end(row: dict[str, object]) -> float | None:
    return parse_manifest_number(row.get("window_end_period_number") or row.get("last_period_number") or row.get("cycle_count"))


def fmt_float(value: object, digits: int = 4) -> str:
    number = parse_manifest_number(value)
    if number is None:
        return "..."
    return f"{number:.{digits}g}"


def manifest_field_varies(manifest: list[dict[str, object]], field: str) -> bool:
    values: list[float] = []
    for record in manifest:
        number = parse_manifest_number(record.get(field))
        if number is None:
            continue
        if not any(abs(number - existing) <= max(1.0e-12, 1.0e-9 * max(abs(number), abs(existing), 1.0)) for existing in values):
            values.append(number)
    return len(values) > 1


def varied_parameter_groups(manifest: list[dict[str, object]]) -> dict[str, list[tuple[str, str, str, int]]]:
    groups: dict[str, list[tuple[str, str, str, int]]] = {}
    for group_name, fields in PARAMETER_GROUPS.items():
        varied = [field for field in fields if manifest_field_varies(manifest, field[0])]
        if varied:
            groups[group_name] = varied
    return groups


def parameter_groups_for_record(
    record: dict[str, object],
    groups: dict[str, list[tuple[str, str, str, int]]],
) -> dict[str, list[dict[str, str]]]:
    rendered: dict[str, list[dict[str, str]]] = {}
    for group_name, fields in groups.items():
        cells: list[dict[str, str]] = []
        for field, label, unit, digits in fields:
            value = fmt_float(record.get(field), digits)
            if unit:
                value = f"{value} {unit}"
            cells.append({"label": label, "value": value})
        if cells:
            rendered[group_name] = cells
    return rendered


def fmt_cycles(value: object) -> str:
    number = parse_manifest_number(value)
    if number is None:
        return "..."
    return f"{number:.0f}"


def growth_summary_html(path: Path | None) -> str:
    if path is None:
        return ""
    data = load_json(path)
    if not isinstance(data, dict):
        return ""
    outlook = data.get("growth_outlook")
    if not isinstance(outlook, dict):
        return ""
    bits = [
        f"DeltaR window {fmt_float(outlook.get('delta_r_criterion_factor'), 3)}x criterion",
        f"amplitude doubles in {fmt_cycles(outlook.get('doubling_cycles'))} cycles",
        rf"max v<sub>surf</sub>/c<sub>s</sub> {fmt_float(outlook.get('max_vsurf_div_cs_latest'), 3)}",
        rf"v<sub>surf</sub>/c<sub>s</sub>=0.8 in {fmt_cycles(outlook.get('cycles_to_vsurf_div_cs_0p8'))} cycles",
    ]
    return f'<p class="metric-row">{" | ".join(bits)}</p>'


def file_size_mb(path: Path | None) -> float | None:
    if path is None or not path.exists():
        return None
    return path.stat().st_size / (1024.0 * 1024.0)


def discover_models(rre_root: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    manifest_path = rre_root / "rsp_batch_runs" / "inputs" / "manifest.json"
    live_status_path = rre_root / "rsp_batch_runs" / "output" / "live_status.json"
    manifest = load_json(manifest_path)
    live_status = load_json(live_status_path)
    if not isinstance(manifest, list):
        raise RuntimeError(f"Could not load manifest list: {manifest_path}")
    live_by_id: dict[str, dict[str, object]] = {}
    if isinstance(live_status, dict):
        for model in live_status.get("models", []):
            if isinstance(model, dict) and model.get("model_id"):
                live_by_id[str(model["model_id"])] = model
    return manifest, live_by_id


def convergence_by_model(rre_root: Path) -> dict[str, dict[str, object]]:
    path = rre_root / "rsp_batch_runs" / "output" / "convergence_summary_last100.json"
    data = load_json(path)
    rows = data.get("models", []) if isinstance(data, dict) else []
    by_model = {
        str(row.get("model_id")): row
        for row in rows
        if isinstance(row, dict) and row.get("model_id")
    }
    trends = load_json(rre_root / "rsp_batch_runs" / "output" / "convergence_trends_last100.json")
    trend_rows = trends.get("latest_by_model", {}) if isinstance(trends, dict) else {}
    if isinstance(trend_rows, dict):
        for model_id, row in trend_rows.items():
            if not isinstance(row, dict):
                continue
            normalized = normalized_trend_row(row)
            trend_end = convergence_window_end(normalized)
            current_end = convergence_window_end(by_model.get(str(model_id), {}))
            if trend_end is not None and (current_end is None or trend_end > current_end):
                by_model[str(model_id)] = normalized
    forecast = load_json(rre_root / "rsp_batch_runs" / "output" / "convergence_forecast_last100.json")
    forecast_rows = forecast.get("models", []) if isinstance(forecast, dict) else []
    for row in forecast_rows:
        if not isinstance(row, dict) or not row.get("model_id"):
            continue
        model_id = str(row["model_id"])
        target = by_model.setdefault(model_id, {})
        if target.get("gate_blocking_metrics") in {None, ""}:
            target["gate_blocking_metrics"] = row.get("blocking_metrics")
        if target.get("gate_blocking_requirements") in {None, ""}:
            target["gate_blocking_requirements"] = row.get("blocking_metrics")
        if target.get("gate_forecast_status") in {None, ""}:
            target["gate_forecast_status"] = row.get("forecast_status")
        if target.get("gate_gamma_ratio_to_tolerance") is None:
            target["gate_gamma_ratio_to_tolerance"] = row.get("gamma_ratio_to_tolerance")
        if target.get("gate_period_ratio_to_tolerance") is None:
            target["gate_period_ratio_to_tolerance"] = row.get("period_ratio_to_tolerance")
        if target.get("gate_delta_r_ratio_to_tolerance") is None:
            target["gate_delta_r_ratio_to_tolerance"] = row.get("delta_r_ratio_to_tolerance")
        if target.get("gate_delta_r_periods_to_tolerance_linear") is None:
            target["gate_delta_r_periods_to_tolerance_linear"] = row.get("delta_r_periods_to_tolerance_linear")
    gate = load_json(rre_root / "rsp_batch_runs" / "output" / "convergence_gate_audit.json")
    gate_rows = gate.get("models", []) if isinstance(gate, dict) else []
    for row in gate_rows:
        if not isinstance(row, dict) or not row.get("model_id"):
            continue
        model_id = str(row["model_id"])
        target = by_model.setdefault(model_id, {})
        target.update(
            {
                "gate_blocking_metrics": row.get("blocking_metrics"),
                "gate_blocking_requirements": row.get("blocking_requirements"),
                "gate_quality_flags": row.get("quality_flags"),
                "gate_forecast_status": row.get("forecast_status"),
                "gate_post_convergence_action": row.get("post_convergence_action"),
                "gate_latest_surface_velocity_status": row.get("latest_surface_velocity_status"),
                "gate_convergence_window_end_period": row.get("convergence_window_end_period"),
                "gate_metric_lag_periods": row.get("gate_metric_lag_periods"),
                "gate_gamma_ratio_to_tolerance": row.get("gamma_ratio_to_tolerance"),
                "gate_period_ratio_to_tolerance": row.get("period_ratio_to_tolerance"),
                "gate_delta_r_ratio_to_tolerance": row.get("delta_r_ratio_to_tolerance"),
                "gate_delta_r_periods_to_tolerance_linear": row.get("delta_r_periods_to_tolerance_linear"),
            }
        )
    return by_model


def cycle_boundary_by_model(rre_root: Path) -> dict[str, dict[str, object]]:
    path = rre_root / "rsp_batch_runs" / "output" / "cycle_boundary_audit.json"
    data = load_json(path)
    rows = data.get("rows", []) if isinstance(data, dict) else []
    return {
        str(row.get("model_id")): row
        for row in rows
        if isinstance(row, dict) and row.get("model_id")
    }


def metric_label(metric: str) -> str:
    labels = {
        "gamma": "Gamma",
        "period": "P",
        "delta_r": "DeltaR",
        "100_cycle_window": "100-cycle window",
        "gamma_missing": "Gamma",
        "period_missing": "P",
        "delta_r_missing": "DeltaR",
        "surface_velocity_supersonic": "surface velocity supersonic",
        "surface_velocity_watch": "surface velocity watch",
    }
    return labels.get(metric.strip().lower(), metric.strip())


def convergence_text(convergence: dict[str, object] | None) -> str:
    if not convergence:
        return ""
    if convergence.get("converged_exact") is True:
        return "strict convergence passed"
    source = convergence.get("source_kind") or "unknown source"
    cycles = convergence.get("cycle_count")
    gamma = convergence.get("gamma_peak_to_peak_last_window")
    period = convergence.get("period_fractional_peak_to_peak_last_window")
    delta_r = convergence.get("delta_r_fractional_peak_to_peak_last_window")
    max_vsurf = convergence.get("max_vsurf_div_cs_max_last_window")
    bits = [f"strict convergence pending", f"{source}", f"{cycles} cycles"]
    gate_window_end = convergence.get("gate_convergence_window_end_period")
    metric_lag = convergence.get("gate_metric_lag_periods")
    if gate_window_end not in {None, ""}:
        window_text = f"gate window through period {gate_window_end}"
        try:
            lag_value = float(metric_lag)
        except (TypeError, ValueError):
            lag_value = 0.0
        if lag_value > 0:
            window_text += f"; live run {lag_value:.0f} period ahead"
        bits.append(window_text)
    blocking = convergence.get("gate_blocking_requirements") or convergence.get("gate_blocking_metrics")
    if blocking:
        metrics = "/".join(metric_label(item) for item in str(blocking).split(",") if item.strip())
        gate_text = f"waiting on {metrics}"
        forecast = convergence.get("gate_forecast_status")
        if forecast:
            gate_text += f"; forecast {forecast}"
        delta_r_ratio = convergence.get("gate_delta_r_ratio_to_tolerance")
        if delta_r_ratio is not None and "delta_r" in str(blocking).lower():
            gate_text += f"; DeltaR {float(delta_r_ratio):.3g}x tol"
        bits.append(gate_text)
    quality_flags = convergence.get("gate_quality_flags")
    if quality_flags:
        flags = "/".join(metric_label(item) for item in str(quality_flags).split(",") if item.strip())
        bits.append(f"quality flag: {flags}")
    if convergence.get("has_full_window") is not True:
        used = convergence.get("last_cycle_count_used") or cycles or 0
        bits.append(f"{used}/100-cycle window")
        return " | ".join(bits)
    if gamma is None:
        bits.append("Gamma not recorded")
    else:
        bits.append(f"Gamma ptp {float(gamma):.3g}")
    if period is not None:
        bits.append(f"P frac {float(period):.3g}")
    if delta_r is not None:
        bits.append(f"DeltaR frac {float(delta_r):.3g}")
    if max_vsurf is not None:
        bits.append(f"window max v_surf/c_s {float(max_vsurf):.3g}")
    return " | ".join(bits)


def strict_convergence_passed(record: dict[str, object], convergence: dict[str, object] | None) -> bool:
    if record.get("registered_existing"):
        return True
    return bool(convergence and convergence.get("converged_exact") is True)


def copy_model_assets(
    rre_root: Path,
    output_dir: Path,
    record: dict[str, object],
    live_record: dict[str, object] | None,
    convergence: dict[str, object] | None,
    boundary: dict[str, object] | None,
) -> dict[str, object]:
    model_id = str(record["model_id"])
    source_output_dir = Path(str(record["output_dir"]))
    product_stem = str(record["product_stem"])
    prefix = str(record["prefix"])
    asset_dir = output_dir / "models" / model_id
    asset_dir.mkdir(parents=True, exist_ok=True)

    verify_path = source_output_dir / "verification_summary.json"
    verify = load_json(verify_path)
    verified = isinstance(verify, dict) and verify.get("passed") is True
    verification_failed = isinstance(verify, dict) and verify.get("passed") is False
    convergence_passed = strict_convergence_passed(record, convergence)
    if live_record is not None and live_record.get("trusted_animation") is not None:
        trusted_animation = live_record.get("trusted_animation") is True
        trusted_animation_reason = str(
            live_record.get("trusted_animation_reason")
            or live_record.get("diagnostic_animation_reason")
            or ""
        )
    else:
        trusted_animation = verified and convergence_passed
        if trusted_animation:
            trusted_animation_reason = "verification passed and strict limit-cycle convergence gate passed"
        elif not verified:
            trusted_animation_reason = "verification has not passed"
        elif not convergence_passed:
            trusted_animation_reason = "strict limit-cycle convergence gate has not passed"
        else:
            trusted_animation_reason = "animation is not trusted"

    copied: dict[str, str | None] = {}
    source_map = {
        "gif": source_output_dir / f"{product_stem}.gif",
        "png": source_output_dir / f"{product_stem}.png",
        "summary": source_output_dir / f"{product_stem}_summary.json",
        "fourier_fixed_png": source_output_dir / f"{prefix}_fourier_fixed_cells_vs_logT.png",
        "fourier_fixed_csv": source_output_dir / f"{prefix}_fourier_fixed_cells_vs_logT.csv",
        "fourier_fixed_summary": source_output_dir / f"{prefix}_fourier_fixed_cells_vs_logT_summary.json",
        "verify": verify_path,
        "lightcurve_csv": source_output_dir / f"{prefix}_final_cycle_lightcurve.csv",
        "final_cycle_summary": source_output_dir / f"{prefix}_final_cycle_summary.json",
        "run_status": source_output_dir / "run_status.json",
    }
    for key, source in source_map.items():
        if source.suffix.lower() == ".json":
            copied[key] = copy_json_if_exists(source, asset_dir / source.name, output_dir, rre_root)
        elif source.suffix.lower() in {".csv", ".txt"}:
            copied[key] = copy_text_if_exists(source, asset_dir / source.name, output_dir, rre_root)
        else:
            copied[key] = copy_if_exists(source, asset_dir / source.name, output_dir)

    summary = load_json(source_map["summary"])
    status = "pending"
    if trusted_animation:
        status = "verified"
    elif verified and not convergence_passed:
        status = "awaiting convergence"
    elif verification_failed:
        status = "verification failed"
    stale_stage_outputs = live_record.get("stale_completed_stage_outputs") if live_record else []
    if stale_stage_outputs:
        status = "stale stage output"
    if live_record and live_record.get("active_stage"):
        status = f"running: {live_record.get('active_stage')}"
    elif live_record and live_record.get("retry_pending"):
        retry_stages = live_record.get("retry_pending_stages") or []
        first_retry = retry_stages[0] if isinstance(retry_stages, list) and retry_stages else {}
        stage_name = first_retry.get("stage") if isinstance(first_retry, dict) else None
        status = f"queued retry: {stage_name}" if stage_name else "queued retry"
    if not source_map["gif"].exists() and not source_map["png"].exists():
        if not (
            live_record
            and (live_record.get("retry_pending") or live_record.get("active_stage"))
        ):
            status = "not rendered"

    phase_breaks = []
    if isinstance(summary, dict):
        phase_breaks = summary.get("phase_curve_break_phases") or []

    return {
        "model_id": model_id,
        "run_name": record.get("run_name"),
        "registered_existing": bool(record.get("registered_existing")),
        "status": status,
        "M": record.get("RSP_mass"),
        "Teff": record.get("RSP_Teff"),
        "L": record.get("RSP_L"),
        "Z": record.get("RSP_Z"),
        "assets": copied,
        "gif_mb": file_size_mb(source_map["gif"]),
        "profile_count": live_record.get("profile_count") if live_record else None,
        "latest_period": live_record.get("latest_period") if live_record else None,
        "active_history_period": live_record.get("active_history_period") if live_record else None,
        "active_history_period_status": live_record.get("active_history_period_status") if live_record else None,
        "active_history_period_gap": live_record.get("active_history_period_gap") if live_record else None,
        "latest_history_model": live_record.get("latest_history_model") if live_record else None,
        "latest_history_mtime": live_record.get("latest_history_mtime") if live_record else None,
        "latest_period_days": live_record.get("latest_period_days") if live_record else None,
        "latest_delta_r": live_record.get("latest_delta_r") if live_record else None,
        "latest_steps": live_record.get("latest_steps") if live_record else None,
        "latest_max_vsurf_div_cs": live_record.get("latest_max_vsurf_div_cs") if live_record else None,
        "latest_surface_velocity_status": live_record.get("latest_surface_velocity_status") if live_record else None,
        "max_periods": live_record.get("max_periods") if live_record else None,
        "retry_pending": live_record.get("retry_pending") if live_record else None,
        "retry_pending_stages": live_record.get("retry_pending_stages") if live_record else None,
        "stale_completed_stage_outputs": stale_stage_outputs,
        "converged_exact": convergence.get("converged_exact") if convergence and not record.get("registered_existing") else None,
        "convergence": "" if record.get("registered_existing") else convergence_text(convergence),
        "verification_passed": live_record.get("verification_passed") if live_record else None,
        "verification_failures": verify.get("failures", []) if isinstance(verify, dict) else [],
        "animation_trusted": trusted_animation,
        "animation_trusted_reason": trusted_animation_reason,
        "phase_curve_break_phases": phase_breaks,
        "cycle_boundary": boundary or {},
        "convergence_blocking_metrics": convergence.get("gate_blocking_metrics") if convergence else None,
        "convergence_blocking_requirements": convergence.get("gate_blocking_requirements") if convergence else None,
        "convergence_quality_flags": convergence.get("gate_quality_flags") if convergence else None,
        "convergence_forecast_status": convergence.get("gate_forecast_status") if convergence else None,
        "gamma_ratio_to_tolerance": convergence.get("gate_gamma_ratio_to_tolerance") if convergence else None,
        "period_ratio_to_tolerance": convergence.get("gate_period_ratio_to_tolerance") if convergence else None,
        "delta_r_ratio_to_tolerance": convergence.get("gate_delta_r_ratio_to_tolerance") if convergence else None,
        "delta_r_periods_to_tolerance_linear": convergence.get("gate_delta_r_periods_to_tolerance_linear") if convergence else None,
    }


def copy_batch_assets(rre_root: Path, output_dir: Path) -> dict[str, str | None]:
    batch_source_dir = rre_root / "rsp_batch_runs" / "output"
    inputs_dir = rre_root / "rsp_batch_runs" / "inputs"
    metadata_dir = output_dir / "metadata"
    diagnostics_dir = output_dir / "cycle_diagnostics"
    copied: dict[str, str | None] = {}
    for name in (
        "live_status.json",
        "batch_audit_summary.json",
        "cycle_modulation_summary.json",
        "cycle_modulation_summary.csv",
        "convergence_summary_last100.json",
        "convergence_summary_last100.csv",
        "convergence_summary_last100.png",
        "convergence_trends_last100.json",
        "convergence_trends_last100.csv",
        "convergence_trends_last100.png",
        "convergence_trends_exact_last100.png",
        "convergence_forecast_last100.json",
        "convergence_forecast_last100.csv",
        "convergence_forecast_last100.png",
        "convergence_gate_audit.json",
        "convergence_gate_audit.csv",
        "phase_seam_audit.json",
        "phase_seam_audit.csv",
        "cycle_boundary_audit.json",
        "cycle_boundary_audit.csv",
        "post_convergence_products_status.json",
        "quality_extension_status.json",
    ):
        source = batch_source_dir / name
        if source.suffix.lower() == ".json":
            copied[name] = copy_json_if_exists(source, metadata_dir / name, output_dir, rre_root)
        elif source.suffix.lower() in {".csv", ".txt"}:
            copied[name] = copy_text_if_exists(source, metadata_dir / name, output_dir, rre_root)
        else:
            copied[name] = copy_if_exists(source, metadata_dir / name, output_dir)
    copied["manifest.json"] = copy_json_if_exists(
        inputs_dir / "manifest.json",
        metadata_dir / "manifest.json",
        output_dir,
        rre_root,
    )
    for diagnostic in (batch_source_dir / "cycle_diagnostics").glob("*.png"):
        copied[diagnostic.name] = copy_if_exists(diagnostic, diagnostics_dir / diagnostic.name, output_dir)
    for diagnostic in batch_source_dir.glob("*diagnostic*.png"):
        copied[diagnostic.name] = copy_if_exists(diagnostic, diagnostics_dir / diagnostic.name, output_dir)
    for diagnostic in batch_source_dir.glob("*growth_diagnostic*.json"):
        copied[diagnostic.name] = copy_json_if_exists(diagnostic, metadata_dir / diagnostic.name, output_dir, rre_root)
    return copied


def write_manifest_csv(output_dir: Path, models: list[dict[str, object]]) -> None:
    path = output_dir / "metadata" / "models.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["model_id", "run_name", "status", "M", "Teff", "L", "Z", "gif_mb", "converged_exact"])
        for model in models:
            writer.writerow(
                [
                    model["model_id"],
                    model.get("run_name") or "",
                    model["status"],
                    fmt_float(model.get("M")),
                    fmt_float(model.get("Teff")),
                    fmt_float(model.get("L")),
                    fmt_float(model.get("Z")),
                    f"{model['gif_mb']:.2f}" if model.get("gif_mb") is not None else "",
                    model.get("converged_exact"),
                ]
            )


def write_fourier_inventory(output_dir: Path, models: list[dict[str, object]]) -> dict[str, str]:
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for model in models:
        assets = model.get("assets")
        assets = assets if isinstance(assets, dict) else {}
        summary_rel = assets.get("fourier_fixed_summary")
        png_rel = assets.get("fourier_fixed_png")
        csv_rel = assets.get("fourier_fixed_csv")
        summary = load_json(output_dir / str(summary_rel)) if summary_rel else None
        summary = summary if isinstance(summary, dict) else {}
        q_range = summary.get("q_env_range") if isinstance(summary.get("q_env_range"), list) else []
        t_range = summary.get("temperature_range_K") if isinstance(summary.get("temperature_range_K"), list) else []
        product_present = bool(summary_rel and png_rel and csv_rel)
        product_state = "pending"
        provenance = ""
        if product_present:
            pending_reason = ""
            if model.get("registered_existing") or model.get("converged_exact") is True:
                product_state = "reference" if model.get("registered_existing") else "converged"
                provenance = "fixed-cell Fourier depth diagnostic from the final/reference deep profiles"
            else:
                product_state = "provisional"
                provenance = (
                    "fixed-cell Fourier depth diagnostic from currently available deep profiles; "
                    "will be refreshed after the strict convergence gate passes"
                )
        elif model.get("profile_count"):
            pending_reason = "fixed-cell Fourier product has not been built from available deep profiles"
        elif "running" in str(model.get("status")):
            pending_reason = "waiting for converged post-saturation deep profiles"
        else:
            pending_reason = "waiting for post-convergence deep profiles"
        rows.append(
            {
                "model_id": model.get("model_id"),
                "run_name": model.get("run_name"),
                "status": "present" if product_present else "pending",
                "product_state": product_state,
                "provenance": provenance,
                "pending_reason": pending_reason,
                "model_status": model.get("status"),
                "animation_trusted": model.get("animation_trusted"),
                "converged_exact": model.get("converged_exact"),
                "convergence_blocking_metrics": model.get("convergence_blocking_metrics"),
                "convergence_blocking_requirements": model.get("convergence_blocking_requirements"),
                "convergence_quality_flags": model.get("convergence_quality_flags"),
                "convergence_forecast_status": model.get("convergence_forecast_status"),
                "gamma_ratio_to_tolerance": model.get("gamma_ratio_to_tolerance"),
                "period_ratio_to_tolerance": model.get("period_ratio_to_tolerance"),
                "delta_r_ratio_to_tolerance": model.get("delta_r_ratio_to_tolerance"),
                "delta_r_periods_to_tolerance_linear": model.get("delta_r_periods_to_tolerance_linear"),
                "latest_period": model.get("latest_period"),
                "max_periods": model.get("max_periods"),
                "profile_count": model.get("profile_count"),
                "fourier_png": png_rel,
                "fourier_csv": csv_rel,
                "fourier_summary": summary_rel,
                "fit_harmonics": summary.get("fit_harmonics"),
                "num_profiles": summary.get("num_profiles"),
                "point_count": summary.get("point_count") or summary.get("num_fixed_cells_plotted"),
                "q_env_min": q_range[0] if len(q_range) >= 2 else None,
                "q_env_max": q_range[1] if len(q_range) >= 2 else None,
                "temperature_min_K": t_range[0] if len(t_range) >= 2 else None,
                "temperature_max_K": t_range[1] if len(t_range) >= 2 else None,
                "photosphere_temperature_K": summary.get("photosphere_temperature_K"),
                "photosphere_q_env": summary.get("photosphere_q_env"),
            }
        )

    json_path = metadata_dir / "fourier_inventory.json"
    csv_path = metadata_dir / "fourier_inventory.csv"
    json_path.write_text(json.dumps({"models": rows}, indent=2) + "\n", encoding="utf-8")
    fieldnames = [
        "model_id",
        "run_name",
        "status",
        "product_state",
        "provenance",
        "pending_reason",
        "model_status",
        "animation_trusted",
        "converged_exact",
        "convergence_blocking_metrics",
        "convergence_blocking_requirements",
        "convergence_quality_flags",
        "convergence_forecast_status",
        "gamma_ratio_to_tolerance",
        "period_ratio_to_tolerance",
        "delta_r_ratio_to_tolerance",
        "delta_r_periods_to_tolerance_linear",
        "latest_period",
        "max_periods",
        "profile_count",
        "fit_harmonics",
        "num_profiles",
        "point_count",
        "q_env_min",
        "q_env_max",
        "temperature_min_K",
        "temperature_max_K",
        "photosphere_temperature_K",
        "photosphere_q_env",
        "fourier_png",
        "fourier_csv",
        "fourier_summary",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})
    return {
        "fourier_inventory.json": str(json_path.relative_to(output_dir)).replace("\\", "/"),
        "fourier_inventory.csv": str(csv_path.relative_to(output_dir)).replace("\\", "/"),
    }


def write_analysis_completion_status(output_dir: Path, models: list[dict[str, object]]) -> dict[str, str]:
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for model in models:
        assets = model.get("assets")
        assets = assets if isinstance(assets, dict) else {}
        registered_reference = bool(model.get("registered_existing"))
        strict_convergence = registered_reference or model.get("converged_exact") is True
        canonical_animation_present = bool(assets.get("gif") and assets.get("summary") and assets.get("verify"))
        fixed_cell_fourier_present = bool(
            assets.get("fourier_fixed_png")
            and assets.get("fourier_fixed_csv")
            and assets.get("fourier_fixed_summary")
        )
        final_cycle_products_present = bool(
            assets.get("final_cycle_summary")
            and assets.get("lightcurve_csv")
        )
        animation_trusted = bool(model.get("animation_trusted"))
        verification_passed = model.get("verification_passed") is True or model.get("status") == "verified"
        final_ready = (
            strict_convergence
            and canonical_animation_present
            and fixed_cell_fourier_present
            and final_cycle_products_present
            and (animation_trusted or registered_reference)
            and (verification_passed or registered_reference)
        )
        missing = []
        if not strict_convergence:
            missing.append("strict convergence")
        if not canonical_animation_present:
            missing.append("canonical animation")
        if not fixed_cell_fourier_present:
            missing.append("fixed-cell Fourier depth diagnostic")
        if not final_cycle_products_present:
            missing.append("final-cycle products")
        if not animation_trusted and not registered_reference:
            missing.append("trusted animation")
        if not verification_passed and not registered_reference:
            missing.append("verification")
        rows.append(
            {
                "model_id": model.get("model_id"),
                "run_name": model.get("run_name"),
                "final_ready": final_ready,
                "registered_reference": registered_reference,
                "strict_convergence": strict_convergence,
                "canonical_animation_present": canonical_animation_present,
                "fixed_cell_fourier_present": fixed_cell_fourier_present,
                "final_cycle_products_present": final_cycle_products_present,
                "animation_trusted": animation_trusted,
                "verification_passed": verification_passed,
                "missing": ", ".join(missing),
                "active_stage": str(model.get("status")).removeprefix("running: ") if "running:" in str(model.get("status")) else "",
                "latest_period": model.get("latest_period"),
                "max_periods": model.get("max_periods"),
                "convergence_blocking_requirements": model.get("convergence_blocking_requirements"),
                "convergence_quality_flags": model.get("convergence_quality_flags"),
                "convergence_forecast_status": model.get("convergence_forecast_status"),
                "delta_r_ratio_to_tolerance": model.get("delta_r_ratio_to_tolerance"),
                "gamma_ratio_to_tolerance": model.get("gamma_ratio_to_tolerance"),
                "period_ratio_to_tolerance": model.get("period_ratio_to_tolerance"),
            }
        )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "analysis_complete": all(row["final_ready"] for row in rows),
        "models_total": len(rows),
        "models_final_ready": sum(1 for row in rows if row["final_ready"]),
        "models_waiting": sum(1 for row in rows if not row["final_ready"]),
        "models_with_fixed_cell_fourier": sum(1 for row in rows if row["fixed_cell_fourier_present"]),
        "models_with_trusted_animation": sum(1 for row in rows if row["animation_trusted"] or row["registered_reference"]),
        "models_with_strict_convergence": sum(1 for row in rows if row["strict_convergence"]),
    }
    json_path = metadata_dir / "analysis_completion_status.json"
    csv_path = metadata_dir / "analysis_completion_status.csv"
    json_path.write_text(json.dumps({"summary": summary, "models": rows}, indent=2) + "\n", encoding="utf-8")
    fieldnames = [
        "model_id",
        "run_name",
        "final_ready",
        "registered_reference",
        "strict_convergence",
        "canonical_animation_present",
        "fixed_cell_fourier_present",
        "final_cycle_products_present",
        "animation_trusted",
        "verification_passed",
        "missing",
        "active_stage",
        "latest_period",
        "max_periods",
        "convergence_blocking_requirements",
        "convergence_quality_flags",
        "convergence_forecast_status",
        "delta_r_ratio_to_tolerance",
        "gamma_ratio_to_tolerance",
        "period_ratio_to_tolerance",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})
    return {
        "analysis_completion_status.json": str(json_path.relative_to(output_dir)).replace("\\", "/"),
        "analysis_completion_status.csv": str(csv_path.relative_to(output_dir)).replace("\\", "/"),
    }


def parameter_table_html(parameter_groups: object) -> str:
    if not isinstance(parameter_groups, dict) or not parameter_groups:
        return ""
    rows: list[str] = []
    for group_name, cells in parameter_groups.items():
        if not isinstance(cells, list) or not cells:
            continue
        rendered_cells = []
        for cell in cells:
            if not isinstance(cell, dict):
                continue
            label = html.escape(str(cell.get("label", "")))
            value = html.escape(str(cell.get("value", "")))
            rendered_cells.append(f"<td><span>{label}</span>{value}</td>")
        if rendered_cells:
            rows.append(f'<tr><th scope="row">{html.escape(str(group_name))}</th>{"".join(rendered_cells)}</tr>')
    if not rows:
        return ""
    return '<table class="param-table"><tbody>' + "".join(rows) + "</tbody></table>"


def gate_waiting_text(model: dict[str, object]) -> str:
    blocking = model.get("convergence_blocking_requirements") or model.get("convergence_blocking_metrics")
    if not blocking:
        return ""
    metrics = [metric_label(item) for item in str(blocking).split(",") if item.strip()]
    if not metrics:
        return ""
    return "waiting on " + "/".join(metrics)


def card_html(model: dict[str, object]) -> str:
    assets = model["assets"]
    assert isinstance(assets, dict)
    gif = assets.get("gif")
    png = assets.get("png")
    summary = assets.get("summary")
    verify = assets.get("verify")
    lightcurve = assets.get("lightcurve_csv")
    fourier_png = assets.get("fourier_fixed_png")
    fourier_csv = assets.get("fourier_fixed_csv")
    fourier_summary = assets.get("fourier_fixed_summary")
    fourier_has_product = bool(fourier_png and fourier_csv and fourier_summary)
    image = gif or png
    badge_class = (
        "ok"
        if model["status"] == "verified"
        else "bad"
        if "failed" in str(model["status"])
        else "warn"
        if "running" in str(model["status"])
        else "muted"
    )
    phase_breaks = model.get("phase_curve_break_phases") or []
    break_text = ""
    if phase_breaks:
        values = ", ".join(f"{float(value):.3f}" for value in phase_breaks)
        break_text = f"<span>phase-curve break: {html.escape(values)}</span>"
    links = []
    for label, href in (
        ("GIF", gif),
        ("PNG", png),
        ("summary", summary),
        ("verify", verify),
        ("lightcurve", lightcurve),
        ("Fourier CSV", fourier_csv),
        ("Fourier summary", fourier_summary),
    ):
        if href:
            links.append(f'<a href="{html.escape(str(href))}">{label}</a>')
    if not links:
        links.append('<span class="muted">awaiting render</span>')
    if image:
        media_class = "media" if model.get("animation_trusted") else "media diagnostic-media"
        image_html = f'<a class="{media_class}" href="{html.escape(str(image))}"><img src="{html.escape(str(image))}" alt="{html.escape(str(model["model_id"]))} animation"></a>'
    else:
        if str(model["status"]) == "awaiting convergence":
            placeholder = "strict convergence pending"
        elif "failed" in str(model["status"]):
            placeholder = "verification failed"
        else:
            placeholder = "queued"
        image_html = f'<div class="media placeholder">{html.escape(placeholder)}</div>'
    if fourier_has_product:
        if model.get("registered_existing") or model.get("converged_exact") is True:
            fourier_state_class = "ok"
            fourier_note = "Fixed-cell Fourier depth diagnostic"
        else:
            fourier_state_class = "warn"
            fourier_note = (
                "Provisional fixed-cell Fourier depth diagnostic; "
                "will refresh after strict convergence"
            )
            gate_text = gate_waiting_text(model)
            if gate_text:
                fourier_note += f" ({gate_text})"
        fourier_html = (
            f'<div class="fourier-note {fourier_state_class}">{html.escape(fourier_note)}</div>'
            f'<a class="fourier-diagnostic" href="{html.escape(str(fourier_png))}">'
            f'<img src="{html.escape(str(fourier_png))}" '
            f'alt="{html.escape(str(model["model_id"]))} fixed-cell Fourier depth diagnostic">'
            '</a>'
        )
    else:
        if model.get("profile_count"):
            fourier_pending = "Fourier depth diagnostic pending: product has not been built from available deep profiles."
        elif "running" in str(model.get("status")):
            fourier_pending = "Fourier depth diagnostic pending: waiting for converged post-saturation deep profiles."
        else:
            fourier_pending = "Fourier depth diagnostic pending: waiting for post-convergence deep profiles."
        gate_text = gate_waiting_text(model)
        if gate_text:
            fourier_pending += f" Current gate is {gate_text}."
        fourier_html = (
            '<div class="fourier-placeholder">'
            f"{html.escape(fourier_pending)}"
            "</div>"
        )
    progress_bits = []
    if model.get("retry_pending"):
        retry_stages = model.get("retry_pending_stages") or []
        retry_names = [
            str(item.get("stage"))
            for item in retry_stages
            if isinstance(item, dict) and item.get("stage")
        ]
        if retry_names:
            progress_bits.append(f"queued retry for {', '.join(retry_names)}")
    stale_stage_outputs = model.get("stale_completed_stage_outputs") or []
    if stale_stage_outputs:
        stale_names = [
            str(item.get("stage"))
            for item in stale_stage_outputs
            if isinstance(item, dict) and item.get("stage")
        ]
        progress_bits.append(
            "stale stage output"
            + (f": {', '.join(stale_names)}" if stale_names else "")
        )
    if model.get("profile_count"):
        progress_bits.append(f"{model['profile_count']} profiles")
    status_text = str(model.get("status"))
    if model.get("max_periods") and "running" in status_text:
        if model.get("active_history_period") is not None:
            period_text = f"active period {fmt_cycles(model['active_history_period'])} / {fmt_cycles(model['max_periods'])}"
            if model.get("active_history_period_status") == "overlap_replay" and model.get("latest_period"):
                period_text += f" (overlap; accepted {fmt_cycles(model['latest_period'])})"
            progress_bits.append(period_text)
        elif model.get("latest_period"):
            progress_bits.append(f"period {fmt_cycles(model['latest_period'])} / {fmt_cycles(model['max_periods'])}")
    if model.get("latest_history_model") and "running" in status_text and (
        ("running: create" in status_text) or not model.get("latest_period")
    ):
        progress_bits.append(f"model {model['latest_history_model']}")
    if model.get("latest_max_vsurf_div_cs") is not None:
        velocity_text = f"max v_surf/c_s {fmt_float(model.get('latest_max_vsurf_div_cs'), 3)}"
        if model.get("latest_surface_velocity_status"):
            velocity_text += f" ({model['latest_surface_velocity_status']})"
        progress_bits.append(velocity_text)
    if model.get("latest_steps") is not None and "running" in str(model.get("status")):
        progress_bits.append(f"{model['latest_steps']} steps/cycle")
    if model.get("convergence"):
        progress_bits.append(str(model["convergence"]))
    boundary = model.get("cycle_boundary")
    if isinstance(boundary, dict) and boundary:
        l_fraction = parse_manifest_number(boundary.get("boundary_luminosity_lsun_seam_fraction"))
        r_fraction = parse_manifest_number(boundary.get("boundary_radius_rsun_seam_fraction"))
        t_fraction = parse_manifest_number(boundary.get("boundary_teff_k_seam_fraction"))
        fractions = [value for value in (l_fraction, r_fraction, t_fraction) if value is not None]
        worst_fraction = max(fractions) if fractions else None
        if worst_fraction is not None and worst_fraction >= 0.01:
            progress_bits.append(
                "cycle boundary not closed: "
                f"L {100.0 * (l_fraction or 0.0):.2g}%, "
                f"R {100.0 * (r_fraction or 0.0):.2g}%, "
                f"Teff {100.0 * (t_fraction or 0.0):.2g}%"
            )
    if image and not model.get("animation_trusted"):
        reason = str(model.get("animation_trusted_reason") or "not trusted")
        progress_bits.append(f"diagnostic animation: {reason}")
    if model.get("gif_mb") is not None:
        progress_bits.append(f"{float(model['gif_mb']):.1f} MB GIF")
    failures = model.get("verification_failures") or []
    if failures:
        progress_bits.append("; ".join(str(item) for item in failures[:2]))
    parameter_table = parameter_table_html(model.get("parameter_groups"))
    return f"""
      <article class="card">
        <div class="card-head">
          <div>
            <h2>{html.escape(str(model["model_id"]))}</h2>
            <p>{html.escape(str(model.get("run_name") or ""))}</p>
          </div>
          <span class="badge {badge_class}">{html.escape(str(model["status"]))}</span>
        </div>
        {image_html}
        {fourier_html}
        <div class="body">
          {parameter_table}
          <p class="details">{html.escape(" | ".join(progress_bits))} {break_text}</p>
          <p class="links">{" ".join(links)}</p>
        </div>
      </article>
    """.strip()


def write_index(output_dir: Path, models: list[dict[str, object]], metadata_links: dict[str, str | None]) -> None:
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    trusted = sum(1 for model in models if model.get("animation_trusted"))
    verified = sum(1 for model in models if model.get("verification_passed") is True or model["status"] == "verified")
    cards = "\n".join(card_html(model) for model in models)
    convergence_png = metadata_links.get("convergence_summary_last100.png")
    convergence_trends_png = metadata_links.get("convergence_trends_last100.png")
    convergence_trends_exact_png = metadata_links.get("convergence_trends_exact_last100.png")
    convergence_forecast_png = metadata_links.get("convergence_forecast_last100.png")
    convergence_items = []
    for title, href, alt in (
        ("Limit-Cycle Convergence", convergence_png, "Strict limit-cycle convergence summary"),
        ("Rolling Trends", convergence_trends_png, "Rolling final-100-cycle convergence trends"),
        ("Exact-History Trends", convergence_trends_exact_png, "Rolling final-100-cycle convergence trends for exact-history runs"),
        ("Convergence Forecast", convergence_forecast_png, "Rolling convergence forecast"),
    ):
        if not href:
            continue
        convergence_items.append(
            f"""
        <figure>
          <a href="{html.escape(str(href))}"><img src="{html.escape(str(href))}" alt="{html.escape(alt)}"></a>
          <figcaption>{html.escape(title)}</figcaption>
        </figure>
"""
        )
    convergence_figure = ""
    if convergence_items:
        convergence_figure = f"""
    <section class="diagnostic compact-diagnostic">
      <h2>Convergence Diagnostics</h2>
      <div class="diagnostic-grid convergence-grid">
{''.join(convergence_items)}
      </div>
    </section>
"""
    convergence_trends_figure = ""
    convergence_trends_exact_figure = ""
    convergence_forecast_figure = ""
    growth_diagnostic_figures = ""
    growth_items = []
    for name, href in sorted(metadata_links.items()):
        if not href or not name.endswith("_growth_diagnostic.png"):
            continue
        label = name.removesuffix("_growth_diagnostic.png").replace("_", " ")
        json_name = name.removesuffix(".png") + ".json"
        json_href = metadata_links.get(json_name)
        json_link = (
            f' <a class="caption-link" href="{html.escape(str(json_href))}">JSON</a>'
            if json_href
            else ""
        )
        json_path = output_dir / str(json_href) if json_href else None
        metric_summary = growth_summary_html(json_path)
        growth_items.append(
            f"""
        <figure>
          <a href="{html.escape(str(href))}"><img src="{html.escape(str(href))}" alt="{html.escape(label)} active amplitude growth diagnostic"></a>
          <figcaption>{html.escape(label)} active amplitude growth{json_link}</figcaption>
          {metric_summary}
        </figure>
"""
        )
    if growth_items:
        growth_diagnostic_figures = f"""
    <section class="diagnostic">
      <h2>Active Amplitude Growth</h2>
      <div class="diagnostic-grid">
{''.join(growth_items)}
      </div>
    </section>
"""
    meta_links = []
    for label, href in (
        ("live status", metadata_links.get("live_status.json")),
        ("audit", metadata_links.get("batch_audit_summary.json")),
        ("manifest", metadata_links.get("manifest.json")),
        ("completion status", metadata_links.get("analysis_completion_status.json")),
        ("completion CSV", metadata_links.get("analysis_completion_status.csv")),
        ("Fourier inventory", metadata_links.get("fourier_inventory.json")),
        ("Fourier inventory CSV", metadata_links.get("fourier_inventory.csv")),
        ("cycle modulation", metadata_links.get("cycle_modulation_summary.json")),
        ("convergence", metadata_links.get("convergence_summary_last100.json")),
        ("convergence plot", convergence_png),
        ("convergence trends", metadata_links.get("convergence_trends_last100.json")),
        ("convergence trend plot", convergence_trends_png),
        ("exact-history trend plot", convergence_trends_exact_png),
        ("convergence forecast", metadata_links.get("convergence_forecast_last100.json")),
        ("convergence forecast plot", convergence_forecast_png),
        ("convergence gate audit", metadata_links.get("convergence_gate_audit.json")),
        ("phase seam audit", metadata_links.get("phase_seam_audit.json")),
        ("phase seam CSV", metadata_links.get("phase_seam_audit.csv")),
        ("cycle boundary audit", metadata_links.get("cycle_boundary_audit.json")),
        ("cycle boundary CSV", metadata_links.get("cycle_boundary_audit.csv")),
        ("product gate status", metadata_links.get("post_convergence_products_status.json")),
        ("models CSV", "metadata/models.csv"),
    ):
        if href:
            meta_links.append(f'<a href="{html.escape(str(href))}">{label}</a>')
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Kappa</title>
  <style>
    :root {{ color-scheme: dark; --bg:#050506; --panel:#101114; --panel2:#181a20; --text:#f7f1e4; --muted:#afa89c; --line:#333640; --red:#c1121f; --gold:#ffb703; --blue:#669bbc; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--text); font: 15px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }}
    header {{ padding: 32px clamp(18px, 4vw, 52px) 22px; background:#08090c; border-bottom: 1px solid var(--line); }}
    h1 {{ margin:0 0 8px; font-size: clamp(34px, 6vw, 72px); letter-spacing:0; line-height: 0.95; }}
    .dek {{ max-width: 900px; color: var(--muted); margin: 0 0 14px; }}
    .meta {{ color: var(--muted); margin: 0; }}
    main {{ padding: 24px clamp(18px, 4vw, 52px) 48px; }}
    .toolbar {{ display:flex; flex-wrap:wrap; gap:12px; margin:0 0 24px; color:var(--muted); }}
    a {{ color:#d9edf8; text-decoration:none; border-bottom:1px solid rgba(217,237,248,.38); }}
    a:hover {{ border-bottom-color:#d9edf8; }}
    .diagnostic {{ margin:0 0 24px; padding:16px; background:var(--panel); border:1px solid var(--line); border-radius:8px; }}
    .compact-diagnostic {{ max-width:1180px; }}
    .diagnostic h2 {{ margin:0 0 12px; font-size:24px; }}
    .diagnostic a {{ display:block; border:0; }}
    .diagnostic img {{ display:block; width:100%; height:auto; max-height:320px; object-fit:contain; border-radius:6px; background:#fff; }}
    .diagnostic-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(min(320px,100%),1fr)); gap:14px; align-items:start; }}
    .convergence-grid {{ grid-template-columns:repeat(auto-fit,minmax(min(360px,100%),1fr)); }}
    figure {{ margin:0; }}
    figcaption {{ margin-top:8px; color:var(--muted); }}
    .caption-link {{ display:inline; margin-left:8px; border-bottom:1px solid rgba(217,237,248,.38); }}
    .metric-row {{ margin:6px 0 0; color:#f7f1e4; font-size:13px; }}
    .grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(480px, 1fr)); gap:22px; align-items:start; }}
    .card {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; overflow:hidden; }}
    .card-head {{ display:flex; justify-content:space-between; gap:16px; align-items:start; padding:16px 18px 12px; background:var(--panel2); border-bottom:1px solid var(--line); }}
    h2 {{ margin:0; font-size:22px; }}
    .card-head p {{ margin:5px 0 0; color:var(--muted); font-size:13px; overflow-wrap:anywhere; }}
    .badge {{ border:1px solid currentColor; border-radius:7px; padding:4px 8px; white-space:nowrap; font-size:12px; }}
    .badge.ok {{ color:#83d18d; }}
    .badge.warn {{ color:var(--gold); }}
    .badge.bad {{ color:#f06a6a; }}
    .badge.muted {{ color:#7a7d88; }}
    .media {{ display:block; border-bottom:0; background:#000; }}
    .diagnostic-media {{ outline:2px solid rgba(255,183,3,.55); outline-offset:-2px; }}
    img {{ display:block; width:100%; height:auto; background:#000; }}
    .fourier-diagnostic {{ display:block; border:0; border-top:1px solid var(--line); background:#fff; }}
    .fourier-diagnostic img {{ background:#fff; }}
    .fourier-note {{ border-top:1px solid var(--line); padding:8px 18px; font-size:13px; background:#12151b; color:var(--muted); }}
    .fourier-note.ok {{ color:#83d18d; }}
    .fourier-note.warn {{ color:var(--gold); }}
    .fourier-placeholder {{ border-top:1px solid var(--line); padding:14px 18px; color:var(--muted); background:#101820; font-size:14px; }}
    .placeholder {{ min-height:260px; display:grid; place-items:center; color:#656977; background:repeating-linear-gradient(135deg,#07080b,#07080b 14px,#0d0f14 14px,#0d0f14 28px); }}
    .body {{ padding: 14px 18px 18px; }}
    .param-table {{ width:100%; border-collapse:collapse; margin:0 0 10px; font-size:13px; color:var(--text); }}
    .param-table th {{ text-align:left; color:var(--muted); font-weight:700; padding:4px 12px 4px 0; white-space:nowrap; vertical-align:top; }}
    .param-table td {{ padding:4px 14px 4px 0; white-space:nowrap; }}
    .param-table td span {{ display:block; color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:0; }}
    .details {{ margin:0 0 12px; color:var(--muted); }}
    .details span {{ margin-left:10px; color:var(--gold); }}
    .links {{ margin:0; display:flex; flex-wrap:wrap; gap:12px; }}
    .muted {{ color:#777b85; }}
    @media (max-width:700px) {{ .grid {{ grid-template-columns:1fr; }} .param-table th {{ display:block; padding-top:6px; }} .param-table tr {{ display:block; }} .param-table td {{ display:inline-block; min-width:88px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Kappa</h1>
    <p class="dek">RR Lyrae RSP animations for following pressure-volume work, gas heating, ionization structure, radius, temperature, luminosity, and radial velocity through pulsation phase.</p>
    <p class="meta">Generated {html.escape(generated)}. Trusted GIFs: {trusted}/{len(models)}. Seam-verified: {verified}/{len(models)}.</p>
  </header>
  <main>
    <nav class="toolbar">{" ".join(meta_links)}</nav>
{convergence_figure}
{convergence_trends_figure}
{convergence_trends_exact_figure}
{convergence_forecast_figure}
{growth_diagnostic_figures}
    <section class="grid">
{cards}
    </section>
  </main>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html_text, encoding="utf-8", newline="\n")
    (output_dir / ".nojekyll").write_text("", encoding="utf-8", newline="\n")


def validate_static_site(output_dir: Path, models: list[dict[str, object]]) -> None:
    model_root = output_dir / "models"
    missing_dirs = [
        str(model.get("model_id"))
        for model in models
        if not (model_root / str(model.get("model_id"))).is_dir()
    ]
    if missing_dirs:
        raise RuntimeError(f"Static site build missing model directories: {', '.join(missing_dirs)}")

    missing_assets: list[str] = []
    for model in models:
        assets = model.get("assets")
        if not isinstance(assets, dict):
            continue
        for rel_path in assets.values():
            if rel_path and not (output_dir / str(rel_path)).exists():
                missing_assets.append(str(rel_path))
    if missing_assets:
        shown = ", ".join(missing_assets[:12])
        suffix = "" if len(missing_assets) <= 12 else f", ... and {len(missing_assets) - 12} more"
        raise RuntimeError(f"Static site build missing copied assets: {shown}{suffix}")


def build_site(rre_root: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest, live_by_id = discover_models(rre_root)
    convergence_by_id = convergence_by_model(rre_root)
    boundary_by_id = cycle_boundary_by_model(rre_root)
    varied_groups = varied_parameter_groups([record for record in manifest if isinstance(record, dict)])
    models = [
        copy_model_assets(
            rre_root,
            output_dir,
            record,
            live_by_id.get(str(record["model_id"])),
            convergence_by_id.get(str(record["model_id"])),
            boundary_by_id.get(str(record["model_id"])),
        )
        for record in manifest
        if isinstance(record, dict)
    ]
    manifest_by_id = {
        str(record["model_id"]): record
        for record in manifest
        if isinstance(record, dict) and record.get("model_id") is not None
    }
    for model in models:
        record = manifest_by_id.get(str(model.get("model_id")))
        if record is not None:
            model["parameter_groups"] = parameter_groups_for_record(record, varied_groups)
    metadata_links = copy_batch_assets(rre_root, output_dir)
    write_manifest_csv(output_dir, models)
    metadata_links.update(write_fourier_inventory(output_dir, models))
    metadata_links.update(write_analysis_completion_status(output_dir, models))
    write_index(output_dir, models, metadata_links)
    validate_static_site(output_dir, models)


def replace_output_dir(staged_dir: Path, output_dir: Path) -> None:
    previous_dir = output_dir.with_name(f".{output_dir.name}.previous")
    if previous_dir.exists():
        shutil.rmtree(previous_dir)
    if output_dir.exists():
        output_dir.replace(previous_dir)
    staged_dir.replace(output_dir)
    if previous_dir.exists():
        shutil.rmtree(previous_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the portable Kappa static app from local RSP batch outputs.")
    parser.add_argument("--rre-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rre_root = args.rre_root.resolve()
    output_dir = args.output.resolve()
    staged_dir = output_dir.with_name(f".{output_dir.name}.tmp")
    if staged_dir.exists():
        shutil.rmtree(staged_dir)
    try:
        build_site(rre_root, staged_dir)
        replace_output_dir(staged_dir, output_dir)
    except Exception:
        if staged_dir.exists():
            shutil.rmtree(staged_dir)
        raise
    print(output_dir)


if __name__ == "__main__":
    main()
