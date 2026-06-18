from __future__ import annotations

import argparse
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_WORKSPACE = ROOT / "rsp_batch_runs"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a local HTML gallery for RSP batch animation products.")
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--refresh-seconds", type=int, default=180)
    return parser.parse_args()


def read_json(path: Path) -> dict | list:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def parse_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).replace("d", "e").replace("D", "e"))
    except ValueError:
        return None


def normalized_trend_row(row: dict) -> dict:
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
        "has_full_window": row.get("window_cycles") is not None,
        "converged_gamma": row.get("converged_gamma"),
        "converged_period": row.get("converged_period"),
        "converged_delta_r": row.get("converged_delta_r"),
        "converged_exact": row.get("converged_exact"),
        "limit_cycle_converged": row.get("converged_exact"),
        "display_source": "convergence_trends_last100.latest_by_model",
    }


def convergence_window_end(row: dict) -> float | None:
    return parse_float(row.get("window_end_period_number") or row.get("last_period_number") or row.get("cycle_count"))


def merge_fresher_trend_rows(convergence_by_model: dict[str, dict], trends_data: dict | list) -> dict[str, dict]:
    merged = dict(convergence_by_model)
    trend_rows = trends_data.get("latest_by_model", {}) if isinstance(trends_data, dict) else {}
    if not isinstance(trend_rows, dict):
        return merged
    for model_id, row in trend_rows.items():
        if not isinstance(row, dict):
            continue
        normalized = normalized_trend_row(row)
        trend_end = convergence_window_end(normalized)
        current_end = convergence_window_end(merged.get(str(model_id), {}))
        if trend_end is not None and (current_end is None or trend_end > current_end):
            merged[str(model_id)] = normalized
    return merged


def read_active_batch_status(output_root: Path) -> tuple[dict | list, Path | None]:
    candidates: list[tuple[float, Path, dict | list]] = []
    for path in output_root.glob("batch*_status.json"):
        data = read_json(path)
        if isinstance(data, dict):
            candidates.append((path.stat().st_mtime, path, data))
    if not candidates:
        return {}, None
    running = [item for item in candidates if item[2].get("status") == "running"]
    selected = max(running, key=lambda item: item[0]) if running else max(candidates, key=lambda item: item[0])
    return selected[2], selected[1]


def parse_fortran_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).replace("d", "e").replace("D", "e"))
    except ValueError:
        return None


def parse_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(float(str(value).replace("d", "e").replace("D", "e")))
    except ValueError:
        return None


def parse_iso_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()
    return parsed


def fmt_float(value: object, digits: int = 4) -> str:
    number = parse_fortran_float(value)
    if number is None:
        return html.escape(str(value))
    return f"{number:.{digits}g}"


def rel_or_uri(path: Path, base: Path) -> str:
    path = path.resolve()
    base = base.resolve()
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_uri()


def stage_status(output_dir: Path) -> tuple[str, dict]:
    status_path = output_dir / "run_status.json"
    status = read_json(status_path)
    if not isinstance(status, dict) or not status:
        return "registered" if output_dir.exists() else "pending", {}
    stages = status.get("stages", {})
    retryable_failures = []
    for name, data in stages.items():
        if not isinstance(data, dict) or data.get("status") != "failed":
            continue
        expected_output = data.get("expected_output")
        if expected_output and not Path(str(expected_output)).exists():
            retryable_failures.append(str(name))
    if retryable_failures:
        return f"queued retry: {retryable_failures[0]}", status
    if any(data.get("status") == "skipped_pending_convergence" for data in stages.values() if isinstance(data, dict)):
        return "awaiting convergence", status
    if stages.get("verify", {}).get("status") == "complete":
        return "verified", status
    for stage in ("verify", "plot", "final_cycle", "deep2cycles", "restart", "continue_saturation", "create", "prepared"):
        stage_value = stages.get(stage, {}).get("status")
        if stage_value == "failed":
            return f"failed: {stage}", status
        if stage_value == "running":
            return f"running: {stage}", status
    completed = [name for name, data in stages.items() if data.get("status") == "complete"]
    if completed:
        return f"complete through {completed[-1]}", status
    return "pending", status


def read_rsp_max_num_periods(run_dir: Path, log_name: str | None = None) -> str | None:
    preferred = {
        "create.log": "inlist_create",
        "continue_saturation.log": "inlist_continue_saturation",
        "restart.log": "inlist_restart",
        "deep2cycles.log": "inlist_deep2cycles",
    }.get(log_name or "")
    inlist_names = [
        name
        for name in (
            preferred,
            "inlist_create",
            "inlist_continue_saturation",
            "inlist_restart",
            "inlist_deep2cycles",
        )
        if name is not None
    ]
    for inlist_name in inlist_names:
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


def completed_progress(products: dict[str, Path | None]) -> str:
    verification = products.get("verification")
    if verification is not None and verification.exists():
        data = read_json(verification)
        if isinstance(data, dict) and data.get("passed") is True:
            bits = ["animation verified"]
            profile_count = data.get("profile_count")
            if profile_count is not None:
                bits.append(f"{profile_count} profiles")
            pressure_mode = data.get("pressure_work_mode")
            heating_mode = data.get("heating_mode")
            if pressure_mode and heating_mode:
                bits.append(f"{pressure_mode}, {heating_mode}")
            return "; ".join(bits)
    summary = products.get("summary")
    if summary is not None and summary.exists():
        data = read_json(summary)
        if isinstance(data, dict) and data.get("gif_path"):
            return "animation built"
    return ""


def active_stage_started_at(status: dict) -> str | None:
    stages = status.get("stages", {}) if isinstance(status, dict) else {}
    if not isinstance(stages, dict):
        return None
    for stage in ("create", "continue_saturation", "restart", "deep2cycles"):
        data = stages.get(stage)
        if isinstance(data, dict) and data.get("status") == "running":
            return data.get("started_at") or data.get("updated_at")
    return None


def period_progress_text(period: str, max_periods: str | None, started_at: str | None) -> str:
    if not max_periods:
        return f"period {period}"
    text = f"period {period}/{max_periods}"
    period_num = parse_int(period)
    max_num = parse_int(max_periods)
    if period_num is not None and max_num is not None and max_num > 0:
        text += f" ({100.0 * period_num / max_num:.1f}%)"
        started = parse_iso_datetime(started_at)
        if started is not None and period_num > 0:
            now = datetime.now(started.tzinfo)
            elapsed = max(0.0, (now - started).total_seconds())
            remaining = max(0.0, elapsed * max_num / period_num - elapsed)
            eta = (now.timestamp() + remaining)
            eta_text = datetime.fromtimestamp(eta).strftime("%H:%M")
            text += f", rough ETA {eta_text}"
    return text


def latest_history_progress(record: dict, products: dict[str, Path | None], status_text: str, status: dict) -> str:
    if status_text == "verified":
        return completed_progress(products)

    run_dir = Path(str(record["run_dir"]))
    output_dir = Path(str(record["output_dir"]))
    if not run_dir.exists():
        return ""

    history_files = sorted(run_dir.glob("LOGS*/history.data"), key=lambda path: path.stat().st_mtime, reverse=True)
    parts: list[str] = []
    log_files = sorted((output_dir / "logs").glob("*.log"), key=lambda path: path.stat().st_mtime, reverse=True) if output_dir.exists() else []
    for log_file in log_files:
        try:
            log_lines = log_file.read_text(errors="ignore").splitlines()
        except OSError:
            continue
        for line in reversed(log_lines):
            match = re.search(r"^\s*period\s+(\d+)\b", line)
            if match:
                max_periods = read_rsp_max_num_periods(run_dir, log_file.name)
                parts.append(period_progress_text(match.group(1), max_periods, active_stage_started_at(status)))
                break
        if parts:
            break

    if history_files:
        history = history_files[0]
        last_model = None
        try:
            lines = history.read_text(errors="ignore").splitlines()
        except OSError:
            lines = []
        for line in reversed(lines):
            fields = line.split()
            if fields and fields[0].lstrip("+-").isdigit():
                last_model = fields[0]
                break
        if last_model is not None:
            parts.append(f"history model {last_model}")
        parts.append(f"updated {datetime.fromtimestamp(history.stat().st_mtime).strftime('%Y-%m-%d %H:%M')}")

    photo_dirs = sorted(run_dir.glob("photos*"), key=lambda path: path.stat().st_mtime if path.exists() else 0, reverse=True)
    latest_photo = None
    for photo_dir in photo_dirs:
        if not photo_dir.is_dir():
            continue
        numeric_photos = [path for path in photo_dir.iterdir() if path.is_file() and path.name.isdigit()]
        if numeric_photos:
            latest_photo = max(numeric_photos, key=lambda path: int(path.name))
            break
    if latest_photo is not None:
        parts.append(f"latest photo {latest_photo.parent.name}/{latest_photo.name}")

    return "; ".join(parts)


def find_products(record: dict, output_root: Path) -> dict[str, Path | None]:
    output_dir = Path(str(record["output_dir"]))
    product_stem = str(record["product_stem"])
    prefix = str(record["prefix"])
    exact = {
        "gif": output_dir / f"{product_stem}.gif",
        "png": output_dir / f"{product_stem}.png",
        "summary": output_dir / f"{product_stem}_summary.json",
        "final_cycle": output_dir / f"{prefix}_final_cycle_summary.json",
        "lightcurve": output_dir / f"{prefix}_final_cycle_lightcurve.csv",
        "verification": output_dir / "verification_summary.json",
        "run_status": output_dir / "run_status.json",
    }
    for key, path in list(exact.items()):
        if not path.exists():
            matches = sorted(output_dir.glob(f"*{key}*")) if output_dir.exists() else []
            exact[key] = matches[0] if matches else None
    return exact


def link(path: Path | None, label: str, base: Path) -> str:
    if path is None or not path.exists():
        return f'<span class="missing">{html.escape(label)}</span>'
    href = html.escape(rel_or_uri(path, base))
    return f'<a href="{href}">{html.escape(label)}</a>'


def verification_diagnostics(verification_path: Path | None) -> str:
    if verification_path is None or not verification_path.exists():
        return ""
    verification = read_json(verification_path)
    if not isinstance(verification, dict):
        return ""

    bits: list[str] = []
    pressure_mode = verification.get("pressure_work_mode")
    heating_mode = verification.get("heating_mode")
    if pressure_mode:
        bits.append(str(pressure_mode))
    if heating_mode:
        bits.append(str(heating_mode))
    if verification.get("saturated_by_grekm") is True:
        bits.append("GREKM saturated")
    elif verification.get("reached_max_periods") is True:
        bits.append("period cap")
    if verification.get("radius_window_contains_photosphere") is True:
        bits.append("radius window ok")
    elif verification.get("radius_window_contains_photosphere") is False:
        bits.append("radius window failed")
    if verification.get("phase_seam_ok") is True:
        seam = (
            verification.get("phase_seam", {})
            .get("metrics", {})
            .get("luminosity_lsun", {})
            .get("fraction_of_amplitude")
        )
        if isinstance(seam, (float, int)):
            bits.append(f"L seam {seam:.2e}")
        else:
            bits.append("phase seam ok")
    elif verification.get("phase_seam_ok") is False:
        bits.append("phase seam failed")
    return " | ".join(bits)


def modulation_row_for(record: dict, modulation_by_model: dict[str, dict]) -> dict:
    return modulation_by_model.get(str(record.get("model_id")), {})


def convergence_row_for(record: dict, convergence_by_model: dict[str, dict]) -> dict:
    return convergence_by_model.get(str(record.get("model_id")), {})


def modulation_diagnostics(modulation: dict) -> str:
    if not modulation:
        return ""
    try:
        max_l = float(modulation.get("max_l_modulation_fraction"))
        min_v = float(modulation.get("min_v_modulation_mag"))
        period = float(modulation.get("period_modulation_fraction"))
    except (TypeError, ValueError):
        return ""
    bits = [f"max L mod {max_l:.3g}", f"min V mod {min_v:.3g} mag"]
    if max_l > 0.05 or min_v > 0.05 or period > 0.05:
        bits.append("not a clean limit cycle")
    return " | ".join(bits)


def convergence_diagnostics(convergence: dict) -> str:
    if not convergence:
        return ""
    if convergence.get("converged_exact") is True:
        return "strict convergence passed"
    source = convergence.get("source_kind") or "unknown source"
    cycles = convergence.get("cycle_count")
    gamma = convergence.get("gamma_peak_to_peak_last_window")
    period = convergence.get("period_fractional_peak_to_peak_last_window")
    delta_r = convergence.get("delta_r_fractional_peak_to_peak_last_window")
    bits = [f"strict convergence pending ({source}, cycles {cycles})"]
    if gamma is not None:
        bits.append(f"Gamma ptp {float(gamma):.3g}")
    else:
        bits.append("Gamma not recorded")
    if period is not None:
        bits.append(f"P frac {float(period):.3g}")
    if delta_r is not None:
        bits.append(f"DeltaR frac {float(delta_r):.3g}")
    return " | ".join(bits)


def convergence_progress(convergence: dict) -> str:
    if not convergence:
        return ""
    cycles = convergence.get("cycle_count")
    last_period = convergence.get("last_period_number")
    window = convergence.get("last_cycle_count_used")
    source = convergence.get("source_kind")
    bits: list[str] = []
    if cycles is not None:
        bits.append(f"saturation cycles {fmt_float(cycles, 5)}")
    if last_period is not None:
        bits.append(f"last saturation period {fmt_float(last_period, 5)}")
    if window is not None:
        bits.append(f"final-window cycles {fmt_float(window, 5)}")
    if source:
        bits.append(str(source))
    return "; ".join(bits)


def strict_convergence_passed(record: dict, convergence: dict) -> bool:
    if record.get("registered_existing"):
        return True
    return bool(convergence and convergence.get("converged_exact") is True)


def modulation_png(modulation: dict) -> Path | None:
    value = modulation.get("diagnostic_png") if isinstance(modulation, dict) else None
    if not value:
        return None
    path = Path(str(value))
    return path if path.exists() else None


def build_card(
    record: dict,
    output_root: Path,
    modulation_by_model: dict[str, dict],
    convergence_by_model: dict[str, dict],
) -> str:
    output_dir = Path(str(record["output_dir"]))
    status_text, status = stage_status(output_dir)
    products = find_products(record, output_root)
    modulation = modulation_row_for(record, modulation_by_model)
    convergence = convergence_row_for(record, convergence_by_model)
    model_id = str(record["model_id"])
    run_name = str(record["run_name"])
    is_verified = status_text == "verified"
    convergence_passed = strict_convergence_passed(record, convergence)
    trusted_animation = is_verified and convergence_passed
    if is_verified and not convergence_passed:
        status_text = "awaiting convergence"
        is_verified = False
    status_class = "ok" if is_verified else "warn" if not status_text.startswith("failed") else "bad"
    mass = fmt_float(record.get("RSP_mass"), 4)
    z = fmt_float(record.get("RSP_Z"), 4)
    teff = fmt_float(record.get("RSP_Teff"), 5)
    lum = fmt_float(record.get("RSP_L"), 5)
    if status_text == "awaiting convergence":
        progress = convergence_progress(convergence)
    else:
        progress = latest_history_progress(record, products, status_text, status)
    progress_html = f'<p class="progress">{html.escape(progress)}</p>' if progress else ""
    verification = read_json(products["verification"]) if products["verification"] is not None else {}
    verification_failed = isinstance(verification, dict) and verification.get("passed") is False
    diagnostics = verification_diagnostics(products["verification"])
    modulation_checks = modulation_diagnostics(modulation)
    if modulation_checks:
        diagnostics = " | ".join(bit for bit in (diagnostics, modulation_checks) if bit)
    convergence_checks = "" if record.get("registered_existing") else convergence_diagnostics(convergence)
    if convergence_checks:
        diagnostics = " | ".join(bit for bit in (diagnostics, convergence_checks) if bit)
    diagnostics_html = f'<p class="checks">{html.escape(diagnostics)}</p>' if diagnostics else ""

    if status_text == "awaiting convergence":
        placeholder_text = "Strict convergence pending"
    else:
        placeholder_text = "Verification failed" if verification_failed else "No animation yet"
    image_html = f'<div class="placeholder">{html.escape(placeholder_text)}</div>'
    gif = products["gif"]
    png = products["png"]
    if gif is not None and gif.exists() and trusted_animation:
        src_path = png if png is not None and png.exists() else gif
        src = html.escape(rel_or_uri(src_path, output_root))
        gif_src = html.escape(rel_or_uri(gif, output_root))
        image_html = (
            f'<img class="preview" src="{src}" data-gif="{gif_src}" data-poster="{src}" '
            f'alt="{html.escape(model_id)} work animation preview">'
            '<button class="play" type="button">Play GIF</button>'
        )

    links = " ".join(
        [
            link(products["gif"] if trusted_animation else None, "GIF", output_root),
            link(products["png"] if trusted_animation else None, "PNG", output_root),
            link(products["summary"], "animation summary", output_root),
            link(products["final_cycle"], "cycle summary", output_root),
            link(products["verification"], "verification", output_root),
            link(modulation_png(modulation), "cycle diagnostic", output_root),
            link(products["run_status"], "status", output_root),
            link(output_dir if output_dir.exists() else None, "folder", output_root),
        ]
    )

    return f"""
      <article class="card">
        <div class="media">{image_html}</div>
        <div class="card-body">
          <div class="row">
            <h2>{html.escape(model_id)}</h2>
            <span class="status {status_class}">{html.escape(status_text)}</span>
          </div>
          <p class="run-name">{html.escape(run_name)}</p>
          {progress_html}
          <dl>
            <div><dt>M</dt><dd>{mass} M_sun</dd></div>
            <div><dt>Z</dt><dd>{z}</dd></div>
            <div><dt>Teff</dt><dd>{teff} K</dd></div>
            <div><dt>L</dt><dd>{lum} L_sun</dd></div>
          </dl>
          {diagnostics_html}
          <p class="links">{links}</p>
        </div>
      </article>
    """


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    output_root = workspace / "output"
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = (args.output or output_root / "index.html").resolve()
    manifest_path = workspace / "inputs" / "manifest.json"
    manifest = read_json(manifest_path)
    if not isinstance(manifest, list):
        raise RuntimeError(f"Missing or invalid manifest: {manifest_path}")

    batch_status_path = output_root / "batch_remaining_006_009_status.json"
    batch_log_path = output_root / "batch_remaining_006_009.log"
    supervisor_status_path = output_root / "batch_supervisor_status.json"
    supervisor_log_path = output_root / "batch_supervisor.log"
    quality_extension_status_path = output_root / "quality_extension_status.json"
    quality_extension_log_path = output_root / "quality_extension.log"
    audit_path = output_root / "batch_audit_summary.json"
    live_status_path = output_root / "live_status.json"
    modulation_path = output_root / "cycle_modulation_summary.json"
    convergence_path = output_root / "convergence_summary_last100.json"
    convergence_png_path = output_root / "convergence_summary_last100.png"
    convergence_trends_path = output_root / "convergence_trends_last100.json"
    convergence_trends_csv_path = output_root / "convergence_trends_last100.csv"
    convergence_trends_png_path = output_root / "convergence_trends_last100.png"
    convergence_trends_exact_png_path = output_root / "convergence_trends_exact_last100.png"
    convergence_forecast_path = output_root / "convergence_forecast_last100.json"
    convergence_forecast_csv_path = output_root / "convergence_forecast_last100.csv"
    convergence_forecast_png_path = output_root / "convergence_forecast_last100.png"
    convergence_gate_audit_path = output_root / "convergence_gate_audit.json"
    convergence_gate_audit_csv_path = output_root / "convergence_gate_audit.csv"
    finished_viewer_path = output_root / "finished_visualizer.html"
    live_status = read_json(live_status_path)
    modulation_data = read_json(modulation_path)
    convergence_data = read_json(convergence_path)
    convergence_trends_data = read_json(convergence_trends_path)
    modulation_models = modulation_data.get("models", []) if isinstance(modulation_data, dict) else []
    modulation_by_model = {
        str(row.get("model_id")): row
        for row in modulation_models
        if isinstance(row, dict) and row.get("model_id")
    }
    convergence_models = convergence_data.get("models", []) if isinstance(convergence_data, dict) else []
    convergence_by_model = {
        str(row.get("model_id")): row
        for row in convergence_models
        if isinstance(row, dict) and row.get("model_id")
    }
    convergence_by_model = merge_fresher_trend_rows(convergence_by_model, convergence_trends_data)
    batch_status, selected_batch_status_path = read_active_batch_status(output_root)
    if selected_batch_status_path is not None:
        batch_status_path = selected_batch_status_path
        if isinstance(batch_status, dict) and batch_status.get("log"):
            batch_log_path = Path(str(batch_status["log"]))
    batch_text = "not launched"
    if isinstance(batch_status, dict) and batch_status:
        batch_text = str(batch_status.get("status", "unknown"))
        current = batch_status.get("current_model")
        if current:
            batch_text += f" ({current})"
    product_text = ""
    if isinstance(live_status, dict) and live_status:
        new_done = live_status.get("new_batch_gif_count")
        new_total = live_status.get("new_batch_model_count")
        baseline = live_status.get("registered_existing_gif_count")
        if new_done is not None and new_total is not None:
            product_text = f" Rendered batch GIF files: {new_done}/{new_total}."
            if baseline:
                product_text += f" Registered baseline GIFs: {baseline}."
            product_text += " Strict convergence controls previews."

    cards = "\n".join(build_card(row, output_root, modulation_by_model, convergence_by_model) for row in manifest)
    convergence_panel = ""
    if convergence_png_path.exists():
        convergence_panel = f"""
    <section class="diagnostic-panel">
      <h2>Limit-Cycle Convergence</h2>
      <a href="{html.escape(rel_or_uri(convergence_png_path, output_root))}">
        <img src="{html.escape(rel_or_uri(convergence_png_path, output_root))}" alt="Strict limit-cycle convergence summary">
      </a>
    </section>
"""
    convergence_trends_panel = ""
    if convergence_trends_png_path.exists():
        convergence_trends_panel = f"""
    <section class="diagnostic-panel">
      <h2>Rolling Convergence Trends</h2>
      <a href="{html.escape(rel_or_uri(convergence_trends_png_path, output_root))}">
        <img src="{html.escape(rel_or_uri(convergence_trends_png_path, output_root))}" alt="Rolling final-100-cycle convergence trends">
      </a>
    </section>
"""
    convergence_trends_exact_panel = ""
    if convergence_trends_exact_png_path.exists():
        convergence_trends_exact_panel = f"""
    <section class="diagnostic-panel">
      <h2>Rolling Convergence Trends: Exact-History Runs</h2>
      <a href="{html.escape(rel_or_uri(convergence_trends_exact_png_path, output_root))}">
        <img src="{html.escape(rel_or_uri(convergence_trends_exact_png_path, output_root))}" alt="Rolling final-100-cycle convergence trends for exact-history runs">
      </a>
    </section>
"""
    convergence_forecast_panel = ""
    if convergence_forecast_png_path.exists():
        convergence_forecast_panel = f"""
    <section class="diagnostic-panel">
      <h2>Convergence Forecast</h2>
      <a href="{html.escape(rel_or_uri(convergence_forecast_png_path, output_root))}">
        <img src="{html.escape(rel_or_uri(convergence_forecast_png_path, output_root))}" alt="Rolling convergence forecast">
      </a>
    </section>
"""
    generated = now_iso()
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="{args.refresh_seconds}">
  <title>RSP Batch Animation Gallery</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #050507;
      --panel: #101116;
      --panel-2: #171922;
      --text: #f5f1e8;
      --muted: #aaa497;
      --line: #30333e;
      --ok: #7bc87b;
      --warn: #ffb703;
      --bad: #f06a6a;
      --accent: #669bbc;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 15px/1.45 system-ui, -apple-system, Segoe UI, sans-serif;
    }}
    header {{
      padding: 28px clamp(18px, 4vw, 46px) 18px;
      border-bottom: 1px solid var(--line);
      background: #08090d;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: clamp(28px, 4vw, 44px);
      letter-spacing: 0;
      font-weight: 750;
    }}
    .meta {{
      margin: 0;
      color: var(--muted);
    }}
    main {{
      padding: 24px clamp(18px, 4vw, 46px) 46px;
    }}
    .batch {{
      margin: 0 0 24px;
      padding: 14px 16px;
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      color: var(--muted);
    }}
    .batch a, .links a {{
      color: var(--accent);
      text-decoration: none;
      margin-right: 12px;
      white-space: nowrap;
    }}
    .batch a:hover, .links a:hover {{ text-decoration: underline; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 20px;
    }}
    .diagnostic-panel {{
      margin: 0 0 24px;
      padding: 16px;
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
    }}
    .diagnostic-panel h2 {{
      margin: 0 0 12px;
      font-size: 22px;
    }}
    .diagnostic-panel a {{
      display: block;
      border: 0;
    }}
    .diagnostic-panel img {{
      display: block;
      width: 100%;
      height: auto;
      border-radius: 6px;
      background: #fff;
    }}
    .card {{
      display: grid;
      grid-template-columns: minmax(220px, 42%) 1fr;
      gap: 0;
      overflow: hidden;
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      min-height: 280px;
    }}
    .media {{
      position: relative;
      min-height: 260px;
      background: #000;
      border-right: 1px solid var(--line);
    }}
    .preview {{
      width: 100%;
      height: 100%;
      min-height: 260px;
      object-fit: contain;
      display: block;
      background: #000;
    }}
    .play {{
      position: absolute;
      left: 12px;
      bottom: 12px;
      border: 1px solid #565b6c;
      background: rgba(8, 9, 13, 0.86);
      color: var(--text);
      border-radius: 6px;
      padding: 7px 10px;
      font: inherit;
      cursor: pointer;
    }}
    .placeholder {{
      height: 100%;
      min-height: 260px;
      display: grid;
      place-items: center;
      color: var(--muted);
      background: repeating-linear-gradient(135deg, #090a0f, #090a0f 12px, #0e1018 12px, #0e1018 24px);
    }}
    .card-body {{ padding: 18px; }}
    .row {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 4px;
    }}
    h2 {{
      margin: 0;
      font-size: 22px;
      letter-spacing: 0;
    }}
    .status {{
      padding: 4px 8px;
      border-radius: 6px;
      border: 1px solid currentColor;
      white-space: nowrap;
      font-size: 13px;
    }}
    .status.ok {{ color: var(--ok); }}
    .status.warn {{ color: var(--warn); }}
    .status.bad {{ color: var(--bad); }}
    .run-name {{
      color: var(--muted);
      margin: 0 0 8px;
      overflow-wrap: anywhere;
    }}
    .progress {{
      margin: 0 0 14px;
      color: #d8cfbd;
      font-size: 13px;
      overflow-wrap: anywhere;
    }}
    .checks {{
      margin: -2px 0 14px;
      color: var(--muted);
      font-size: 13px;
      overflow-wrap: anywhere;
    }}
    dl {{
      display: grid;
      grid-template-columns: repeat(2, minmax(120px, 1fr));
      gap: 10px;
      margin: 0 0 14px;
    }}
    dl div {{
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
    }}
    dt {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0;
    }}
    dd {{
      margin: 2px 0 0;
      font-size: 15px;
    }}
    .links {{
      margin: 0;
      color: var(--muted);
    }}
    .missing {{
      color: #5f6370;
      margin-right: 12px;
      white-space: nowrap;
    }}
    @media (max-width: 760px) {{
      .grid {{ grid-template-columns: 1fr; }}
      .card {{ grid-template-columns: 1fr; }}
      .media {{ border-right: 0; border-bottom: 1px solid var(--line); }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>RSP Batch Animation Gallery</h1>
    <p class="meta">Generated {html.escape(generated)} from {html.escape(str(manifest_path))}</p>
  </header>
  <main>
    <p class="batch">
      Remaining batch: {html.escape(batch_text)}.{html.escape(product_text)}
      {link(finished_viewer_path if finished_viewer_path.exists() else None, "finished visualizer", output_root)}
      {link(batch_status_path if batch_status_path.exists() else None, "batch status", output_root)}
      {link(live_status_path if live_status_path.exists() else None, "live status", output_root)}
      {link(audit_path if audit_path.exists() else None, "batch audit", output_root)}
      {link(batch_log_path if batch_log_path.exists() else None, "batch log", output_root)}
      {link(supervisor_status_path if supervisor_status_path.exists() else None, "supervisor status", output_root)}
      {link(supervisor_log_path if supervisor_log_path.exists() else None, "supervisor log", output_root)}
      {link(quality_extension_status_path if quality_extension_status_path.exists() else None, "quality extension status", output_root)}
      {link(quality_extension_log_path if quality_extension_log_path.exists() else None, "quality extension log", output_root)}
      {link(convergence_path if convergence_path.exists() else None, "convergence JSON", output_root)}
      {link(convergence_png_path if convergence_png_path.exists() else None, "convergence plot", output_root)}
      {link(convergence_trends_path if convergence_trends_path.exists() else None, "convergence trends JSON", output_root)}
      {link(convergence_trends_csv_path if convergence_trends_csv_path.exists() else None, "convergence trends CSV", output_root)}
      {link(convergence_trends_png_path if convergence_trends_png_path.exists() else None, "convergence trends plot", output_root)}
      {link(convergence_trends_exact_png_path if convergence_trends_exact_png_path.exists() else None, "exact-history trend plot", output_root)}
      {link(convergence_forecast_path if convergence_forecast_path.exists() else None, "convergence forecast JSON", output_root)}
      {link(convergence_forecast_csv_path if convergence_forecast_csv_path.exists() else None, "convergence forecast CSV", output_root)}
      {link(convergence_forecast_png_path if convergence_forecast_png_path.exists() else None, "convergence forecast plot", output_root)}
      {link(convergence_gate_audit_path if convergence_gate_audit_path.exists() else None, "convergence gate audit JSON", output_root)}
      {link(convergence_gate_audit_csv_path if convergence_gate_audit_csv_path.exists() else None, "convergence gate audit CSV", output_root)}
      {link(manifest_path, "manifest", output_root)}
    </p>
{convergence_panel}
{convergence_trends_panel}
{convergence_trends_exact_panel}
{convergence_forecast_panel}
    <section class="grid">
{cards}
    </section>
  </main>
  <script>
    document.querySelectorAll(".play").forEach((button) => {{
      button.addEventListener("click", () => {{
        const img = button.previousElementSibling;
        const playing = img.dataset.playing === "true";
        img.src = playing ? img.dataset.poster : img.dataset.gif;
        img.dataset.playing = playing ? "false" : "true";
        button.textContent = playing ? "Play GIF" : "Show Poster";
      }});
    }});
  </script>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
