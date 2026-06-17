from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ANIMATION_SUFFIX = "_work_r_over_R_phase_cycle_dark_main_terms_gas_heating_pav_work"


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
    )
    return destination.relative_to(site_root).as_posix()


def sanitize_text_value(text: str, rre_root: Path) -> str:
    return (
        text.replace(str(rre_root), "<local-rre-root>")
        .replace(rre_root.as_posix(), "<local-rre-root>")
        .replace("\\", "/")
    )


def copy_text_if_exists(source: Path, destination: Path, site_root: Path, rre_root: Path) -> str | None:
    if not source.exists():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(sanitize_text_value(source.read_text(encoding="utf-8"), rre_root), encoding="utf-8")
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
    return by_model


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
    trusted_animation = verified and convergence_passed

    copied: dict[str, str | None] = {}
    source_map = {
        "gif": source_output_dir / f"{product_stem}.gif",
        "png": source_output_dir / f"{product_stem}.png",
        "summary": source_output_dir / f"{product_stem}_summary.json",
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
        "converged_exact": convergence.get("converged_exact") if convergence and not record.get("registered_existing") else None,
        "convergence": "" if record.get("registered_existing") else convergence_text(convergence),
        "verification_passed": live_record.get("verification_passed") if live_record else None,
        "verification_failures": verify.get("failures", []) if isinstance(verify, dict) else [],
        "animation_trusted": trusted_animation,
        "phase_curve_break_phases": phase_breaks,
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
        writer = csv.writer(handle)
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


def card_html(model: dict[str, object]) -> str:
    assets = model["assets"]
    assert isinstance(assets, dict)
    gif = assets.get("gif")
    png = assets.get("png")
    summary = assets.get("summary")
    verify = assets.get("verify")
    lightcurve = assets.get("lightcurve_csv")
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
    for label, href in (("GIF", gif), ("PNG", png), ("summary", summary), ("verify", verify), ("lightcurve", lightcurve)):
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
    if model.get("profile_count"):
        progress_bits.append(f"{model['profile_count']} profiles")
    status_text = str(model.get("status"))
    if model.get("latest_period") and model.get("max_periods") and "running" in status_text:
        progress_bits.append(f"period {model['latest_period']} / {model['max_periods']}")
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
    if image and not model.get("animation_trusted"):
        progress_bits.append("diagnostic animation, not trusted")
    if model.get("gif_mb") is not None:
        progress_bits.append(f"{float(model['gif_mb']):.1f} MB GIF")
    failures = model.get("verification_failures") or []
    if failures:
        progress_bits.append("; ".join(str(item) for item in failures[:2]))
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
        <div class="body">
          <p class="params">M={fmt_float(model.get("M"))} M_sun &nbsp; Teff={fmt_float(model.get("Teff"), 5)} K &nbsp; L={fmt_float(model.get("L"))} L_sun &nbsp; Z={fmt_float(model.get("Z"))}</p>
          <p class="details">{html.escape(" | ".join(progress_bits))} {break_text}</p>
          <p class="links">{" ".join(links)}</p>
        </div>
      </article>
    """


def write_index(output_dir: Path, models: list[dict[str, object]], metadata_links: dict[str, str | None]) -> None:
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    trusted = sum(1 for model in models if model.get("animation_trusted"))
    verified = sum(1 for model in models if model.get("verification_passed") is True or model["status"] == "verified")
    cards = "\n".join(card_html(model) for model in models)
    convergence_figure = ""
    convergence_png = metadata_links.get("convergence_summary_last100.png")
    if convergence_png:
        convergence_figure = f"""
    <section class="diagnostic">
      <h2>Limit-Cycle Convergence</h2>
      <a href="{html.escape(str(convergence_png))}"><img src="{html.escape(str(convergence_png))}" alt="Strict limit-cycle convergence summary"></a>
    </section>
"""
    convergence_trends_figure = ""
    convergence_trends_png = metadata_links.get("convergence_trends_last100.png")
    if convergence_trends_png:
        convergence_trends_figure = f"""
    <section class="diagnostic">
      <h2>Rolling Convergence Trends</h2>
      <a href="{html.escape(str(convergence_trends_png))}"><img src="{html.escape(str(convergence_trends_png))}" alt="Rolling final-100-cycle convergence trends"></a>
    </section>
"""
    convergence_trends_exact_figure = ""
    convergence_trends_exact_png = metadata_links.get("convergence_trends_exact_last100.png")
    if convergence_trends_exact_png:
        convergence_trends_exact_figure = f"""
    <section class="diagnostic">
      <h2>Rolling Convergence Trends: Exact-History Runs</h2>
      <a href="{html.escape(str(convergence_trends_exact_png))}"><img src="{html.escape(str(convergence_trends_exact_png))}" alt="Rolling final-100-cycle convergence trends for exact-history runs"></a>
    </section>
"""
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
        ("cycle modulation", metadata_links.get("cycle_modulation_summary.json")),
        ("convergence", metadata_links.get("convergence_summary_last100.json")),
        ("convergence plot", convergence_png),
        ("convergence trends", metadata_links.get("convergence_trends_last100.json")),
        ("convergence trend plot", convergence_trends_png),
        ("exact-history trend plot", convergence_trends_exact_png),
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
    .diagnostic h2 {{ margin:0 0 12px; font-size:24px; }}
    .diagnostic a {{ display:block; border:0; }}
    .diagnostic img {{ width:100%; height:auto; border-radius:6px; background:#fff; }}
    .diagnostic-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(420px,1fr)); gap:16px; }}
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
    .placeholder {{ min-height:260px; display:grid; place-items:center; color:#656977; background:repeating-linear-gradient(135deg,#07080b,#07080b 14px,#0d0f14 14px,#0d0f14 28px); }}
    .body {{ padding: 14px 18px 18px; }}
    .params {{ margin:0 0 8px; font-weight:650; }}
    .details {{ margin:0 0 12px; color:var(--muted); }}
    .details span {{ margin-left:10px; color:var(--gold); }}
    .links {{ margin:0; display:flex; flex-wrap:wrap; gap:12px; }}
    .muted {{ color:#777b85; }}
    @media (max-width:700px) {{ .grid {{ grid-template-columns:1fr; }} }}
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
{growth_diagnostic_figures}
    <section class="grid">
{cards}
    </section>
  </main>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")
    (output_dir / ".nojekyll").write_text("", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the portable Kappa static app from local RSP batch outputs.")
    parser.add_argument("--rre-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rre_root = args.rre_root.resolve()
    output_dir = args.output.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest, live_by_id = discover_models(rre_root)
    convergence_by_id = convergence_by_model(rre_root)
    models = [
        copy_model_assets(
            rre_root,
            output_dir,
            record,
            live_by_id.get(str(record["model_id"])),
            convergence_by_id.get(str(record["model_id"])),
        )
        for record in manifest
        if isinstance(record, dict)
    ]
    metadata_links = copy_batch_assets(rre_root, output_dir)
    write_manifest_csv(output_dir, models)
    write_index(output_dir, models, metadata_links)
    print(output_dir)


if __name__ == "__main__":
    main()
