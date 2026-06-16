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
        sanitized = value.replace(root_text, "<local-rre-root>").replace(root_forward, "<local-rre-root>")
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


def fmt_float(value: object, digits: int = 4) -> str:
    number = parse_manifest_number(value)
    if number is None:
        return "..."
    return f"{number:.{digits}g}"


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


def copy_model_assets(
    rre_root: Path,
    output_dir: Path,
    record: dict[str, object],
    live_record: dict[str, object] | None,
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
        if key in {"gif", "png"} and not verified:
            copied[key] = None
            continue
        if source.suffix.lower() == ".json":
            copied[key] = copy_json_if_exists(source, asset_dir / source.name, output_dir, rre_root)
        elif source.suffix.lower() in {".csv", ".txt"}:
            copied[key] = copy_text_if_exists(source, asset_dir / source.name, output_dir, rre_root)
        else:
            copied[key] = copy_if_exists(source, asset_dir / source.name, output_dir)

    summary = load_json(source_map["summary"])
    status = "pending"
    if verified:
        status = "verified"
    elif verification_failed:
        status = "verification failed"
    if live_record and live_record.get("active_stage"):
        status = f"running: {live_record.get('active_stage')}"
    if not source_map["gif"].exists() and not source_map["png"].exists():
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
        "gif_mb": file_size_mb(source_map["gif"]) if verified else None,
        "profile_count": live_record.get("profile_count") if live_record else None,
        "latest_period": live_record.get("latest_period") if live_record else None,
        "max_periods": live_record.get("max_periods") if live_record else None,
        "verification_passed": live_record.get("verification_passed") if live_record else None,
        "verification_failures": verify.get("failures", []) if isinstance(verify, dict) else [],
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
        copy_if_exists(diagnostic, diagnostics_dir / diagnostic.name, output_dir)
    for diagnostic in batch_source_dir.glob("*diagnostic*.png"):
        copy_if_exists(diagnostic, diagnostics_dir / diagnostic.name, output_dir)
    return copied


def write_manifest_csv(output_dir: Path, models: list[dict[str, object]]) -> None:
    path = output_dir / "metadata" / "models.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model_id", "run_name", "status", "M", "Teff", "L", "Z", "gif_mb"])
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
        image_html = f'<a class="media" href="{html.escape(str(image))}"><img src="{html.escape(str(image))}" alt="{html.escape(str(model["model_id"]))} animation"></a>'
    else:
        placeholder = "verification failed" if "failed" in str(model["status"]) else "queued"
        image_html = f'<div class="media placeholder">{html.escape(placeholder)}</div>'
    progress_bits = []
    if model.get("profile_count"):
        progress_bits.append(f"{model['profile_count']} profiles")
    if model.get("latest_period") and model.get("max_periods"):
        progress_bits.append(f"period {model['latest_period']} / {model['max_periods']}")
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
    completed = sum(1 for model in models if model["assets"].get("gif"))
    verified = sum(1 for model in models if model.get("verification_passed") is True or model["status"] == "verified")
    cards = "\n".join(card_html(model) for model in models)
    meta_links = []
    for label, href in (
        ("live status", metadata_links.get("live_status.json")),
        ("audit", metadata_links.get("batch_audit_summary.json")),
        ("manifest", metadata_links.get("manifest.json")),
        ("cycle modulation", metadata_links.get("cycle_modulation_summary.json")),
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
    <p class="meta">Generated {html.escape(generated)}. Completed GIFs: {completed}/{len(models)}. Verified: {verified}/{len(models)}.</p>
  </header>
  <main>
    <nav class="toolbar">{" ".join(meta_links)}</nav>
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
    models = [
        copy_model_assets(rre_root, output_dir, record, live_by_id.get(str(record["model_id"])))
        for record in manifest
        if isinstance(record, dict)
    ]
    metadata_links = copy_batch_assets(rre_root, output_dir)
    write_manifest_csv(output_dir, models)
    write_index(output_dir, models, metadata_links)
    print(output_dir)


if __name__ == "__main__":
    main()
