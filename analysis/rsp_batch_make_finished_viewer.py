from __future__ import annotations

import argparse
import html
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_WORKSPACE = ROOT / "rsp_batch_runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a finished-only local viewer for verified RSP batch animations."
    )
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def read_json(path: Path) -> dict | list:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def fortran_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).replace("d", "e").replace("D", "e"))
    except ValueError:
        return None


def fmt(value: object, digits: int = 4) -> str:
    number = fortran_float(value)
    if number is None:
        return html.escape(str(value))
    return f"{number:.{digits}g}"


def href(path: Path | None, base: Path) -> str:
    if path is None:
        return ""
    path = path.resolve()
    base = base.resolve()
    try:
        return html.escape(path.relative_to(base).as_posix())
    except ValueError:
        return html.escape(path.as_uri())


def file_path(container: dict, key: str) -> Path | None:
    item = container.get(key)
    if isinstance(item, dict) and item.get("exists") and item.get("path"):
        return Path(str(item["path"]))
    return None


def model_gif_path(model: dict) -> Path | None:
    if model.get("registered_existing"):
        return file_path(model.get("checks", {}), "gif")
    return file_path(model.get("files", {}), "gif")


def model_png_path(model: dict) -> Path | None:
    return file_path(model.get("files", {}), "png")


def model_summary_path(model: dict) -> Path | None:
    return file_path(model.get("files", {}), "animation_summary")


def model_lightcurve_path(model: dict) -> Path | None:
    return file_path(model.get("files", {}), "final_cycle_lightcurve")


def model_verification_path(model: dict) -> Path | None:
    return file_path(model.get("files", {}), "verification_summary")


def link(label: str, path: Path | None, base: Path) -> str:
    if path is None:
        return f'<span class="muted">{html.escape(label)}</span>'
    return f'<a href="{href(path, base)}">{html.escape(label)}</a>'


def modulation_label(modulation: dict) -> str | None:
    if not modulation:
        return None
    try:
        max_l = float(modulation.get("max_l_modulation_fraction"))
        min_v = float(modulation.get("min_v_modulation_mag"))
    except (TypeError, ValueError):
        return None
    return f"max L mod {max_l:.3g}, min V mod {min_v:.3g} mag"


def modulation_png_path(modulation: dict) -> Path | None:
    value = modulation.get("diagnostic_png") if isinstance(modulation, dict) else None
    if not value:
        return None
    path = Path(str(value))
    return path if path.exists() else None


def phase_seam_label(model: dict) -> str | None:
    if model.get("phase_seam_ok") is True:
        seam = (
            model.get("phase_seam", {})
            .get("metrics", {})
            .get("luminosity_lsun", {})
            .get("fraction_of_amplitude")
        )
        if isinstance(seam, (float, int)):
            return f"L seam {seam:.2e}"
        return "phase seam ok"
    if model.get("phase_seam_ok") is False:
        return "phase seam failed"
    return None


def build_card(model: dict, manifest: dict, modulation: dict, base: Path) -> str:
    model_id = str(model.get("model_id", "model"))
    run_name = str(manifest.get("run_name", model.get("run_name", model_id)))
    gif = model_gif_path(model)
    png = model_png_path(model)
    summary = model_summary_path(model)
    lightcurve = model_lightcurve_path(model)
    verification = model_verification_path(model)
    registered = bool(model.get("registered_existing"))
    verified = bool(model.get("verification_passed")) or registered
    quality_warnings = model.get("quality_warnings", [])
    if registered:
        badge = "registered baseline"
    elif quality_warnings:
        badge = "quality warning"
    else:
        badge = "verified"
    profile_count = model.get("profile_count")
    mode_bits = []
    if model.get("pressure_work_mode"):
        mode_bits.append(str(model["pressure_work_mode"]))
    if model.get("heating_mode"):
        mode_bits.append(str(model["heating_mode"]))
    if model.get("saturated_by_grekm") is True:
        mode_bits.append("GREKM saturated")
    elif model.get("reached_max_periods") is True:
        mode_bits.append("period cap")
    if profile_count is not None:
        mode_bits.append(f"{profile_count} profiles")
    if model.get("radius_window_contains_photosphere") is True:
        mode_bits.append("radius window ok")
    seam_text = phase_seam_label(model)
    if seam_text:
        mode_bits.append(seam_text)
    modulation_text = modulation_label(modulation)
    if modulation_text:
        mode_bits.append(modulation_text)
    if quality_warnings:
        mode_bits.append("not a clean limit cycle")
    details = " | ".join(mode_bits) if mode_bits else "existing completed product"
    params = (
        f"M={fmt(manifest.get('RSP_mass'), 4)} Msun, "
        f"Teff={fmt(manifest.get('RSP_Teff'), 5)} K, "
        f"L={fmt(manifest.get('RSP_L'), 4)} Lsun, "
        f"Z={fmt(manifest.get('RSP_Z'), 4)}"
    )
    links = " ".join(
        [
            link("GIF", gif, base),
            link("PNG", png, base),
            link("summary", summary, base),
            link("lightcurve", lightcurve, base),
            link("verify", verification, base),
            link("cycle diagnostic", modulation_png_path(modulation), base),
        ]
    )
    img = ""
    if gif is not None:
        img = f'<a class="gif-link" href="{href(gif, base)}"><img src="{href(gif, base)}" alt="{html.escape(model_id)} animation"></a>'
    else:
        img = '<div class="placeholder">No GIF yet</div>'
    card_class = "card quality-warning" if quality_warnings else "card verified" if verified else "card"
    return f"""
    <article class="{card_class}">
      <div class="card-head">
        <div>
          <h2>{html.escape(model_id)}</h2>
          <p>{html.escape(run_name)}</p>
        </div>
        <span class="badge">{html.escape(badge)}</span>
      </div>
      {img}
      <p class="params">{html.escape(params)}</p>
      <p class="details">{html.escape(details)}</p>
      <p class="links">{links}</p>
    </article>
    """


def build_html(workspace: Path, output_path: Path) -> str:
    output_root = workspace / "output"
    audit_path = output_root / "batch_audit_summary.json"
    live_path = output_root / "live_status.json"
    modulation_path = output_root / "cycle_modulation_summary.json"
    quality_extension_status_path = output_root / "quality_extension_status.json"
    quality_extension_log_path = output_root / "quality_extension.log"
    manifest_path = workspace / "inputs" / "manifest.json"
    audit = read_json(audit_path)
    live = read_json(live_path)
    modulation_data = read_json(modulation_path)
    manifest_list = read_json(manifest_path)
    manifest_by_model = {
        str(record.get("model_id")): record
        for record in manifest_list
        if isinstance(record, dict) and record.get("model_id")
    }
    models = audit.get("models", []) if isinstance(audit, dict) else []
    modulation_models = modulation_data.get("models", []) if isinstance(modulation_data, dict) else []
    modulation_by_model = {
        str(record.get("model_id")): record
        for record in modulation_models
        if isinstance(record, dict) and record.get("model_id")
    }
    finished = [
        model
        for model in models
        if isinstance(model, dict)
        and model.get("status") in {"complete", "quality_warning"}
        and model_gif_path(model) is not None
    ]
    finished.sort(key=lambda item: str(item.get("model_id", "")))
    generated = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    live_models = live.get("models", []) if isinstance(live, dict) else []
    active = next((m for m in live_models if isinstance(m, dict) and m.get("active_stage")), None)
    active_text = "No active model reported."
    if active:
        stage = active.get("active_stage")
        period = active.get("latest_period")
        max_periods = active.get("max_periods")
        eta = active.get("estimated_stage_eta")
        bits = [str(active.get("model_id")), str(stage)]
        if period and max_periods:
            bits.append(f"period {period}/{max_periods}")
        if eta:
            bits.append(f"ETA {eta}")
        active_text = " | ".join(bits)

    cards = "\n".join(
        build_card(
            model,
            manifest_by_model.get(str(model.get("model_id")), {}),
            modulation_by_model.get(str(model.get("model_id")), {}),
            output_path.parent,
        )
        for model in finished
    )
    if not cards:
        cards = '<section class="empty">No finished GIFs found yet.</section>'

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Finished RSP Animation Viewer</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #050505;
      --panel: #101010;
      --panel-2: #181818;
      --text: #f4f1ea;
      --muted: #b8b2a8;
      --line: #3b3936;
      --gold: #ffb703;
      --red: #fb8500;
      --blue: #669bbc;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: "Segoe UI", Arial, sans-serif;
      line-height: 1.45;
    }}
    header {{
      padding: 28px 32px 16px;
      border-bottom: 1px solid var(--line);
      background: #080808;
      position: sticky;
      top: 0;
      z-index: 2;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
      font-weight: 700;
    }}
    header p {{
      margin: 4px 0;
      color: var(--muted);
    }}
    main {{
      padding: 24px 32px 40px;
    }}
    .tools {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 14px;
    }}
    a {{
      color: #d9edf8;
      text-decoration: none;
      border-bottom: 1px solid rgba(217,237,248,0.4);
    }}
    a:hover {{
      border-bottom-color: #d9edf8;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(520px, 1fr));
      gap: 22px;
      align-items: start;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 10px 32px rgba(0,0,0,0.35);
    }}
    .card-head {{
      display: flex;
      align-items: start;
      justify-content: space-between;
      gap: 16px;
      padding: 16px 18px 12px;
      background: var(--panel-2);
      border-bottom: 1px solid var(--line);
    }}
    h2 {{
      margin: 0;
      font-size: 22px;
      line-height: 1.1;
    }}
    .card-head p {{
      margin: 5px 0 0;
      color: var(--muted);
      font-size: 13px;
      overflow-wrap: anywhere;
    }}
    .badge {{
      border: 1px solid var(--blue);
      color: var(--text);
      border-radius: 8px;
      padding: 4px 8px;
      white-space: nowrap;
      font-size: 12px;
      background: rgba(102,155,188,0.15);
    }}
    .quality-warning .badge {{
      border-color: var(--red);
      background: rgba(251,133,0,0.16);
    }}
    .quality-warning {{
      border-color: rgba(251,133,0,0.55);
    }}
    .gif-link {{
      display: block;
      border: 0;
      background: #000;
    }}
    img {{
      display: block;
      width: 100%;
      height: auto;
      background: #000;
    }}
    .params,
    .details,
    .links {{
      margin: 10px 18px;
    }}
    .params {{
      color: var(--text);
      font-weight: 600;
    }}
    .details {{
      color: var(--muted);
    }}
    .links {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      padding-bottom: 10px;
    }}
    .muted {{
      color: #6f6a63;
    }}
    .empty {{
      padding: 32px;
      border: 1px solid var(--line);
      border-radius: 8px;
      color: var(--muted);
      background: var(--panel);
    }}
    @media (max-width: 720px) {{
      header, main {{ padding-left: 16px; padding-right: 16px; }}
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Finished RSP Animation Viewer</h1>
    <p>{len(finished)} completed animations embedded here. Generated {html.escape(generated)}.</p>
    <p>Active batch: {html.escape(active_text)}</p>
    <div class="tools">
      <a href="{href(output_root / 'index.html', output_path.parent)}">live batch dashboard</a>
      <a href="{href(audit_path, output_path.parent)}">batch audit JSON</a>
      <a href="{href(live_path, output_path.parent)}">live status JSON</a>
      <a href="{href(quality_extension_status_path, output_path.parent)}">quality extension status</a>
      <a href="{href(quality_extension_log_path, output_path.parent)}">quality extension log</a>
    </div>
  </header>
  <main>
    <section class="grid">
      {cards}
    </section>
  </main>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    output_path = args.output or workspace / "output" / "finished_visualizer.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_html(workspace, output_path), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
