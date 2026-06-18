from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rsp_batch_run import local_maxima, read_history


ROOT = Path(__file__).resolve().parent
DEFAULT_WORKSPACE = ROOT / "rsp_batch_runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure cycle-to-cycle modulation in RSP batch runs."
    )
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument(
        "--last-cycles",
        type=int,
        default=80,
        help="Number of latest radial cycles to show in each per-model diagnostic plot.",
    )
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


def history_candidates(run_dir: Path) -> list[tuple[str, Path]]:
    saturation_resume_histories = sorted(
        run_dir.glob("LOGS_saturation_resume_*/history.data"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    continue_resume_histories = sorted(
        run_dir.glob("LOGS_continue_saturation_resume_*/history.data"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    candidates = [(f"saturation_resume:{path.parent.name}", path) for path in saturation_resume_histories]
    candidates.extend((f"continue_saturation_resume:{path.parent.name}", path) for path in continue_resume_histories)
    candidates.extend(
        [
            ("continue_saturation", run_dir / "LOGS_continue_saturation" / "history.data"),
            ("deep", run_dir / "LOGS" / "history.data"),
            ("saturation", run_dir / "LOGS_saturation" / "history.data"),
        ]
    )
    seen: set[Path] = set()
    unique: list[tuple[str, Path]] = []
    for label, path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append((label, path))
    return unique


def cycle_table(history_path: Path) -> list[dict[str, float]]:
    history = read_history(history_path)
    required = ("star_age", "log_R", "log_L", "abs_mag_V", "log_Teff")
    missing = [name for name in required if name not in history]
    if missing:
        raise RuntimeError(f"{history_path} is missing columns: {', '.join(missing)}")

    age_days = np.asarray(history["star_age"], dtype=float) * 365.25
    radius = np.power(10.0, np.asarray(history["log_R"], dtype=float))
    luminosity = np.power(10.0, np.asarray(history["log_L"], dtype=float))
    mag_v = np.asarray(history["abs_mag_V"], dtype=float)
    teff = np.power(10.0, np.asarray(history["log_Teff"], dtype=float))

    maxima = local_maxima(radius.tolist())
    if len(radius) >= 2 and radius[-1] > radius[-2]:
        maxima.append(len(radius) - 1)
    cycles: list[dict[str, float]] = []
    for cycle_index, (start, end) in enumerate(zip(maxima[:-1], maxima[1:]), start=1):
        if end <= start:
            continue
        sl = slice(start, end + 1)
        phase_age = age_days[sl] - age_days[start]
        luminosity_cycle = luminosity[sl]
        radius_cycle = radius[sl]
        mag_cycle = mag_v[sl]
        teff_cycle = teff[sl]
        max_l_idx = int(np.nanargmax(luminosity_cycle))
        min_mag_idx = int(np.nanargmin(mag_cycle))
        cycles.append(
            {
                "cycle_number": float(cycle_index),
                "start_age_days": float(age_days[start]),
                "end_age_days": float(age_days[end]),
                "period_days": float(age_days[end] - age_days[start]),
                "radius_start_rsun": float(radius[start]),
                "radius_end_rsun": float(radius[end]),
                "radius_min_rsun": float(np.nanmin(radius_cycle)),
                "radius_max_rsun": float(np.nanmax(radius_cycle)),
                "radius_peak_to_peak_rsun": float(np.nanmax(radius_cycle) - np.nanmin(radius_cycle)),
                "max_l_lsun": float(np.nanmax(luminosity_cycle)),
                "min_l_lsun": float(np.nanmin(luminosity_cycle)),
                "luminosity_peak_to_peak_lsun": float(np.nanmax(luminosity_cycle) - np.nanmin(luminosity_cycle)),
                "min_abs_mag_v": float(np.nanmin(mag_cycle)),
                "max_abs_mag_v": float(np.nanmax(mag_cycle)),
                "teff_min_k": float(np.nanmin(teff_cycle)),
                "teff_max_k": float(np.nanmax(teff_cycle)),
                "max_l_phase_in_radial_cycle": float(phase_age[max_l_idx] / max(age_days[end] - age_days[start], 1e-99)),
                "min_mag_v_phase_in_radial_cycle": float(phase_age[min_mag_idx] / max(age_days[end] - age_days[start], 1e-99)),
            }
        )
    return cycles


def modulation_metrics(cycles: list[dict[str, float]], last_cycles: int) -> dict[str, float | int | None]:
    if not cycles:
        return {
            "cycle_count": 0,
            "last_cycle_count_used": 0,
            "max_l_modulation_fraction": None,
            "min_v_modulation_mag": None,
            "period_modulation_fraction": None,
        }
    tail = cycles[-int(last_cycles) :]
    max_l = np.asarray([cycle["max_l_lsun"] for cycle in tail], dtype=float)
    min_v = np.asarray([cycle["min_abs_mag_v"] for cycle in tail], dtype=float)
    period = np.asarray([cycle["period_days"] for cycle in tail], dtype=float)
    radius_amp = np.asarray([cycle["radius_peak_to_peak_rsun"] for cycle in tail], dtype=float)

    def frac_peak_to_peak(values: np.ndarray) -> float:
        scale = float(np.nanmedian(np.abs(values)))
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        return float((np.nanmax(values) - np.nanmin(values)) / scale)

    return {
        "cycle_count": len(cycles),
        "last_cycle_count_used": len(tail),
        "max_l_modulation_fraction": frac_peak_to_peak(max_l),
        "min_v_modulation_mag": float(np.nanmax(min_v) - np.nanmin(min_v)),
        "period_modulation_fraction": frac_peak_to_peak(period),
        "radius_amplitude_modulation_fraction": frac_peak_to_peak(radius_amp),
        "last_max_l_lsun": float(max_l[-1]),
        "last_min_abs_mag_v": float(min_v[-1]),
        "last_period_days": float(period[-1]),
    }


def choose_history_cycles(
    run_dir: Path,
    last_cycles: int,
) -> tuple[str, Path, list[dict[str, float]], list[dict[str, object]]] | None:
    usable: list[tuple[str, Path, list[dict[str, float]]]] = []
    diagnostics: list[dict[str, object]] = []
    for source, path in history_candidates(run_dir):
        if not path.exists():
            continue
        try:
            cycles = cycle_table(path)
        except Exception as exc:
            diagnostics.append(
                {
                    "history_source": source,
                    "history_path": str(path),
                    "cycle_count": None,
                    "error": repr(exc),
                }
            )
            continue
        diagnostics.append(
            {
                "history_source": source,
                "history_path": str(path),
                "cycle_count": len(cycles),
            }
        )
        usable.append((source, path, cycles))
    if not usable:
        return None

    for source, path, cycles in usable:
        if len(cycles) >= int(last_cycles):
            return source, path, cycles, diagnostics
    source, path, cycles = max(usable, key=lambda item: len(item[2]))
    return source, path, cycles, diagnostics


def deep_lightcurve_data(run_dir: Path) -> dict[str, np.ndarray] | None:
    path = run_dir / "LOGS" / "history.data"
    if not path.exists():
        return None
    try:
        history = read_history(path)
    except Exception:
        return None
    required = ("star_age", "log_R", "log_L", "abs_mag_V")
    if any(name not in history for name in required):
        return None
    age_days = np.asarray(history["star_age"], dtype=float) * 365.25
    radius = np.power(10.0, np.asarray(history["log_R"], dtype=float))
    luminosity = np.power(10.0, np.asarray(history["log_L"], dtype=float))
    mag_v = np.asarray(history["abs_mag_V"], dtype=float)
    minima = [
        idx
        for idx in range(1, len(mag_v) - 1)
        if mag_v[idx] < mag_v[idx - 1] and mag_v[idx] <= mag_v[idx + 1]
    ]
    return {
        "age_days": age_days,
        "radius_rsun": radius,
        "luminosity_lsun": luminosity,
        "abs_mag_v": mag_v,
        "radius_maxima": np.asarray(local_maxima(radius.tolist()), dtype=int),
        "v_minima": np.asarray(minima, dtype=int),
    }


def plot_model_diagnostic(
    record: dict[str, object],
    cycles: list[dict[str, float]],
    history_source: str,
    history_path: Path,
    output_dir: Path,
    last_cycles: int,
) -> Path:
    model_id = str(record["model_id"])
    run_name = str(record["run_name"])
    path = output_dir / f"{run_name}_cycle_modulation_diagnostic.png"
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)

    deep_data = deep_lightcurve_data(Path(str(record["run_dir"])))
    ax = axes[0]
    if deep_data is None:
        ax.text(0.5, 0.5, "No deep history yet", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        age = deep_data["age_days"]
        luminosity = deep_data["luminosity_lsun"]
        t0 = float(age[0])
        ax.plot(age - t0, luminosity, color="#FBBF24", lw=1.35, label="photosphere L")
        for idx in deep_data["radius_maxima"]:
            ax.axvline(float(age[int(idx)] - t0), color="0.35", lw=0.85, ls="--", zorder=0)
        v_minima = deep_data["v_minima"]
        if v_minima.size:
            ax.scatter(
                age[v_minima] - t0,
                luminosity[v_minima],
                s=14,
                color="#C1121F",
                zorder=3,
                label="V minima",
            )
        ax.set_xlabel("days since first deep profile")
        ax.set_ylabel("L [Lsun]")
        ax.legend(frameon=False, fontsize=8, loc="best")

    ax = axes[1]
    if not cycles:
        ax.text(0.5, 0.5, f"No radial cycles in {history_source}", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        tail = cycles[-int(last_cycles) :]
        cycle_no = np.asarray([cycle["cycle_number"] for cycle in tail], dtype=float)
        max_l = np.asarray([cycle["max_l_lsun"] for cycle in tail], dtype=float)
        min_v = np.asarray([cycle["min_abs_mag_v"] for cycle in tail], dtype=float)
        ax.plot(cycle_no, max_l, color="#FBBF24", lw=1.2, label="max L per radial cycle")
        ax.set_xlabel("radial cycle number")
        ax.set_ylabel("max L [Lsun]")
        ax2 = ax.twinx()
        ax2.plot(cycle_no, min_v, color="#669BBC", lw=1.0, alpha=0.95, label="min V mag")
        ax2.set_ylabel("min abs_mag_V")
        ax2.invert_yaxis()
        lines = ax.get_lines() + ax2.get_lines()
        labels = [line.get_label() for line in lines]
        ax.legend(lines, labels, frameon=False, fontsize=8, loc="best")

    mass = fortran_float(record.get("RSP_mass"))
    z_value = fortran_float(record.get("RSP_Z"))
    teff = fortran_float(record.get("RSP_Teff"))
    fig.suptitle(
        f"{model_id}: cycle modulation ({history_source}; M={mass:.4g} Msun, Z={z_value:.4g}, Teff={teff:.5g} K)"
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_overview(rows: list[dict[str, object]], output_dir: Path) -> Path:
    path = output_dir / "cycle_modulation_overview.png"
    finished = [
        row
        for row in rows
        if row.get("max_l_modulation_fraction") is not None
        and row.get("last_cycle_count_used", 0)
    ]
    finished.sort(key=lambda row: str(row["model_id"]))
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
    labels = [str(row["model_id"]) for row in finished]
    x = np.arange(len(labels), dtype=float)
    metrics = [
        ("max_l_modulation_fraction", "max L modulation fraction"),
        ("min_v_modulation_mag", "min V modulation [mag]"),
        ("period_modulation_fraction", "period modulation fraction"),
    ]
    for ax, (key, ylabel) in zip(axes, metrics):
        values = np.asarray([float(row[key]) for row in finished], dtype=float)
        colors = [
            "#C1121F"
            if (
                float(row.get("max_l_modulation_fraction") or 0.0) > 0.02
                or float(row.get("min_v_modulation_mag") or 0.0) > 0.02
                or float(row.get("period_modulation_fraction") or 0.0) > 0.02
            )
            else "#669BBC"
            for row in finished
        ]
        ax.bar(x, values, color=colors, alpha=0.88)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, labels)
        ax.axhline(0.0, color="0.2", lw=0.9)
    fig.suptitle("Cycle-to-cycle modulation over latest available radial cycles")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_outputs(rows: list[dict[str, object]], output_dir: Path) -> tuple[Path, Path]:
    csv_path = output_dir / "cycle_modulation_summary.csv"
    json_path = output_dir / "cycle_modulation_summary.json"
    fieldnames = [
        "model_id",
        "run_name",
        "history_source",
        "history_path",
        "cycle_count",
        "last_cycle_count_used",
        "max_l_modulation_fraction",
        "min_v_modulation_mag",
        "period_modulation_fraction",
        "radius_amplitude_modulation_fraction",
        "last_max_l_lsun",
        "last_min_abs_mag_v",
        "last_period_days",
        "diagnostic_png",
        "history_candidate_cycle_counts",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    return csv_path, json_path


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    output_dir = workspace / "output"
    diagnostic_dir = output_dir / "cycle_diagnostics"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_json(workspace / "inputs" / "manifest.json")
    if not isinstance(manifest, list):
        raise RuntimeError(f"Could not read manifest from {workspace / 'inputs' / 'manifest.json'}")

    rows: list[dict[str, object]] = []
    for record in manifest:
        if not isinstance(record, dict) or "model_id" not in record:
            continue
        run_dir = Path(str(record["run_dir"]))
        chosen = choose_history_cycles(run_dir, int(args.last_cycles))
        if chosen is None:
            rows.append(
                {
                    "model_id": str(record["model_id"]),
                    "run_name": str(record.get("run_name", record["model_id"])),
                    "history_source": None,
                    "history_path": None,
                    "cycle_count": 0,
                    "last_cycle_count_used": 0,
                    "diagnostic_png": None,
                    "history_candidate_cycle_counts": [],
                }
            )
            continue
        source, path, cycles, candidate_diagnostics = chosen
        try:
            metrics = modulation_metrics(cycles, int(args.last_cycles))
            png_path = plot_model_diagnostic(record, cycles, source, path, diagnostic_dir, int(args.last_cycles))
            row = {
                "model_id": str(record["model_id"]),
                "run_name": str(record.get("run_name", record["model_id"])),
                "history_source": source,
                "history_path": str(path),
                "diagnostic_png": str(png_path),
                "history_candidate_cycle_counts": candidate_diagnostics,
                **metrics,
            }
        except Exception as exc:
            row = {
                "model_id": str(record["model_id"]),
                "run_name": str(record.get("run_name", record["model_id"])),
                "history_source": source,
                "history_path": str(path),
                "cycle_count": 0,
                "last_cycle_count_used": 0,
                "diagnostic_png": None,
                "error": repr(exc),
                "history_candidate_cycle_counts": candidate_diagnostics,
            }
        rows.append(row)

    overview_path = plot_overview(rows, diagnostic_dir)
    csv_path, json_path = write_outputs(rows, output_dir)
    print(csv_path)
    print(json_path)
    print(overview_path)


if __name__ == "__main__":
    main()
