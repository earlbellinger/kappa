from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import sys
from pathlib import Path

WORKSPACE_VENDOR = Path(__file__).resolve().parent / ".vendor"
if WORKSPACE_VENDOR.exists():
    sys.path.insert(0, str(WORKSPACE_VENDOR))
else:
    WORKSPACE_PYDEPS = Path(__file__).resolve().parent / ".pydeps"
    if WORKSPACE_PYDEPS.exists():
        sys.path.insert(0, str(WORKSPACE_PYDEPS))

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.transforms import blended_transform_factory
import numpy as np

from plot_fourier_vs_massdepth_profiles import (
    build_zone_spans,
    parse_profile,
    profile_key,
    profile_photosphere_state,
)

TWOPI = 2.0 * math.pi

COMPLEX_TRANSFER_PHASE_YLIM = (-math.pi / 4.0, math.pi / 2.0)
COMPLEX_TRANSFER_HARMONIC_COLORS = {
    1: "#3B528B",
    2: "#21918C",
    3: "#5EC962",
}
COMPLEX_TRANSFER_REFERENCE_COLORS = {
    "He II Ionization": "#6C8A6B",
    "He I Ionization": "#A38B62",
    "H Ionization": "#B8756A",
    "H/He I Ionization": "#8D7A67",
}
IONIZATION_ZONE_ORDER = (
    "He II Ionization",
    "He I Ionization",
    "H Ionization",
    "H/He I Ionization",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Fourier amplitudes and phase differences against the mean-light shell temperature."
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory containing Fourier scan outputs")
    parser.add_argument("--prefix", required=True, help="Output filename prefix, e.g. mesa_rsp_profile1188")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def load_fourier_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    columns: dict[str, np.ndarray] = {}
    for field in reader.fieldnames or []:
        columns[field] = np.asarray([float(row[field]) for row in rows], dtype=float)
    return columns


def numeric_value(text: str) -> float:
    return float(text.replace("D", "E").replace("d", "e"))


def read_profile_header(path: Path) -> dict[str, float]:
    with path.open() as handle:
        for line in handle:
            if line.strip().startswith("model_number"):
                header_names = line.split()
                header_values = shlex.split(next(handle))
                header: dict[str, float] = {}
                for name, value in zip(header_names, header_values):
                    try:
                        header[name] = numeric_value(value)
                    except ValueError:
                        continue
                return header
    raise RuntimeError(f"Could not locate the header in {path}")


def break_wrapped_series(x_values: np.ndarray, y_values: np.ndarray, wrap: float) -> tuple[np.ndarray, np.ndarray]:
    x_plot = np.asarray(x_values, dtype=float)
    y_plot = np.asarray(y_values, dtype=float)

    if x_plot.size <= 1:
        return x_plot, y_plot

    break_indices = np.where(np.abs(np.diff(y_plot)) > 0.5 * wrap)[0] + 1
    for offset, idx in enumerate(break_indices):
        insert_at = int(idx + offset)
        x_plot = np.insert(x_plot, insert_at, np.nan)
        y_plot = np.insert(y_plot, insert_at, np.nan)

    return x_plot, y_plot


def relative_phase(phases: np.ndarray, reference_phase: float) -> np.ndarray:
    delta = np.angle(np.exp(1j * (np.asarray(phases, dtype=float) - float(reference_phase))))
    return np.unwrap(delta)


def geometric_midpoint(x0: float, x1: float) -> float:
    return float(10.0 ** (0.5 * (math.log10(x0) + math.log10(x1))))


def draw_convection_swirls(
    ax: plt.Axes,
    x0: float,
    x1: float,
    convection_temperature: np.ndarray,
    convection_strength: np.ndarray,
    convection_strength_max: float,
) -> None:
    left = float(min(x0, x1))
    right = float(max(x0, x1))
    log_left = math.log10(left)
    log_right = math.log10(right)
    transform = blended_transform_factory(ax.transData, ax.transAxes)
    span_width = max(log_right - log_left, 1.0e-6)

    log_temperature = np.log10(np.clip(np.asarray(convection_temperature, dtype=float), 1.0e-99, None))
    strength = np.asarray(convection_strength, dtype=float)
    interp_order = np.argsort(log_temperature)
    log_temperature = log_temperature[interp_order]
    strength = strength[interp_order]
    strength_scale = max(float(convection_strength_max), 1.0e-12)

    span_mean_strength = 0.0
    in_span = (log_temperature >= log_left) & (log_temperature <= log_right)
    if np.any(in_span):
        span_mean_strength = float(np.nanmean(strength[in_span]) / strength_scale)

    n_cols = max(10, int(math.ceil(span_width / 0.045)) + int(round(6.0 * span_mean_strength)))
    n_rows = 12 + int(round(3.0 * span_mean_strength))
    theta = np.linspace(0.0, 2.0 * math.pi, 180)

    for row in range(n_rows):
        y_center = 0.06 + row * (0.88 / max(n_rows - 1, 1))
        row_phase = 0.41 * row
        for col in range(n_cols):
            frac = (col + 0.5 + 0.33 * (row % 2)) / (n_cols + 0.6)
            center_log = log_left + frac * span_width
            jitter = 0.012 * math.sin(2.7 * row + 1.9 * col)
            center_log += jitter
            if center_log <= log_left or center_log >= log_right:
                continue

            local_strength = float(np.interp(center_log, log_temperature, strength, left=0.0, right=0.0))
            local_norm = float(np.clip(local_strength / strength_scale, 0.0, 1.0))
            keep_probability = 0.15 + 0.85 * (local_norm ** 0.85)
            pseudo = 0.5 * (1.0 + math.sin(12.9898 * (row + 1) + 78.233 * (col + 1) + 37.719 * center_log))
            if pseudo > keep_probability:
                continue

            half_width_log = 0.017 + 0.004 * math.sin(1.3 * row + 0.8 * col)
            half_height = 0.030 + 0.006 * math.cos(1.1 * row - 0.6 * col)
            petals = 4 + ((row + col) % 2)
            scallop = 0.26 + 0.05 * math.sin(0.7 * row + 1.4 * col)
            radial = 1.0 + scallop * np.cos(petals * theta + row_phase) + 0.08 * np.sin(2.0 * theta + 0.3 * col)
            x_log = center_log + half_width_log * radial * np.cos(theta)
            y = y_center + half_height * radial * np.sin(theta)

            valid = (
                (x_log >= log_left)
                & (x_log <= log_right)
                & (y >= 0.02)
                & (y <= 0.98)
            )
            if np.count_nonzero(valid) < 20:
                continue

            x = np.power(10.0, x_log[valid])
            ax.plot(
                x,
                y[valid],
                color="0.68",
                linewidth=0.38 + 0.24 * local_norm,
                alpha=0.18 + 0.54 * local_norm,
                transform=transform,
                zorder=-5,
                solid_capstyle="round",
            )


def configure_temperature_axis(axes: list[plt.Axes], temperature: np.ndarray) -> None:
    tick_candidates: list[float] = []
    min_temperature = float(np.min(temperature))
    max_temperature = float(np.max(temperature))
    exponent_min = int(math.floor(math.log10(min_temperature)))
    exponent_max = int(math.ceil(math.log10(max_temperature)))
    for exponent in range(exponent_min, exponent_max + 1):
        for mantissa in (1.0, 2.0, 5.0):
            tick_value = mantissa * (10.0 ** exponent)
            if min_temperature <= tick_value <= max_temperature:
                tick_candidates.append(tick_value)

    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlim(max_temperature, min_temperature)
        ax.set_xticks(tick_candidates)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _pos: f"{value:,.0f}"))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())


def load_mean_light_profile(
    run_dir: Path,
    final_cycle_summary_path: Path,
) -> dict[str, object]:
    logs_dir = run_dir / "LOGS"
    profile_paths = sorted(logs_dir.glob("profile*.data"), key=profile_key)
    if not profile_paths:
        raise RuntimeError(f"No profile files found in {logs_dir}")

    cycle_summary = load_json(final_cycle_summary_path) if final_cycle_summary_path.exists() else {}
    cycle_start_age_days = cycle_summary.get("cycle_start_age_days")
    cycle_end_age_days = cycle_summary.get("cycle_end_age_days")

    all_records: list[dict[str, object]] = []
    for path in profile_paths:
        header = read_profile_header(path)
        age_days = float(header["star_age"]) * 365.25
        all_records.append(
            {
                "path": path,
                "header": header,
                "age_days": age_days,
                "photosphere_l_lsun": float(header["photosphere_L"]),
            }
        )

    candidate_records = all_records
    selection_source = "all_available_profiles"
    if cycle_start_age_days is not None and cycle_end_age_days is not None:
        cycle_filtered = [
            record
            for record in all_records
            if float(cycle_start_age_days) <= float(record["age_days"]) <= float(cycle_end_age_days)
        ]
        if cycle_filtered:
            candidate_records = cycle_filtered
            selection_source = "final_cycle_summary_window"

    if not candidate_records:
        raise RuntimeError(f"No profile snapshots were available in {logs_dir}")

    cycle_luminosity = np.asarray([record["photosphere_l_lsun"] for record in candidate_records], dtype=float)
    mean_light_luminosity = float(np.mean(cycle_luminosity))
    selected_index = int(np.argmin(np.abs(cycle_luminosity - mean_light_luminosity)))
    selected_record = candidate_records[selected_index]

    header, columns = parse_profile(Path(selected_record["path"]))
    return {
        "path": str(selected_record["path"]),
        "header": header,
        "columns": columns,
        "age_days": float(selected_record["age_days"]),
        "photosphere_l_lsun": float(selected_record["photosphere_l_lsun"]),
        "mean_cycle_photosphere_l_lsun": mean_light_luminosity,
        "profiles_in_cycle": len(candidate_records),
        "selection_source": selection_source,
    }


def mean_light_zone_structure(
    header: dict[str, float],
    columns: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, list[tuple[float, float]]], dict[str, object], dict[str, object]]:
    photosphere = profile_photosphere_state(header, columns)
    q_surface_order = np.asarray(columns["q"], dtype=float)
    q_order = np.argsort(q_surface_order)
    q_sorted = q_surface_order[q_order]
    temperature_sorted = np.power(10.0, np.asarray(columns["logT"], dtype=float)[q_order])
    cp_sorted = np.asarray(columns["cp"], dtype=float)[q_order]
    gamma1_sorted = np.asarray(columns["gamma1"], dtype=float)[q_order]

    if "rsp_Lc_div_L" in columns:
        convective_fraction_sorted = np.asarray(columns["rsp_Lc_div_L"], dtype=float)[q_order]
    else:
        luminosity_cgs = np.asarray(columns["luminosity"], dtype=float) * float(header["lsun"])
        convective_fraction_sorted = np.divide(
            np.asarray(columns["rsp_Lc"], dtype=float),
            luminosity_cgs,
            out=np.full(q_sorted.size, np.nan, dtype=float),
            where=np.abs(luminosity_cgs) > 0.0,
        )[q_order]

    visible_mask = (
        np.isfinite(temperature_sorted)
        & np.isfinite(q_sorted)
        & (q_sorted <= float(photosphere["q_env"]))
    )

    def optional_sorted(name: str) -> np.ndarray | None:
        values = columns.get(name)
        if values is None:
            return None
        return np.asarray(values, dtype=float)[q_order]

    ionization_he4_sorted = optional_sorted("ionization_he4")
    zone_spans, _convective_threshold, zone_details = build_zone_spans(
        massdepth_sorted=temperature_sorted,
        mean_cp_sorted=cp_sorted,
        mean_gamma1_sorted=gamma1_sorted,
        mean_abs_convective_fraction_sorted=np.abs(convective_fraction_sorted),
        visible_mask=visible_mask,
        mean_typical_charge_h1_sorted=optional_sorted("typical_charge_h1"),
        mean_ionization_h1_sorted=optional_sorted("ionization_h1"),
        mean_typical_charge_he4_sorted=optional_sorted("typical_charge_he4"),
        mean_ionization_he4_sorted=ionization_he4_sorted,
    )
    for zone_name in ("He II Ionization", "He I Ionization", "H Ionization", "H/He I Ionization"):
        details = zone_details.get(zone_name)
        if isinstance(details, dict) and "source" in details:
            details["source"] = str(details["source"]).replace("phase-averaged", "mean-light")
        if isinstance(details, dict) and "message" in details:
            details["message"] = str(details["message"]).replace("phase-averaged", "mean-light")
    convection_details = zone_details.get("Convection")
    if isinstance(convection_details, dict) and "source" in convection_details:
        convection_details["source"] = "mean-light |Lc|/L on the profile mesh"

    if not zone_spans.get("He II Ionization") and ionization_he4_sorted is not None:
        visible_indices = np.flatnonzero(visible_mask)
        ion_visible = ionization_he4_sorted[visible_mask]
        temp_visible = temperature_sorted[visible_mask]
        jump_indices = np.where(np.abs(np.diff(ion_visible)) > 0.5)[0]
        if jump_indices.size > 0:
            jump = int(jump_indices[0])
            x0 = float(temp_visible[jump])
            x1 = float(temp_visible[min(jump + 1, temp_visible.size - 1)])
            zone_spans["He II Ionization"] = [[min(x0, x1), max(x0, x1)]]
            zone_details["He II Ionization"] = {
                "source": "mean-light ionization_he4 jump on the profile mesh",
                "detected": True,
                "fallback_discrete_jump": True,
                "jump_temperatures_K": [min(x0, x1), max(x0, x1)],
                "jump_profile_indices": [
                    int(visible_indices[jump]),
                    int(visible_indices[min(jump + 1, visible_indices.size - 1)]),
                ],
            }

    convection_profile = {
        "temperature_K": temperature_sorted[visible_mask],
        "strength": np.abs(convective_fraction_sorted[visible_mask]),
        "max_strength": float(np.nanmax(np.abs(convective_fraction_sorted[visible_mask])))
        if np.any(np.isfinite(convective_fraction_sorted[visible_mask]))
        else 0.0,
    }

    return q_sorted, temperature_sorted, zone_spans, zone_details, convection_profile


def add_zone_overlays(
    axes: list[plt.Axes],
    zone_spans: dict[str, list[tuple[float, float]]],
    photosphere_temperature: float,
    convection_profile: dict[str, object],
) -> None:
    label_axis = axes[1]
    text_transform = blended_transform_factory(label_axis.transData, label_axis.transAxes)

    for x0, x1 in zone_spans.get("Convection", []):
        for ax in axes:
            draw_convection_swirls(
                ax,
                x0,
                x1,
                np.asarray(convection_profile["temperature_K"], dtype=float),
                np.asarray(convection_profile["strength"], dtype=float),
                float(convection_profile["max_strength"]),
            )

    for name in IONIZATION_ZONE_ORDER:
        spans = zone_spans.get(name, [])
        if not spans:
            continue
        color = COMPLEX_TRANSFER_REFERENCE_COLORS.get(name, "0.45")
        for x0, x1 in spans:
            for ax in axes:
                ax.axvspan(
                    x0,
                    x1,
                    facecolor=color,
                    edgecolor="none",
                    linewidth=0.0,
                    alpha=0.17,
                    zorder=-4,
                )
        label_x = geometric_midpoint(float(spans[0][0]), float(spans[0][1]))
        label_axis.text(
            label_x,
            0.98,
            name,
            rotation=90,
            ha="right",
            va="top",
            fontsize=8,
            color=color,
            transform=text_transform,
        )

    for ax in axes:
        ax.axvline(photosphere_temperature, color="0.2", linewidth=1.15, linestyle=":", zorder=-3)
    label_axis.text(
        photosphere_temperature * (10.0 ** -0.010),
        0.98,
        "Photosphere",
        rotation=90,
        ha="left",
        va="top",
        fontsize=8,
        color="0.2",
        transform=text_transform,
    )


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir.resolve()
    prefix = args.prefix

    fourier_csv_path = output_dir / f"{prefix}_fourier_depth_scan.csv"
    fourier_summary_path = output_dir / f"{prefix}_fourier_depth_scan_summary.json"
    final_cycle_summary_path = output_dir / f"{prefix}_final_cycle_summary.json"
    png_path = output_dir / f"{prefix}_fourier_vs_logT.png"
    summary_path = output_dir / f"{prefix}_fourier_vs_logT_summary.json"

    if not fourier_csv_path.exists():
        raise RuntimeError(f"Missing Fourier CSV: {fourier_csv_path}")
    if not fourier_summary_path.exists():
        raise RuntimeError(f"Missing Fourier summary JSON: {fourier_summary_path}")

    fourier = load_fourier_csv(fourier_csv_path)
    fourier_summary = load_json(fourier_summary_path)
    run_dir = Path(str(fourier_summary["run_dir"])).resolve()

    mean_light_profile = load_mean_light_profile(run_dir, final_cycle_summary_path)
    mean_light_header = mean_light_profile["header"]
    mean_light_columns = mean_light_profile["columns"]
    photosphere = profile_photosphere_state(mean_light_header, mean_light_columns)
    photosphere_temperature = float(10.0 ** float(photosphere["logT"]))

    q_sorted, temperature_profile_sorted, zone_spans, zone_details, convection_profile = mean_light_zone_structure(
        mean_light_header,
        mean_light_columns,
    )
    sampled_temperature = np.interp(
        np.asarray(fourier["q"], dtype=float),
        q_sorted,
        temperature_profile_sorted,
    )

    order = np.argsort(sampled_temperature)
    temperature = sampled_temperature[order]
    q_sorted_on_plot = np.asarray(fourier["q"], dtype=float)[order]
    A1 = fourier["A1"][order]
    A2 = fourier["A2"][order]
    A3 = fourier["A3"][order]
    phi1 = fourier["phi1_rad"][order]
    phi2 = fourier["phi2_rad"][order]
    phi3 = fourier["phi3_rad"][order]
    phase_reference_q = float(photosphere["q_env"])
    photosphere_idx = int(np.argmin(np.abs(q_sorted_on_plot - phase_reference_q)))
    phase_reference_source = "mean-light photosphere tau=2/3 mapped to q"
    phi1_lag = relative_phase(phi1, phi1[photosphere_idx])
    phi2_lag = relative_phase(phi2, phi2[photosphere_idx])
    phi3_lag = relative_phase(phi3, phi3[photosphere_idx])

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )

    fig, (ax_amp, ax_phase) = plt.subplots(2, 1, figsize=(9.4, 6.7), sharex=True, constrained_layout=True)
    axes = [ax_amp, ax_phase]

    add_zone_overlays(axes, zone_spans, photosphere_temperature, convection_profile)

    ax_amp.plot(temperature, A1, color=COMPLEX_TRANSFER_HARMONIC_COLORS[1], linewidth=1.6, label=r"$A_1$")
    ax_amp.plot(temperature, A2, color=COMPLEX_TRANSFER_HARMONIC_COLORS[2], linewidth=1.55, label=r"$A_2$")
    ax_amp.plot(temperature, A3, color=COMPLEX_TRANSFER_HARMONIC_COLORS[3], linewidth=1.55, label=r"$A_3$")
    ax_amp.axhline(0.0, color="k", linewidth=1.0, linestyle="--", zorder=-2)
    ax_amp.set_ylabel("Amplitude")
    ax_amp.grid(False)
    ax_amp.legend(loc="upper left", ncol=3, frameon=True, framealpha=0.9)

    ax_phase.plot(
        temperature,
        phi1_lag,
        color=COMPLEX_TRANSFER_HARMONIC_COLORS[1],
        linewidth=1.5,
        label=r"$\Delta\phi_{1}$",
    )
    ax_phase.plot(
        temperature,
        phi2_lag,
        color=COMPLEX_TRANSFER_HARMONIC_COLORS[2],
        linewidth=1.5,
        label=r"$\Delta\phi_{2}$",
    )
    ax_phase.plot(
        temperature,
        phi3_lag,
        color=COMPLEX_TRANSFER_HARMONIC_COLORS[3],
        linewidth=1.5,
        label=r"$\Delta\phi_{3}$",
    )
    ax_phase.axhline(0.0, color="k", linewidth=1.0, linestyle="--", zorder=-2)
    ax_phase.axhline(math.pi / 4.0, color="0.35", linewidth=1.0, linestyle="--", zorder=-2)
    ax_phase.set_ylabel("Phase Lag [rad]")
    ax_phase.set_xlabel(r"$T\ [{\rm K}]$")
    ax_phase.grid(False)
    ax_phase.legend(loc="upper left", ncol=3, frameon=True, framealpha=0.9)
    ax_phase.set_yticks(
        [-math.pi / 4.0, 0.0, math.pi / 4.0, math.pi / 2.0],
        [r"$-\pi/4$", r"$0$", r"$\pi/4$", r"$\pi/2$"],
    )
    ax_phase.text(
        0.015,
        math.pi / 4.0 + 0.03,
        "quarter cycle",
        transform=blended_transform_factory(ax_phase.transAxes, ax_phase.transData),
        ha="left",
        va="bottom",
        fontsize=9,
        color="0.35",
    )

    ax_phase.set_ylim(*COMPLEX_TRANSFER_PHASE_YLIM)

    configure_temperature_axis(axes, temperature)

    summary = {
        "prefix": prefix,
        "output_dir": str(output_dir),
        "run_dir": str(run_dir),
        "source_fourier_csv": str(fourier_csv_path),
        "source_fourier_summary_json": str(fourier_summary_path),
        "source_final_cycle_summary_json": str(final_cycle_summary_path),
        "png_path": str(png_path),
        "temperature_coordinate_source": "temperature at the mean-light RSP profile interpolated to the Fourier-scan q samples",
        "mean_light_profile_path": str(mean_light_profile["path"]),
        "mean_light_profile_age_days": float(mean_light_profile["age_days"]),
        "mean_light_profile_photosphere_l_lsun": float(mean_light_profile["photosphere_l_lsun"]),
        "mean_selected_cycle_photosphere_l_lsun": float(mean_light_profile["mean_cycle_photosphere_l_lsun"]),
        "profiles_in_selected_cycle": int(mean_light_profile["profiles_in_cycle"]),
        "mean_light_selection_source": str(mean_light_profile["selection_source"]),
        "photosphere_temperature_K": photosphere_temperature,
        "mean_light_photosphere_q_env": float(photosphere["q_env"]),
        "phase_reference_q": phase_reference_q,
        "phase_reference_temperature_K_on_plot": float(temperature[photosphere_idx]),
        "phase_reference_source": phase_reference_source,
        "zone_detection_details": zone_details,
        "zone_spans_temperature_K": {
            name: [[float(x0), float(x1)] for x0, x1 in spans]
            for name, spans in zone_spans.items()
            if name != "Convection"
        },
        "convection_spans_temperature_K": [
            [float(x0), float(x1)] for x0, x1 in zone_spans.get("Convection", [])
        ],
        "fourier_phase_convention": fourier_summary.get("fourier_phase_convention"),
        "phase_plot_reference": "phase differences relative to the photosphere values of phi1, phi2, and phi3",
        "phase_plot_series": ["Delta phi1", "Delta phi2", "Delta phi3"],
        "phase_plot_ylim": list(COMPLEX_TRANSFER_PHASE_YLIM),
        "phase_plot_wrapping": "photosphere-referenced relative phase with unwrapped continuity",
    }

    fig.savefig(png_path, dpi=220)
    plt.close(fig)
    summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"Saved {png_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
