from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

WORKSPACE_VENDOR = Path(__file__).resolve().parent / ".vendor"
if WORKSPACE_VENDOR.exists():
    sys.path.insert(0, str(WORKSPACE_VENDOR))
else:
    WORKSPACE_PYDEPS = Path(__file__).resolve().parent / ".pydeps"
    if WORKSPACE_PYDEPS.exists():
        sys.path.insert(0, str(WORKSPACE_PYDEPS))

from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
import numpy as np

from plot_fourier_vs_logT import COMPLEX_TRANSFER_REFERENCE_COLORS, geometric_midpoint, load_mean_light_profile
from plot_fourier_vs_massdepth_profiles import profile_photosphere_state
from plot_luminosity_logT_phase_cycle_gif import (
    PHASE_CURVE_COLOR,
    PHOTOSPHERE_COLOR,
    fractional_padding,
    instantaneous_zone_structure,
    interpolate_profile_at_phase,
    uniform_phase_grid,
)
from plot_mean_light_work_terms_vs_logT import collect_profile_records, load_json, select_last_complete_cycle
from plot_work_logT_phase_cycle_gif import (
    DARK_THEME,
    FIGURE_DPI,
    FIGURE_HEIGHT_PX,
    FIGURE_WIDTH_PX,
    scaled_font,
    style_axis_for_theme,
    style_legend_for_theme,
    theme_value,
)

HOT_TEMPERATURE_LIMIT = 2.0e5
DEFAULT_FPS = 24
DEFAULT_MAX_FRAMES = 240
LOOP_LINE_ALPHA = 0.28
TRAIL_LINE_ALPHA = 0.95
SHELL_SPECS = (
    ("He II Ionization", "He II", COMPLEX_TRANSFER_REFERENCE_COLORS["He II Ionization"]),
    ("H/He I Ionization", "H / He I", COMPLEX_TRANSFER_REFERENCE_COLORS["H/He I Ionization"]),
)
PHOTOSPHERE_SPEC = ("Photosphere", "Photosphere", PHOTOSPHERE_COLOR)
DARK_PV_SHELL_COLORS = {
    "He II Ionization": "#8EC5E8",
    "H/He I Ionization": "#D8B48A",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Animate representative shell pressure-volume loops across one nonlinear pulsation cycle."
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory containing LOGS/profile*.data")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for the output GIF and summary")
    parser.add_argument("--prefix", required=True, help="Output filename prefix, e.g. mesa_rsp_combined_14507")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="GIF playback speed in frames per second")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=DEFAULT_MAX_FRAMES,
        help="Maximum number of sampled cycle phases to include in the GIF",
    )
    parser.add_argument(
        "--dark-mode",
        action="store_true",
        help="Render a black-canvas version using the same canvas size as the dark r/R work animation.",
    )
    return parser.parse_args()


def padded_limits(values: np.ndarray, pad_fraction: float = 0.08) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    vrange = vmax - vmin
    if vrange <= 0.0:
        pad = max(abs(vmin), 1.0) * pad_fraction
        return vmin - pad, vmax + pad
    pad = vrange * pad_fraction
    return vmin - pad, vmax + pad


def cumulative_specific_work_curve(
    pressure: np.ndarray,
    specific_volume: np.ndarray,
    phase: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    pressure_array = np.asarray(pressure, dtype=float)
    specific_volume_array = np.asarray(specific_volume, dtype=float)
    phase_array = np.asarray(phase, dtype=float)
    if pressure_array.ndim != 1 or specific_volume_array.ndim != 1 or phase_array.ndim != 1:
        raise ValueError("Pressure, specific volume, and phase must be one-dimensional.")
    if pressure_array.size != specific_volume_array.size or pressure_array.size != phase_array.size:
        raise ValueError("Pressure, specific volume, and phase arrays must have the same length.")
    if pressure_array.size < 2:
        raise ValueError("Need at least two phase samples to construct a work curve.")

    phase_closed = np.append(phase_array, 1.0)
    pressure_closed = np.append(pressure_array, pressure_array[0])
    specific_volume_closed = np.append(specific_volume_array, specific_volume_array[0])

    cumulative_work = np.zeros_like(phase_closed, dtype=float)
    for index in range(1, phase_closed.size):
        d_specific_volume = float(specific_volume_closed[index] - specific_volume_closed[index - 1])
        mean_pressure = 0.5 * float(pressure_closed[index] + pressure_closed[index - 1])
        cumulative_work[index] = cumulative_work[index - 1] + mean_pressure * d_specific_volume

    return phase_closed, cumulative_work, float(cumulative_work[-1])


def themed_shell_color(shell: dict[str, object], dark_mode: bool) -> str:
    if not dark_mode:
        return str(shell["color"])
    if bool(shell.get("moving_reference", False)):
        return theme_value(True, "photosphere", str(shell["color"]))
    if str(shell["zone_name"]) in DARK_PV_SHELL_COLORS:
        return DARK_PV_SHELL_COLORS[str(shell["zone_name"])]
    dark_zone_colors = DARK_THEME.get("zone_reference_colors", {})
    if isinstance(dark_zone_colors, dict):
        zone_color = dark_zone_colors.get(str(shell["zone_name"]))
        if zone_color is not None:
            return str(zone_color)
    return str(shell["color"])


def style_pv_axis_for_theme(ax: plt.Axes, dark_mode: bool) -> None:
    if not dark_mode:
        return
    style_axis_for_theme(ax, True)
    text_color = str(DARK_THEME["text"])
    ax.xaxis.get_offset_text().set_color(text_color)
    ax.yaxis.get_offset_text().set_color(text_color)


def q_at_temperature(q_sorted: np.ndarray, temperature_sorted: np.ndarray, target_temperature: float) -> float:
    temperature = np.asarray(temperature_sorted, dtype=float)
    q = np.asarray(q_sorted, dtype=float)
    finite = np.isfinite(temperature) & np.isfinite(q)
    if np.count_nonzero(finite) < 2:
        raise RuntimeError("Could not map temperature to q because too few finite samples were available.")
    order = np.argsort(temperature[finite])
    temperature_ascending = temperature[finite][order]
    q_ascending = q[finite][order]
    clipped_temperature = float(
        np.clip(
            float(target_temperature),
            float(temperature_ascending[0]),
            float(temperature_ascending[-1]),
        )
    )
    return float(np.interp(clipped_temperature, temperature_ascending, q_ascending))


def select_reference_shells(
    run_dir: Path,
    final_cycle_summary_path: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    mean_light_profile = load_mean_light_profile(run_dir, final_cycle_summary_path)
    header = mean_light_profile["header"]
    columns = mean_light_profile["columns"]
    mean_light_frame = instantaneous_zone_structure(header, columns, HOT_TEMPERATURE_LIMIT)

    q_surface_order = np.asarray(columns["q"], dtype=float)
    q_order = np.argsort(q_surface_order)
    q_sorted = q_surface_order[q_order]
    temperature_sorted = np.power(10.0, np.asarray(columns["logT"], dtype=float)[q_order])
    pressure_sorted = np.asarray(columns["pressure"], dtype=float)[q_order]
    density_sorted = np.power(10.0, np.asarray(columns["logRho"], dtype=float)[q_order])
    radius_sorted = np.asarray(columns["radius"], dtype=float)[q_order]

    shells: list[dict[str, object]] = []
    for zone_name, label, color in SHELL_SPECS:
        spans = mean_light_frame["display_zone_spans"].get(zone_name, [])
        if not spans:
            raise RuntimeError(f"Could not locate a {zone_name} span on the mean-light profile.")
        span = spans[0]
        temperature_midpoint = geometric_midpoint(float(span[0]), float(span[1]))
        q_ref = q_at_temperature(q_sorted, temperature_sorted, temperature_midpoint)
        pressure_ref = float(np.interp(q_ref, q_sorted, pressure_sorted))
        density_ref = float(np.interp(q_ref, q_sorted, density_sorted))
        radius_ref = float(np.interp(q_ref, q_sorted, radius_sorted))
        shells.append(
            {
                "zone_name": zone_name,
                "label": label,
                "color": color,
                "temperature_midpoint_K": float(temperature_midpoint),
                "q_ref": float(q_ref),
                "pressure_mean_light": pressure_ref,
                "density_mean_light": density_ref,
                "specific_volume_mean_light": float(1.0 / density_ref),
                "radius_mean_light_rsun": radius_ref,
                "moving_reference": False,
            }
        )

    photosphere = profile_photosphere_state(header, columns)
    photosphere_q = float(photosphere["q_env"])
    photosphere_pressure = float(np.interp(photosphere_q, q_sorted, pressure_sorted))
    photosphere_density = float(np.interp(photosphere_q, q_sorted, density_sorted))
    shells.append(
        {
            "zone_name": str(PHOTOSPHERE_SPEC[0]),
            "label": str(PHOTOSPHERE_SPEC[1]),
            "color": str(PHOTOSPHERE_SPEC[2]),
            "temperature_midpoint_K": float(10.0 ** float(photosphere["logT"])),
            "q_ref": photosphere_q,
            "reference_tau": float(photosphere["tau"]),
            "pressure_mean_light": photosphere_pressure,
            "density_mean_light": photosphere_density,
            "specific_volume_mean_light": float(1.0 / photosphere_density),
            "radius_mean_light_rsun": float(photosphere["radius_rsun"]),
            "moving_reference": True,
        }
    )

    return shells, mean_light_profile


def sample_shell_thermodynamics(
    cycle_records: list[dict[str, object]],
    cycle_phase: np.ndarray,
    sampled_phase: np.ndarray,
    shells: list[dict[str, object]],
) -> tuple[list[dict[str, object]], np.ndarray]:
    profile_cache: dict[int, dict[str, object]] = {}
    frame_rows: list[dict[str, object]] = []
    photosphere_l = np.empty(sampled_phase.size, dtype=float)

    for frame_index, phase_target in enumerate(sampled_phase):
        interpolated_profile = interpolate_profile_at_phase(
            cycle_records,
            cycle_phase,
            float(phase_target),
            profile_cache,
        )
        header = interpolated_profile["header"]
        columns = interpolated_profile["columns"]
        q_surface_order = np.asarray(columns["q"], dtype=float)
        q_order = np.argsort(q_surface_order)
        q_sorted = q_surface_order[q_order]
        pressure_sorted = np.asarray(columns["pressure"], dtype=float)[q_order]
        density_sorted = np.power(10.0, np.asarray(columns["logRho"], dtype=float)[q_order])
        temperature_sorted = np.power(10.0, np.asarray(columns["logT"], dtype=float)[q_order])
        radius_sorted = np.asarray(columns["radius"], dtype=float)[q_order]
        photosphere = profile_photosphere_state(header, columns)

        photosphere_l[frame_index] = float(photosphere["luminosity_lsun"])

        for shell in shells:
            if bool(shell.get("moving_reference", False)):
                q_ref = float(photosphere["q_env"])
            else:
                q_ref = float(shell["q_ref"])
            pressure = float(np.interp(q_ref, q_sorted, pressure_sorted))
            density = float(np.interp(q_ref, q_sorted, density_sorted))
            frame_rows.append(
                {
                    "frame_index": int(frame_index),
                    "phase": float(phase_target),
                    "age_days": float(interpolated_profile["age_days"]),
                    "shell_label": str(shell["label"]),
                    "zone_name": str(shell["zone_name"]),
                    "q_ref": q_ref,
                    "pressure": pressure,
                    "density": density,
                    "specific_volume": float(1.0 / density),
                    "temperature_K": float(np.interp(q_ref, q_sorted, temperature_sorted)),
                    "radius_rsun": float(np.interp(q_ref, q_sorted, radius_sorted)),
                    "photosphere_l_lsun": float(photosphere["luminosity_lsun"]),
                }
            )

    return frame_rows, photosphere_l


def shell_series_from_rows(
    frame_rows: list[dict[str, object]],
    sampled_phase: np.ndarray,
    shells: list[dict[str, object]],
) -> list[dict[str, object]]:
    shell_series: list[dict[str, object]] = []
    for shell in shells:
        shell_rows = [row for row in frame_rows if row["shell_label"] == shell["label"]]
        shell_rows.sort(key=lambda row: int(row["frame_index"]))
        pressure = np.asarray([float(row["pressure"]) for row in shell_rows], dtype=float)
        specific_volume = np.asarray([float(row["specific_volume"]) for row in shell_rows], dtype=float)
        work_phase_closed, cumulative_work_closed, cycle_work = cumulative_specific_work_curve(
            pressure,
            specific_volume,
            sampled_phase,
        )
        shell_series.append(
            {
                **shell,
                "pressure": pressure,
                "specific_volume": specific_volume,
                "pressure_limits": padded_limits(pressure),
                "specific_volume_limits": padded_limits(specific_volume),
                "closed_pressure": np.append(pressure, pressure[0]),
                "closed_specific_volume": np.append(specific_volume, specific_volume[0]),
                "work_phase_closed": work_phase_closed,
                "cumulative_work_closed": cumulative_work_closed,
                "cycle_work": cycle_work,
                "moving_reference": bool(shell.get("moving_reference", False)),
                "reference_tau": shell.get("reference_tau"),
            }
        )
    return shell_series


def write_csv(path: Path, frame_rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "frame_index",
        "phase",
        "age_days",
        "shell_label",
        "zone_name",
        "q_ref",
        "pressure",
        "density",
        "specific_volume",
        "temperature_K",
        "radius_rsun",
        "photosphere_l_lsun",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in frame_rows:
            writer.writerow(
                {
                    key: f"{float(value):.12g}" if isinstance(value, (float, np.floating)) else value
                    for key, value in row.items()
                }
            )


def main() -> None:
    args = parse_args()

    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix
    dark_mode = bool(args.dark_mode)

    phase_curve_color = theme_value(dark_mode, "phase_curve", PHASE_CURVE_COLOR)
    photosphere_color = theme_value(dark_mode, "photosphere", PHOTOSPHERE_COLOR)
    zero_line_color = theme_value(dark_mode, "zero_line", "black")
    marker_edge_color = theme_value(dark_mode, "marker_edge", "white")
    figure_facecolor = theme_value(dark_mode, "figure_face", "white")
    loop_line_alpha = 0.38 if dark_mode else LOOP_LINE_ALPHA

    stem = f"{prefix}_pv_phase_cycle"
    if dark_mode:
        stem = f"{stem}_dark"
    gif_path = output_dir / f"{stem}.gif"
    png_path = output_dir / f"{stem}.png"
    csv_path = output_dir / f"{stem}.csv"
    summary_path = output_dir / f"{stem}_summary.json"
    final_cycle_summary_path = output_dir / f"{prefix}_final_cycle_summary.json"

    if not final_cycle_summary_path.exists():
        raise RuntimeError(f"Missing final-cycle summary JSON: {final_cycle_summary_path}")

    final_cycle_summary = load_json(final_cycle_summary_path)
    profile_records = collect_profile_records(run_dir)
    cycle_records, period_days, phase_reference_days, cycle_source = select_last_complete_cycle(
        profile_records,
        final_cycle_summary,
    )

    cycle_phase = np.mod(
        (np.asarray([float(record["age_days"]) for record in cycle_records], dtype=float) - float(phase_reference_days))
        / float(period_days),
        1.0,
    )
    order = np.argsort(cycle_phase)
    cycle_records = [cycle_records[int(index)] for index in order]
    cycle_phase = np.asarray(cycle_phase[order], dtype=float)

    shells, mean_light_profile = select_reference_shells(run_dir, final_cycle_summary_path)
    sampled_phase = uniform_phase_grid(len(cycle_records), int(args.max_frames))
    frame_rows, photosphere_l = sample_shell_thermodynamics(
        cycle_records,
        cycle_phase,
        sampled_phase,
        shells,
    )
    shell_series = shell_series_from_rows(frame_rows, sampled_phase, shells)
    for shell in shell_series:
        shell["color"] = themed_shell_color(shell, dark_mode)
    write_csv(csv_path, frame_rows)

    right_ylim = fractional_padding(photosphere_l, fraction=0.08)
    cycle_phase_curve = np.append(sampled_phase, 1.0)
    cycle_luminosity_curve = np.append(photosphere_l, photosphere_l[0])
    cycle_phase_curve_two = np.concatenate([cycle_phase_curve, cycle_phase_curve[1:] + 1.0])
    cycle_luminosity_curve_two = np.concatenate([cycle_luminosity_curve, cycle_luminosity_curve[1:]])
    work_shell_series = shell_series

    plt.rcParams.update(
        {
            "font.size": scaled_font(9) if dark_mode else 9,
            "axes.labelsize": scaled_font(11) if dark_mode else 11,
            "xtick.labelsize": scaled_font(8) if dark_mode else 8,
            "ytick.labelsize": scaled_font(8) if dark_mode else 8,
        }
    )

    figure_size_inches = (
        (FIGURE_WIDTH_PX / FIGURE_DPI, FIGURE_HEIGHT_PX / FIGURE_DPI) if dark_mode else (17.2, 6.2)
    )
    figure_dpi = FIGURE_DPI if dark_mode else None
    fig = plt.figure(
        figsize=figure_size_inches,
        dpi=figure_dpi,
        constrained_layout=True,
        facecolor=figure_facecolor,
    )
    grid = fig.add_gridspec(
        2,
        4,
        width_ratios=[1.0, 1.0, 1.0, 1.05],
        height_ratios=[1.0, 0.72],
    )
    pv_axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[0, 2])]
    ax_work = fig.add_subplot(grid[1, 0:3])
    ax_right = fig.add_subplot(grid[:, 3])
    fig.patch.set_facecolor(figure_facecolor)
    for ax in [*pv_axes, ax_work, ax_right]:
        style_pv_axis_for_theme(ax, dark_mode)

    trail_lines: list[plt.Line2D] = []
    current_dots: list[plt.Line2D] = []
    work_trail_lines: list[plt.Line2D] = []
    work_dots: list[plt.Line2D] = []

    for ax, shell in zip(pv_axes, shell_series):
        ax.plot(
            np.asarray(shell["closed_specific_volume"], dtype=float),
            np.asarray(shell["closed_pressure"], dtype=float),
            color=str(shell["color"]),
            linewidth=1.25,
            alpha=loop_line_alpha,
            zorder=1,
        )
        (trail_line,) = ax.plot(
            [float(shell["specific_volume"][0])],
            [float(shell["pressure"][0])],
            color=str(shell["color"]),
            linewidth=2.1,
            alpha=TRAIL_LINE_ALPHA,
            zorder=2,
        )
        (current_dot,) = ax.plot(
            [float(shell["specific_volume"][0])],
            [float(shell["pressure"][0])],
            marker="o",
            markersize=6.0,
            color=str(shell["color"]),
            markeredgecolor=marker_edge_color,
            markeredgewidth=0.6,
            linestyle="None",
            zorder=3,
        )
        ax.set_xlim(*shell["specific_volume_limits"])
        ax.set_ylim(*shell["pressure_limits"])
        ax.set_xlabel(r"$V = 1/\rho\ [{\rm cm}^3\,{\rm g}^{-1}]$")
        ax.set_ylabel(r"$P\ [{\rm dyn}\,{\rm cm}^{-2}]$")
        ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
        ax.grid(False)
        style_pv_axis_for_theme(ax, dark_mode)
        ax.text(
            0.03,
            0.96,
            (
                f"{shell['label']}\n"
                f"$\\tau$ = {float(shell['reference_tau']):.3g}\n"
                f"T = {float(shell['temperature_midpoint_K']):,.0f} K"
                if bool(shell.get("moving_reference", False))
                else f"{shell['label']}\n"
                f"q = {float(shell['q_ref']):.7f}\n"
                f"T = {float(shell['temperature_midpoint_K']):,.0f} K"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            color=str(shell["color"]),
        )
        trail_lines.append(trail_line)
        current_dots.append(current_dot)

    cumulative_work_arrays = [np.asarray(shell["cumulative_work_closed"], dtype=float) for shell in work_shell_series]
    work_ylim = padded_limits(np.concatenate(cumulative_work_arrays + [np.array([0.0], dtype=float)]))
    ax_work.axhline(0.0, color=zero_line_color, linewidth=0.9, linestyle="--", alpha=0.75, zorder=0)
    for shell in work_shell_series:
        work_phase_closed = np.asarray(shell["work_phase_closed"], dtype=float)
        cumulative_work_closed = np.asarray(shell["cumulative_work_closed"], dtype=float)
        ax_work.plot(
            work_phase_closed,
            cumulative_work_closed,
            color=str(shell["color"]),
            linewidth=1.25,
            alpha=loop_line_alpha,
            zorder=1,
        )
        (work_trail_line,) = ax_work.plot(
            [float(work_phase_closed[0])],
            [float(cumulative_work_closed[0])],
            color=str(shell["color"]),
            linewidth=2.1,
            alpha=TRAIL_LINE_ALPHA,
            zorder=2,
            label=str(shell["label"]),
        )
        (work_dot,) = ax_work.plot(
            [float(work_phase_closed[0])],
            [float(cumulative_work_closed[0])],
            marker="o",
            markersize=5.8,
            color=str(shell["color"]),
            markeredgecolor=marker_edge_color,
            markeredgewidth=0.55,
            linestyle="None",
            zorder=3,
        )
        work_trail_lines.append(work_trail_line)
        work_dots.append(work_dot)
    ax_work.set_xlabel("Pulsation phase")
    ax_work.set_ylabel(r"Cumulative $\int P\,dV\ [{\rm erg}\,{\rm g}^{-1}]$")
    ax_work.set_xlim(0.0, 1.0)
    ax_work.set_ylim(*work_ylim)
    ax_work.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax_work.grid(False)
    style_pv_axis_for_theme(ax_work, dark_mode)
    work_legend = ax_work.legend(
        loc="upper left",
        ncol=min(3, len(work_shell_series)),
        frameon=False,
        fontsize=scaled_font(8.4) if dark_mode else 8.4,
        handlelength=2.8,
        columnspacing=1.4,
    )
    style_legend_for_theme(work_legend, dark_mode)

    ax_right.plot(
        cycle_phase_curve_two,
        cycle_luminosity_curve_two,
        color=phase_curve_color,
        linewidth=1.65,
        zorder=1,
    )
    (phase_dots,) = ax_right.plot(
        [sampled_phase[0], sampled_phase[0] + 1.0],
        [float(photosphere_l[0]), float(photosphere_l[0])],
        marker="o",
        markersize=6.3,
        color=photosphere_color,
        markeredgecolor=marker_edge_color,
        markeredgewidth=0.65,
        linestyle="None",
        zorder=3,
    )
    ax_right.set_xlabel("Pulsation phase")
    ax_right.set_ylabel(r"Photosphere $L\ [L_\odot]$")
    ax_right.set_xlim(0.0, 2.0)
    ax_right.set_ylim(*right_ylim)
    ax_right.grid(False)
    style_pv_axis_for_theme(ax_right, dark_mode)

    def update(frame_index: int) -> tuple[object, ...]:
        artists: list[object] = [phase_dots]
        for shell, trail_line, current_dot in zip(shell_series, trail_lines, current_dots):
            pressure = np.asarray(shell["pressure"], dtype=float)
            specific_volume = np.asarray(shell["specific_volume"], dtype=float)
            trail_line.set_data(specific_volume[: frame_index + 1], pressure[: frame_index + 1])
            current_dot.set_data([float(specific_volume[frame_index])], [float(pressure[frame_index])])
            artists.extend([trail_line, current_dot])
        for shell, work_trail_line, work_dot in zip(work_shell_series, work_trail_lines, work_dots):
            work_phase_closed = np.asarray(shell["work_phase_closed"], dtype=float)
            cumulative_work_closed = np.asarray(shell["cumulative_work_closed"], dtype=float)
            work_trail_line.set_data(
                work_phase_closed[: frame_index + 1],
                cumulative_work_closed[: frame_index + 1],
            )
            work_dot.set_data(
                [float(work_phase_closed[frame_index])],
                [float(cumulative_work_closed[frame_index])],
            )
            artists.extend([work_trail_line, work_dot])

        current_phase = float(sampled_phase[frame_index])
        current_luminosity = float(photosphere_l[frame_index])
        phase_dots.set_data([current_phase, current_phase + 1.0], [current_luminosity, current_luminosity])
        return tuple(artists)

    animation = FuncAnimation(
        fig,
        update,
        frames=len(sampled_phase),
        interval=1000.0 / max(int(args.fps), 1),
        blit=False,
    )
    animation.save(gif_path, writer=PillowWriter(fps=max(int(args.fps), 1)))

    update(len(sampled_phase) - 1)
    fig.savefig(png_path, dpi=FIGURE_DPI if dark_mode else 220, facecolor=fig.get_facecolor())
    plt.close(fig)

    summary = {
        "prefix": prefix,
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "gif_path": str(gif_path),
        "png_path": str(png_path),
        "csv_path": str(csv_path),
        "cycle_source": cycle_source,
        "cycle_profile_count": len(cycle_records),
        "frame_count": len(sampled_phase),
        "fps": int(args.fps),
        "dark_mode": dark_mode,
        "figure_size_px": [
            int(round(float(figure_size_inches[0]) * float(fig.dpi))),
            int(round(float(figure_size_inches[1]) * float(fig.dpi))),
        ],
        "period_days_used": float(period_days),
        "phase_reference_days_used": float(phase_reference_days),
        "pressure_definition": "MESA profile pressure column = total thermodynamic pressure pgas + prad",
        "specific_volume_definition": "V = 1/rho using the nonlinear hydro density field",
        "pdv_definition": "Local loop orientation corresponds to the sign of integral[P d(1/rho)] over the cycle for the chosen shell",
        "mean_light_profile_path": str(mean_light_profile["path"]),
        "mean_light_profile_age_days": float(mean_light_profile["age_days"]),
        "selected_shells": [
            {
                "zone_name": str(shell["zone_name"]),
                "label": str(shell["label"]),
                "color": str(shell["color"]),
                "q_ref": float(shell["q_ref"]),
                "moving_reference": bool(shell.get("moving_reference", False)),
                "reference_tau": (
                    float(shell["reference_tau"]) if shell.get("reference_tau") is not None else None
                ),
                "temperature_midpoint_K": float(shell["temperature_midpoint_K"]),
                "pressure_mean_light": float(shell["pressure_mean_light"]),
                "density_mean_light": float(shell["density_mean_light"]),
                "specific_volume_mean_light": float(shell["specific_volume_mean_light"]),
                "radius_mean_light_rsun": float(shell["radius_mean_light_rsun"]),
                "pressure_limits": [
                    float(shell["pressure_limits"][0]),
                    float(shell["pressure_limits"][1]),
                ],
                "specific_volume_limits": [
                    float(shell["specific_volume_limits"][0]),
                    float(shell["specific_volume_limits"][1]),
                ],
                "cycle_work_erg_per_g": float(shell["cycle_work"]),
            }
            for shell in shell_series
        ],
        "work_panel_shell_labels": [str(shell["label"]) for shell in work_shell_series],
        "sampled_phases": [float(value) for value in sampled_phase],
        "work_panel_ylim": [float(work_ylim[0]), float(work_ylim[1])],
        "photosphere_luminosity_ylim": [float(right_ylim[0]), float(right_ylim[1])],
    }
    summary_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
