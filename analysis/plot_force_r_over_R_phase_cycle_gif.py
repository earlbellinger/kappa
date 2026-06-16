from __future__ import annotations

import argparse
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
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

from plot_fourier_vs_massdepth_profiles import profile_photosphere_state
from plot_luminosity_logT_phase_cycle_gif import (
    DEFAULT_FPS,
    DEFAULT_MAX_FRAMES,
    HOT_TEMPERATURE_LIMIT,
    PHASE_CURVE_COLOR,
    PHOTOSPHERE_COLOR,
    fractional_padding,
    instantaneous_zone_structure,
    interpolate_profile_at_phase,
    uniform_phase_grid,
    smooth_display_zone_spans,
)
from plot_mean_light_work_terms_vs_logT import collect_profile_records, load_json, select_last_complete_cycle
from plot_work_logT_phase_cycle_gif import (
    DARK_THEME,
    G_CGS,
    RADIUS_GAUGE_HALF_HEIGHT,
    RADIUS_GAUGE_X,
    RADIUS_TEXT_POSITION,
    add_photosphere_exterior_glow,
    add_photosphere_visual_state,
    add_zone_overlays_coordinate,
    build_series_label,
    compute_main_radius_xlim,
    label_positions_from_spans,
    load_blackbody_color_table,
    map_q_spans_to_coordinate,
    render_photosphere_sphere_rgba,
    scaled_font,
    smooth_zone_boundaries_in_q,
    style_axis_for_theme,
    style_legend_for_theme,
    theme_value,
)

PRESSURE_GRADIENT_COLOR = "#73DACA"
GRAVITY_FORCE_COLOR = "#FF6B6B"
PRESSURE_GRADIENT_SMOOTHING_KERNEL = np.asarray([1.0, 2.0, 3.0, 2.0, 1.0], dtype=float)
PRESSURE_GRADIENT_SMOOTHING_KERNEL /= np.sum(PRESSURE_GRADIENT_SMOOTHING_KERNEL)
FORCE_YLIM = (-100.0, 1000.0)
SPHERE_CENTER_RIGHT_PANEL = (0.55, 64.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Animate phase-local pressure-gradient and gravitational accelerations versus radius."
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
        "--hot-limit",
        type=float,
        default=HOT_TEMPERATURE_LIMIT,
        help="Hot-side temperature limit in K for the displayed shell window",
    )
    parser.add_argument(
        "--radius-xmin",
        type=float,
        default=None,
        help="Optional lower radius limit in solar radii.",
    )
    parser.add_argument(
        "--radius-xmax",
        type=float,
        default=None,
        help="Optional upper radius limit in solar radii.",
    )
    parser.add_argument(
        "--dark-mode",
        action="store_true",
        help="Render a black-canvas version with the phase-dependent photosphere visualization on the right panel.",
    )
    parser.add_argument(
        "--blackbody-color-file",
        type=Path,
        default=None,
        help="Optional local path to the Vendian blackbody color table.",
    )
    return parser.parse_args()


def interpolate_on_q(q_sorted: np.ndarray, values_sorted: np.ndarray, target_q: float) -> float:
    q = np.asarray(q_sorted, dtype=float)
    values = np.asarray(values_sorted, dtype=float)
    if q.size == 0:
        return float("nan")
    target = float(np.clip(float(target_q), float(q[0]), float(q[-1])))
    return float(np.interp(target, q, values))


def smooth_force_series(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size < PRESSURE_GRADIENT_SMOOTHING_KERNEL.size:
        return array.copy()
    pad = PRESSURE_GRADIENT_SMOOTHING_KERNEL.size // 2
    padded = np.pad(array, (pad, pad), mode="edge")
    return np.convolve(padded, PRESSURE_GRADIENT_SMOOTHING_KERNEL, mode="valid")


def compute_visible_convective_strength(
    header: dict[str, float],
    columns: dict[str, np.ndarray],
    q_order: np.ndarray,
) -> np.ndarray:
    if "rsp_Lc_div_L" in columns:
        return np.abs(np.asarray(columns["rsp_Lc_div_L"], dtype=float)[q_order])
    luminosity_cgs = np.asarray(columns["luminosity"], dtype=float) * float(header["lsun"])
    return np.abs(
        np.divide(
            np.asarray(columns["rsp_Lc"], dtype=float),
            luminosity_cgs,
            out=np.full_like(luminosity_cgs, np.nan, dtype=float),
            where=np.abs(luminosity_cgs) > 0.0,
        )[q_order]
    )


def instantaneous_force_structure(
    header: dict[str, float],
    columns: dict[str, np.ndarray],
    hot_limit: float,
) -> dict[str, object]:
    frame = instantaneous_zone_structure(header, columns, hot_limit)
    photosphere = profile_photosphere_state(header, columns)

    q_surface_order = np.asarray(columns["q"], dtype=float)
    q_order = np.argsort(q_surface_order)
    q_sorted = q_surface_order[q_order]
    temperature_sorted = np.power(10.0, np.asarray(columns["logT"], dtype=float)[q_order])
    radius_rsun_sorted = np.asarray(columns["radius"], dtype=float)[q_order]
    radius_cm_sorted = radius_rsun_sorted * float(header["rsun"])
    pressure_sorted = np.asarray(columns["pressure"], dtype=float)[q_order]
    density_sorted = np.power(10.0, np.asarray(columns["logRho"], dtype=float)[q_order])
    convective_strength_sorted = compute_visible_convective_strength(header, columns, q_order)

    star_mass_g = float(header["star_mass"]) * float(header["msun"])
    envelope_mass_g = star_mass_g - float(header["M_center"])
    if envelope_mass_g <= 0.0:
        raise RuntimeError("Encountered a non-positive envelope mass while computing the force diagnostics.")
    enclosed_mass_sorted_g = float(header["M_center"]) + q_sorted * envelope_mass_g

    visible_mask = (
        np.isfinite(q_sorted)
        & np.isfinite(temperature_sorted)
        & np.isfinite(radius_cm_sorted)
        & np.isfinite(pressure_sorted)
        & np.isfinite(density_sorted)
        & (q_sorted <= float(photosphere["q_env"]))
    )

    q_visible = q_sorted[visible_mask]
    temperature_visible = temperature_sorted[visible_mask]
    radius_rsun_visible = radius_rsun_sorted[visible_mask]
    radius_cm_visible = radius_cm_sorted[visible_mask]
    pressure_visible = pressure_sorted[visible_mask]
    density_visible = density_sorted[visible_mask]
    enclosed_mass_visible_g = enclosed_mass_sorted_g[visible_mask]
    convective_strength_visible = convective_strength_sorted[visible_mask]

    dpressure_dq = np.gradient(pressure_visible, q_visible)
    dradius_dq = np.gradient(radius_cm_visible, q_visible)
    dpressure_dr = np.divide(
        dpressure_dq,
        dradius_dq,
        out=np.full_like(dpressure_dq, np.nan, dtype=float),
        where=np.abs(dradius_dq) > 0.0,
    )
    pressure_gradient_accel_visible = smooth_force_series(-dpressure_dr / density_visible)
    gravity_accel_visible = G_CGS * enclosed_mass_visible_g / np.clip(radius_cm_visible**2, 1.0e-99, None)

    plot_mask = (
        np.isfinite(temperature_visible)
        & np.isfinite(radius_rsun_visible)
        & np.isfinite(pressure_gradient_accel_visible)
        & np.isfinite(gravity_accel_visible)
        & np.isfinite(convective_strength_visible)
        & (temperature_visible <= float(hot_limit))
    )

    photosphere_radius_cm = max(float(photosphere["radius_rsun"]) * float(header["rsun"]), 1.0e-30)
    photosphere_logg_cgs = math.log10(max(G_CGS * star_mass_g / photosphere_radius_cm**2, 1.0e-30))

    frame.update(
        {
            "temperature_plot": temperature_visible[plot_mask],
            "q_plot": q_visible[plot_mask],
            "radius_rsun_plot": radius_rsun_visible[plot_mask],
            "convection_strength_plot": convective_strength_visible[plot_mask],
            "pressure_gradient_force_plot": pressure_gradient_accel_visible[plot_mask],
            "gravity_force_plot": gravity_accel_visible[plot_mask],
            "photosphere_radius_rsun": float(photosphere["radius_rsun"]),
            "photosphere_velocity_km_per_s": float(photosphere["velocity_km_per_s"]),
            "photosphere_logg_cgs": float(photosphere_logg_cgs),
            "photosphere_pressure_gradient_force": interpolate_on_q(
                q_visible,
                pressure_gradient_accel_visible,
                float(photosphere["q_env"]),
            ),
            "photosphere_gravity_force": interpolate_on_q(
                q_visible,
                gravity_accel_visible,
                float(photosphere["q_env"]),
            ),
            "initial_z": float(header.get("initial_z", 2.0e-2)),
        }
    )
    return frame


def global_force_limits(frame_data: list[dict[str, object]]) -> tuple[float, float]:
    combined: list[np.ndarray] = []
    for frame in frame_data:
        combined.append(np.asarray(frame["pressure_gradient_force_plot"], dtype=float))
        combined.append(np.asarray(frame["gravity_force_plot"], dtype=float))
    values = np.concatenate(combined)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    max_abs = float(np.nanmax(np.abs(finite)))
    return -1.08 * max_abs, 1.08 * max_abs


def main() -> None:
    args = parse_args()

    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix
    dark_mode = bool(args.dark_mode)

    phase_curve_color = theme_value(dark_mode, "phase_curve", PHASE_CURVE_COLOR)
    photosphere_color = theme_value(dark_mode, "photosphere", PHOTOSPHERE_COLOR)
    zero_line_color = theme_value(dark_mode, "zero_line", "k")
    marker_edge_color = theme_value(dark_mode, "marker_edge", "white")

    stem = f"{prefix}_forces_r_over_R_phase_cycle"
    if dark_mode:
        stem = f"{stem}_dark"
    gif_path = output_dir / f"{stem}.gif"
    png_path = output_dir / f"{stem}.png"
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

    sampled_phase = uniform_phase_grid(len(cycle_records), int(args.max_frames))
    profile_cache: dict[int, dict[str, object]] = {}
    frame_data: list[dict[str, object]] = []
    for phase_target in sampled_phase:
        interpolated_profile = interpolate_profile_at_phase(
            cycle_records,
            cycle_phase,
            float(phase_target),
            profile_cache,
        )
        frame = instantaneous_force_structure(
            interpolated_profile["header"],
            interpolated_profile["columns"],
            float(args.hot_limit),
        )
        frame["path"] = (
            f"interp:{Path(str(interpolated_profile['left_path'])).name}"
            f"->{Path(str(interpolated_profile['right_path'])).name}"
        )
        frame["age_days"] = float(interpolated_profile["age_days"])
        frame["left_path"] = str(interpolated_profile["left_path"])
        frame["right_path"] = str(interpolated_profile["right_path"])
        frame["left_phase"] = float(interpolated_profile["left_phase"])
        frame["right_phase"] = float(interpolated_profile["right_phase"])
        frame["interpolation_weight"] = float(interpolated_profile["weight"])
        frame_data.append(frame)

    sampled_photosphere_l = np.asarray(
        [float(frame["photosphere_luminosity_lsun"]) for frame in frame_data],
        dtype=float,
    )

    smooth_display_zone_spans(frame_data)
    smooth_zone_boundaries_in_q(frame_data, sampled_phase)

    photosphere_visual_metadata: dict[str, object] = {}
    if dark_mode:
        blackbody_color_table = load_blackbody_color_table(
            args.blackbody_color_file.resolve() if args.blackbody_color_file is not None else None
        )
        photosphere_visual_metadata = add_photosphere_visual_state(frame_data, blackbody_color_table)

    radius_rsun_xlim = (
        float(np.nanmin(np.concatenate([np.asarray(frame["radius_rsun_plot"], dtype=float) for frame in frame_data]))),
        float(np.nanmax(np.concatenate([np.asarray(frame["radius_rsun_plot"], dtype=float) for frame in frame_data]))),
    )
    main_radius_xlim, main_radius_xlim_metadata = compute_main_radius_xlim(
        frame_data,
        radius_rsun_xlim,
        args.radius_xmin,
        args.radius_xmax,
    )
    left_force_ylim = FORCE_YLIM
    right_ylim = fractional_padding(sampled_photosphere_l, fraction=0.08)

    cycle_phase_curve = np.append(sampled_phase, 1.0)
    cycle_luminosity_curve = np.append(sampled_photosphere_l, sampled_photosphere_l[0])
    cycle_phase_curve_two = np.concatenate([cycle_phase_curve, cycle_phase_curve[1:] + 1.0])
    cycle_luminosity_curve_two = np.concatenate([cycle_luminosity_curve, cycle_luminosity_curve[1:]])

    plt.rcParams.update(
        {
            "font.size": scaled_font(9),
            "axes.labelsize": scaled_font(11),
            "xtick.labelsize": scaled_font(8),
            "ytick.labelsize": scaled_font(8),
        }
    )

    figure_facecolor = theme_value(dark_mode, "figure_face", "white")
    fig = plt.figure(figsize=(13.8, 5.9), constrained_layout=True, facecolor=figure_facecolor)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0])
    ax_left = fig.add_subplot(grid[0, 0])
    ax_right = fig.add_subplot(grid[0, 1])
    fig.patch.set_facecolor(figure_facecolor)
    style_axis_for_theme(ax_left, dark_mode)
    style_axis_for_theme(ax_right, dark_mode)

    ax_right.plot(
        cycle_phase_curve_two,
        cycle_luminosity_curve_two,
        color=phase_curve_color,
        linewidth=1.55,
        zorder=1,
    )
    sphere_image: OffsetImage | None = None
    sphere_box: AnnotationBbox | None = None
    radius_axis_line = None
    radius_axis_dot = None
    radius_text = None
    if dark_mode:
        initial_frame = frame_data[0]
        initial_rgba = render_photosphere_sphere_rgba(
            np.asarray(initial_frame["photosphere_blackbody_rgb"], dtype=float),
            float(initial_frame["photosphere_luminosity_sphere_brightness"]),
            tuple(initial_frame["photosphere_limb_darkening"]),
        )
        sphere_image = OffsetImage(initial_rgba, zoom=float(initial_frame["photosphere_sphere_zoom"]))
        sphere_box = AnnotationBbox(
            sphere_image,
            SPHERE_CENTER_RIGHT_PANEL,
            xycoords="data",
            box_alignment=(0.5, 0.5),
            frameon=False,
            pad=0.0,
            zorder=2,
        )
        ax_right.add_artist(sphere_box)
        gauge_y0 = float(RADIUS_TEXT_POSITION[1] - RADIUS_GAUGE_HALF_HEIGHT)
        gauge_y1 = float(RADIUS_TEXT_POSITION[1] + RADIUS_GAUGE_HALF_HEIGHT)
        (radius_axis_line,) = ax_right.plot(
            [RADIUS_GAUGE_X, RADIUS_GAUGE_X],
            [gauge_y0, gauge_y1],
            color=theme_value(dark_mode, "text", "black"),
            linewidth=1.4,
            zorder=3,
        )
        initial_radius_dot_y = gauge_y0 + float(initial_frame["photosphere_radius_axis_unit"]) * (gauge_y1 - gauge_y0)
        (radius_axis_dot,) = ax_right.plot(
            [RADIUS_GAUGE_X],
            [initial_radius_dot_y],
            marker="o",
            markersize=6.0,
            color=tuple(np.asarray(initial_frame["photosphere_marker_rgb"], dtype=float)),
            markeredgecolor=marker_edge_color,
            markeredgewidth=0.6,
            linestyle="None",
            zorder=4,
        )
        radius_text = ax_right.text(
            float(RADIUS_TEXT_POSITION[0]),
            float(RADIUS_TEXT_POSITION[1]),
            rf"{float(initial_frame['photosphere_radius_rsun']):.2f} R$_\odot$",
            color=theme_value(dark_mode, "text", "black"),
            fontsize=scaled_font(7.8),
            ha="center",
            va="center",
            zorder=4,
            path_effects=[pe.withStroke(linewidth=2.0, foreground=theme_value(dark_mode, "axes_face", "white"))],
        )

    (phase_dots,) = ax_right.plot(
        [sampled_phase[0], sampled_phase[0] + 1.0],
        [float(frame_data[0]["photosphere_luminosity_lsun"]), float(frame_data[0]["photosphere_luminosity_lsun"])],
        marker="o",
        markersize=6.5,
        color=tuple(np.asarray(frame_data[0]["photosphere_marker_rgb"], dtype=float)) if dark_mode else photosphere_color,
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
    style_axis_for_theme(ax_right, dark_mode)

    def draw_left_panel(frame_index: int) -> float:
        frame = frame_data[frame_index]
        x_plot = np.asarray(frame["radius_rsun_plot"], dtype=float)
        photosphere_x = float(frame["photosphere_radius_rsun"])
        zone_spans = map_q_spans_to_coordinate(
            frame.get("display_zone_spans_q", frame["display_zone_spans"]),
            np.asarray(frame["q_plot"], dtype=float),
            np.asarray(frame["radius_rsun_plot"], dtype=float),
        )
        current_label_positions = label_positions_from_spans(zone_spans, coordinate="radius_rsun")
        coordinate_profile = {
            "x": np.asarray(frame["radius_rsun_plot"], dtype=float),
            "strength": np.asarray(frame["convection_strength_plot"], dtype=float),
            "max_strength": float(np.nanmax(np.asarray(frame["convection_strength_plot"], dtype=float)))
            if np.any(np.isfinite(np.asarray(frame["convection_strength_plot"], dtype=float)))
            else 0.0,
        }
        series_definitions = [
            (
                np.asarray(frame["pressure_gradient_force_plot"], dtype=float),
                float(frame["photosphere_pressure_gradient_force"]),
                PRESSURE_GRADIENT_COLOR,
                2.0,
                "Pressure Gradient  -(1/rho) dP/dr",
            ),
            (
                np.asarray(frame["gravity_force_plot"], dtype=float),
                float(frame["photosphere_gravity_force"]),
                GRAVITY_FORCE_COLOR,
                2.2,
                "Gravity  G M_r/r^2",
            ),
        ]

        ax_left.cla()
        style_axis_for_theme(ax_left, dark_mode)
        if dark_mode:
            tau_placeholder = np.full_like(np.asarray(frame["radius_rsun_plot"], dtype=float), 2.0 / 3.0)
            add_photosphere_exterior_glow(
                ax_left,
                np.asarray(frame["radius_rsun_plot"], dtype=float),
                tau_placeholder,
                photosphere_x,
                2.0 / 3.0,
                float(main_radius_xlim[1]),
                left_force_ylim,
                np.asarray(frame["photosphere_blackbody_rgb"], dtype=float),
                float(frame["photosphere_luminosity_sphere_brightness"]),
            )
        add_zone_overlays_coordinate(
            ax_left,
            zone_spans,
            current_label_positions,
            coordinate_profile,
            dark_mode=dark_mode,
        )
        ax_left.axhline(0.0, color=zero_line_color, linewidth=0.9, linestyle="--", alpha=0.75, zorder=-7)

        for values, photosphere_value, color, linewidth, label in series_definitions:
            ax_left.plot(
                x_plot,
                values,
                color=color,
                linewidth=linewidth,
                zorder=2,
                label=label,
            )
            ax_left.plot(
                [photosphere_x],
                [photosphere_value],
                marker="o",
                markersize=4.8,
                color=color,
                markeredgecolor=marker_edge_color,
                markeredgewidth=0.5,
                linestyle="None",
                zorder=4,
            )

        ax_left.set_xlim(*main_radius_xlim)
        ax_left.set_ylim(*left_force_ylim)
        ax_left.set_xlabel(r"r / R$_\odot$")
        ax_left.set_ylabel(r"Acceleration [cm s$^{-2}$]")
        ax_left.grid(False)
        legend = ax_left.legend(
            loc="upper left",
            frameon=False,
            fontsize=scaled_font(8),
            ncol=1,
            handlelength=1.8,
            columnspacing=1.0,
            prop={"family": "monospace", "size": scaled_font(8)},
        )
        style_legend_for_theme(legend, dark_mode)
        return float(frame["photosphere_luminosity_lsun"])

    def update(frame_index: int) -> tuple[object, ...]:
        photosphere_luminosity = draw_left_panel(frame_index)
        current_phase = float(sampled_phase[frame_index])
        frame = frame_data[frame_index]
        phase_dots.set_data([current_phase, current_phase + 1.0], [photosphere_luminosity, photosphere_luminosity])
        if dark_mode:
            phase_dots.set_color(tuple(np.asarray(frame["photosphere_marker_rgb"], dtype=float)))
        artists: list[object] = [phase_dots]
        if sphere_image is not None and sphere_box is not None:
            sphere_image.set_data(
                render_photosphere_sphere_rgba(
                    np.asarray(frame["photosphere_blackbody_rgb"], dtype=float),
                    float(frame["photosphere_luminosity_sphere_brightness"]),
                    tuple(frame["photosphere_limb_darkening"]),
                )
            )
            sphere_image.set_zoom(float(frame["photosphere_sphere_zoom"]))
            artists.append(sphere_box)
        if radius_axis_dot is not None and radius_text is not None:
            gauge_y0 = float(RADIUS_TEXT_POSITION[1] - RADIUS_GAUGE_HALF_HEIGHT)
            gauge_y1 = float(RADIUS_TEXT_POSITION[1] + RADIUS_GAUGE_HALF_HEIGHT)
            radius_dot_y = gauge_y0 + float(frame["photosphere_radius_axis_unit"]) * (gauge_y1 - gauge_y0)
            radius_axis_dot.set_data([RADIUS_GAUGE_X], [radius_dot_y])
            radius_axis_dot.set_color(tuple(np.asarray(frame["photosphere_marker_rgb"], dtype=float)))
            radius_text.set_text(rf"{float(frame['photosphere_radius_rsun']):.2f} R$_\odot$")
            artists.extend([radius_axis_dot, radius_text])
        if radius_axis_line is not None:
            artists.append(radius_axis_line)
        return tuple(artists)

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frame_data),
        interval=1000.0 / max(int(args.fps), 1),
        blit=False,
    )
    animation.save(gif_path, writer=PillowWriter(fps=max(int(args.fps), 1)))

    update(len(frame_data) - 1)
    fig.savefig(png_path, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)

    summary = {
        "prefix": prefix,
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "gif_path": str(gif_path),
        "png_path": str(png_path),
        "cycle_source": cycle_source,
        "frame_generation": "fixed-q linear interpolation between bracketing nonlinear profiles on a uniform phase grid",
        "layout": "left panel: phase-local pressure-gradient and gravitational accelerations versus r/R_sun; right panel: photosphere luminosity curve",
        "zone_boundary_smoothing": "ionization-zone boundaries are converted from temperature spans to q spans, smoothed over phase with periodic cubic splines, then mapped from q to radius",
        "left_panel_quantity": "phase-local accelerations",
        "left_panel_units": "cm s^-2",
        "component_definitions": {
            "pressure_gradient_force": "outward pressure-gradient acceleration, -(1/rho) dP/dr, lightly smoothed with a 5-point triangular kernel after differentiation",
            "gravity_force": "negated gravitational acceleration magnitude, +G M_r / r^2",
        },
        "cycle_profile_count": len(cycle_records),
        "frame_count": len(frame_data),
        "closing_frame_added_for_seamless_loop": False,
        "fps": int(args.fps),
        "period_days_used": float(period_days),
        "phase_reference_days_used": float(phase_reference_days),
        "hot_temperature_limit_K": float(args.hot_limit),
        "left_force_ylim": [float(left_force_ylim[0]), float(left_force_ylim[1])],
        "radius_rsun_xlim": [float(radius_rsun_xlim[0]), float(radius_rsun_xlim[1])],
        "main_radius_xlim_used": [float(main_radius_xlim[0]), float(main_radius_xlim[1])],
        "main_radius_xlim_selection": main_radius_xlim_metadata,
        "right_luminosity_ylim": [float(right_ylim[0]), float(right_ylim[1])],
        "sampled_frame_paths": [str(frame["path"]) for frame in frame_data],
        "sampled_frame_left_paths": [str(frame["left_path"]) for frame in frame_data],
        "sampled_frame_right_paths": [str(frame["right_path"]) for frame in frame_data],
        "sampled_frame_interpolation_weights": [float(frame["interpolation_weight"]) for frame in frame_data],
        "sampled_phases": [float(value) for value in sampled_phase],
        "dark_mode": dark_mode,
        "photosphere_visualization": (
            {
                "style": "dark-mode limb-darkened sphere on the right panel",
                "anchor_phase_luminosity": [float(SPHERE_CENTER_RIGHT_PANEL[0]), float(SPHERE_CENTER_RIGHT_PANEL[1])],
                "radius_text_anchor_phase_luminosity": [float(RADIUS_TEXT_POSITION[0]), float(RADIUS_TEXT_POSITION[1])],
                "radius_gauge_x_phase": float(RADIUS_GAUGE_X),
                "radius_gauge_half_height_lsun": float(RADIUS_GAUGE_HALF_HEIGHT),
                **photosphere_visual_metadata,
            }
            if dark_mode
            else None
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
