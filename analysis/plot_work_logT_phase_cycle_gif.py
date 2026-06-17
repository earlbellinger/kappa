from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
from pathlib import Path

WORKSPACE_VENDOR = Path(__file__).resolve().parent / ".vendor"
if WORKSPACE_VENDOR.exists():
    sys.path.insert(0, str(WORKSPACE_VENDOR))
else:
    WORKSPACE_PYDEPS = Path(__file__).resolve().parent / ".pydeps"
    if WORKSPACE_PYDEPS.exists():
        sys.path.insert(0, str(WORKSPACE_PYDEPS))

from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from matplotlib.transforms import blended_transform_factory
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from plot_fourier_vs_logT import COMPLEX_TRANSFER_REFERENCE_COLORS, load_mean_light_profile
from plot_fourier_vs_massdepth_profiles import (
    build_fourier_design_matrix,
    normalized_transition_mask,
    profile_photosphere_state,
)
from plot_luminosity_logT_phase_cycle_gif import (
    CONVECTIVE_COLOR,
    DEFAULT_FPS,
    DEFAULT_MAX_FRAMES,
    HOT_TEMPERATURE_LIMIT,
    PHASE_CURVE_COLOR,
    PHOTOSPHERE_COLOR,
    RADIATIVE_COLOR,
    SHELL_CURVE_COLOR,
    add_zone_overlays_single_axis,
    configure_temperature_axis,
    fractional_padding,
    instantaneous_zone_structure,
    interpolate_profile_at_phase,
    load_profile_cached,
    mean_light_zone_label_positions,
    smooth_display_zone_spans,
    uniform_phase_grid,
)
from plot_mean_light_work_terms_vs_logT import collect_profile_records, load_json, select_last_complete_cycle

PDV_COLOR = "#FFB703"
NET_HEATING_COLOR = "#FB8500"
LOWER_PANEL_TEMPERATURE_LIMIT = 1.0e5
INSET_RADIUS_BELOW_PHOTOSPHERE_RSUN = 0.3
INSET_RADIUS_ABOVE_PHOTOSPHERE_RSUN = 0.01
MAIN_RADIUS_INNER_MARGIN_RSUN = 1.0
MAIN_RADIUS_OUTER_MARGIN_RSUN = 0.10
MAIN_RADIUS_ZONE_PADDING_RSUN = 0.05
MAIN_RADIUS_MIN_SPAN_RSUN = 1.0
PDV_PERIODIC_FIT_HARMONICS = 24
ZONE_BOUNDARY_FIT_HARMONICS = 6
SECONDS_PER_DAY = 86400.0
TWO_PI = 2.0 * np.pi
G_CGS = 6.67430e-8
H_CGS = 6.62607015e-27
C_CGS = 2.99792458e10
K_B_CGS = 1.380649e-16
BLACKBODY_COLOR_URL = "http://vendian.org/mncharity/dir3/blackbody/UnstableURLs/bbr_color.txt"
BLACKBODY_CACHE_PATH = Path(__file__).resolve().parent / "bbr_color.txt"
BLACKBODY_TABLE_CMF = "2deg"
BLACKBODY_HIGH_T_ANCHORS_RGB255 = (
    (40000.0, (161.0, 184.0, 255.0)),
    (50000.0, (159.0, 183.0, 255.0)),
    (100000.0, (155.0, 180.0, 255.0)),
    (1000000.0, (152.0, 177.0, 255.0)),
)
SOLAR_Z_REFERENCE = 2.0e-2
SPHERE_CENTER_LUM_PANEL = (0.27, 0.76)
SPHERE_CENTER_RV_PANEL = (0.27, 0.76)
SPHERE_IMAGE_SIZE = 192
SPHERE_BASE_ZOOM = 0.42
SPHERE_EDGE_SOFTENING = 0.03
RADIUS_TEXT_POSITION = (0.33, 0.88)
TEFF_TEXT_POSITION = (0.33, 0.62)
RADIUS_GAUGE_X = 0.23
RADIUS_GAUGE_HALF_HEIGHT = 0.085
TEFF_GAUGE_X = 0.23
TEFF_GAUGE_HALF_HEIGHT = 0.085
EXTERIOR_GLOW_MAX_ALPHA = 0.30
EXTERIOR_GLOW_FALLOFF_POWER = 4.0
EXTERIOR_GLOW_SAMPLES_X = 384
EXTERIOR_GLOW_SAMPLES_Y = 48
INTERIOR_SHADE_SAMPLES_X = 512
INTERIOR_SHADE_SAMPLES_Y = 64
INTERIOR_SHADE_ALPHA_DARK = 0.24
INTERIOR_SHADE_ALPHA_LIGHT = 0.10
INTERIOR_SHADE_LUMINOSITY_FLOOR = 0.30
PHOTOSPHERE_MARKER_BASE_SIZE = 8.4
MAX_TEFF_DOT_COLOR = "#C1121F"
MIN_TEFF_DOT_COLOR = "#C1121F"
V_BAND_WAVELENGTHS_CM = np.linspace(4.75e-5, 6.25e-5, 161)
V_BAND_RESPONSE = np.exp(-0.5 * ((V_BAND_WAVELENGTHS_CM - 5.50e-5) / 4.20e-6) ** 2)
PLOT_SCALE_FACTORS = {
    "heating_total": 1.0,
    "heating_radiative": 5.0e-2,
    "heating_convective": 5.0e-2,
    "pdv_power": 1.0,
}
DISPLAY_POWER_SCALE = 1.0e9
ANIMATION_SCALING_VERSION = "per-panel-visible-window-v2"
ZONE_LABEL_COLOR = "#2B2B2B"
ZONE_LABEL_FONT_SIZE = 9.2
ZONE_LABEL_STROKE_WIDTH = 1.3
LEGEND_NAME_WIDTH = 18
LEGEND_SCALE_WIDTH = 4
TEXT_SCALE_FACTOR = 1.5
AXIS_LINE_WIDTH = 1.5
TARGET_MAJOR_TICKS = 5
FIGURE_WIDTH_PX = 1724
FIGURE_HEIGHT_PX = 800
FIGURE_DPI = 100
SCALED_KAPPA_COLOR = "#CFA8FF"
SCALED_DIAGNOSTIC_PAD_FRACTION = 0.08
POWER_PANEL_PAD_FRACTION = 0.08
POWER_PANEL_MIN_HALF_RANGE = 0.05
OPACITY_PANEL_TOP_FRACTION = 0.88
HIGH_OPACITY_REGION_THRESHOLD = 0.9
HIGH_OPACITY_REGION_FALLBACK_TOP_N = 5
POWER_GUIDE_X_FRACTION_FROM_LEFT = 0.04
POWER_GUIDE_RED = "#C1121F"
POWER_GUIDE_BLUE = "#8ECAE6"
POWER_GUIDE_INACTIVE = "#4A4A4A"
POWER_GUIDE_LABELS = (
    ("heating", 0.74, POWER_GUIDE_RED),
    ("expanding", 0.64, POWER_GUIDE_RED),
    ("contracting", 0.36, POWER_GUIDE_BLUE),
    ("cooling", 0.26, POWER_GUIDE_BLUE),
)
PHOTOSPHERE_LABEL_X_OFFSET_RSUN = 0.035
PHOTOSPHERE_LINK_LINEWIDTH = 1.4
PHASE_REFERENCE_DOT_SIZE = 3.6
LUM_SPHERE_PHASE = 0.54
LUM_SPHERE_Y_FRACTION = 0.76
LUM_SPHERE_DROP_LSUN = 1.5
LUM_VISUAL_STACK_OFFSET_LSUN = 2.1
LUM_GAUGE_X_OFFSET_PHASE = -0.028
LUM_TEXT_X_OFFSET_PHASE = 0.006
LUM_GAUGE_HALF_HEIGHT_LSUN = 0.2964
LUM_THERMOMETER_PHASE_SHIFT = -0.05
LUM_RADIUS_THERMOMETER_PHASE_EXTRA_SHIFT = -0.02
LUM_RADIUS_THERMOMETER_DY_LSUN = 1.1
LUM_TEFF_THERMOMETER_DY_LSUN = -1.1
RV_SPHERE_Y_FRACTION = 0.77
LIGHT_ZONE_REFERENCE_COLORS = dict(COMPLEX_TRANSFER_REFERENCE_COLORS)
DARK_THEME = {
    "figure_face": "#000000",
    "axes_face": "#000000",
    "text": "#F0F0F0",
    "spine": "#CFCFCF",
    "zero_line": "#585858",
    "photosphere": "#FF8A6B",
    "phase_curve": "#B8B3AE",
    "shell_curve": NET_HEATING_COLOR,
    "radiative": "#F6BD60",
    "convective": "#73DACA",
    "pdv": PDV_COLOR,
    "marker_edge": "#FFFFFF",
    "zone_label_stroke": "#000000",
    "convection_edge_base": 0.48,
    "convection_edge_span": 0.32,
    "zone_reference_colors": {
        "He II Ionization": "#669BBC",
        "He I Ionization": "#F6BE66",
        "H Ionization": "#780000",
        "H/He I Ionization": "#780000",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Animate scaled phase-local power diagnostics versus temperature across one pulsation cycle."
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
        help="Hot-side temperature limit in K for the power panel",
    )
    parser.add_argument(
        "--coordinate",
        choices=("temperature", "r_over_R"),
        default="temperature",
        help="Horizontal coordinate for the left panel",
    )
    parser.add_argument(
        "--radius-xmin",
        type=float,
        default=None,
        help="Optional lower radius limit in solar radii for --coordinate r_over_R.",
    )
    parser.add_argument(
        "--radius-xmax",
        type=float,
        default=None,
        help="Optional upper radius limit in solar radii for --coordinate r_over_R.",
    )
    parser.add_argument(
        "--dark-mode",
        action="store_true",
        help="Render a separate black-canvas version with a phase-dependent photosphere visualization on the right panel.",
    )
    parser.add_argument(
        "--blackbody-color-file",
        type=Path,
        default=None,
        help="Optional local path to the Vendian blackbody color table. If omitted, a cached copy is downloaded on demand.",
    )
    parser.add_argument(
        "--main-terms-only",
        action="store_true",
        help="Plot only pressure-volume work and net heating, omitting the radiative and convective heating components.",
    )
    parser.add_argument(
        "--ymin",
        type=float,
        default=None,
        help="Optional lower y-axis limit for the left panel.",
    )
    parser.add_argument(
        "--ymax",
        type=float,
        default=None,
        help="Optional upper y-axis limit for the left panel.",
    )
    parser.add_argument(
        "--pdv-full-rsp-pressure",
        action="store_true",
        help="Use pressure + rsp_Pt + rsp_Pvsc in the mechanical work term.",
    )
    parser.add_argument(
        "--pdv-subtract-rsp-eq",
        action="store_true",
        help="Subtract rsp_Eq from the mechanical work term.",
    )
    parser.add_argument(
        "--pressure-work-mode",
        choices=("base", "gas_plus_pav", "full_rsp"),
        default=None,
        help="Pressure term to use in the pressure-volume work curve.",
    )
    parser.add_argument(
        "--heating-mode",
        choices=("dLdm", "dLdm_plus_eq", "gas_minus_c"),
        default="dLdm",
        help="Definition of the orange heating curve on the left panel.",
    )
    return parser.parse_args()


def normalize_unit_interval(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    normalized = np.full_like(array, np.nan, dtype=float)
    finite = np.isfinite(array)
    if not np.any(finite):
        return normalized
    vmin = float(np.nanmin(array[finite]))
    vmax = float(np.nanmax(array[finite]))
    vrange = vmax - vmin
    if vrange <= 0.0:
        normalized[finite] = 0.5
        return normalized
    normalized[finite] = (array[finite] - vmin) / vrange
    return normalized


def theme_value(dark_mode: bool, key: str, fallback: str) -> str:
    if dark_mode:
        return str(DARK_THEME[key])
    return fallback


def build_series_label(name: str, expression: str, scale_text: str = "") -> str:
    if scale_text:
        return f"{name:<{LEGEND_NAME_WIDTH}}{scale_text:>{LEGEND_SCALE_WIDTH}} {expression}"
    return f"{name:<{LEGEND_NAME_WIDTH}}  {expression}"


def pressure_work_mode_from_args(args: argparse.Namespace) -> str:
    explicit_mode = getattr(args, "pressure_work_mode", None)
    if explicit_mode is not None:
        return str(explicit_mode)
    return "full_rsp" if bool(args.pdv_full_rsp_pressure) else "base"


def mechanical_work_expression(
    *,
    pressure_work_mode: str,
    subtract_rsp_eq: bool,
) -> str:
    if pressure_work_mode == "full_rsp":
        pressure_symbol = "(P + P_t + P_av)"
    elif pressure_work_mode == "gas_plus_pav":
        pressure_symbol = "(P + P_av)"
    else:
        pressure_symbol = "P"
    expression = f"{pressure_symbol} dV/dt"
    if subtract_rsp_eq:
        expression += " - Eq"
    return expression


def mechanical_work_legend_expression(
    *,
    pressure_work_mode: str,
    subtract_rsp_eq: bool,
) -> str:
    if pressure_work_mode == "gas_plus_pav" and not subtract_rsp_eq:
        return "P dV/dT"
    return mechanical_work_expression(
        pressure_work_mode=pressure_work_mode,
        subtract_rsp_eq=subtract_rsp_eq,
    )


def heating_total_expression(heating_mode: str) -> str:
    if heating_mode == "dLdm_plus_eq":
        return "-dL/dm + Eq"
    if heating_mode == "gas_minus_c":
        return "-dL/dm - C"
    return "-dL/dm"


def heating_total_description(heating_mode: str) -> str:
    if heating_mode == "dLdm_plus_eq":
        return (
            "phase-local total-energy heating rate, -dL/dm + Eq, "
            "with positive meaning net addition to the combined gas+turbulent energy reservoir"
        )
    if heating_mode == "gas_minus_c":
        return (
            "phase-local gas heating rate, -d(Lr + Lc)/dm - C, "
            "with positive meaning net heating of the gas reservoir"
        )
    return "phase-local heating rate, -dL/dm, with positive meaning net heating of the gas"


def heating_mode_stem_suffix(heating_mode: str) -> str:
    if heating_mode == "dLdm_plus_eq":
        return "_heating_plus_eq"
    if heating_mode == "gas_minus_c":
        return "_gas_heating"
    return ""


def pressure_work_mode_stem_suffix(pressure_work_mode: str, subtract_rsp_eq: bool) -> str:
    if pressure_work_mode == "gas_plus_pav" and not subtract_rsp_eq:
        return "_pav_work"
    if pressure_work_mode == "full_rsp" and subtract_rsp_eq:
        return ""
    if pressure_work_mode == "full_rsp":
        return "_full_pressure"
    return ""


def phase_reference_positions(phase: float) -> np.ndarray:
    phase_value = float(phase)
    return np.asarray([phase_value, phase_value + 1.0], dtype=float)


def phase_moving_positions(phase: float) -> np.ndarray:
    phase_value = float(phase)
    return np.asarray([phase_value - 1.0, phase_value, phase_value + 1.0, phase_value + 2.0], dtype=float)


def repeated_phase_curve(
    phase: np.ndarray,
    values: np.ndarray,
    repeat_count: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    phase_array = np.asarray(phase, dtype=float)
    value_array = np.asarray(values, dtype=float)
    if phase_array.ndim != 1 or value_array.ndim != 1 or phase_array.size != value_array.size:
        raise ValueError("phase and values must be one-dimensional arrays of the same length.")

    x_segments: list[np.ndarray] = []
    y_segments: list[np.ndarray] = []
    separator = np.asarray([np.nan], dtype=float)
    for offset in range(int(repeat_count)):
        if offset > 0:
            x_segments.append(separator)
            y_segments.append(separator)
        x_segments.append(phase_array + float(offset))
        y_segments.append(value_array)
    return np.concatenate(x_segments), np.concatenate(y_segments)


def repeated_phase_colors(
    colors: np.ndarray,
    repeat_count: int = 3,
) -> np.ndarray:
    color_array = np.asarray(colors, dtype=float)
    if color_array.ndim != 2 or color_array.shape[1] != 3:
        raise ValueError("colors must be a two-dimensional RGB array.")

    segments: list[np.ndarray] = []
    separator = np.full((1, 3), np.nan, dtype=float)
    for offset in range(int(repeat_count)):
        if offset > 0:
            segments.append(separator)
        segments.append(color_array)
    return np.vstack(segments)


def q_cell_widths(q_values: np.ndarray) -> np.ndarray:
    q_array = np.asarray(q_values, dtype=float)
    if q_array.ndim != 1 or q_array.size == 0:
        return np.zeros_like(q_array, dtype=float)
    if q_array.size == 1:
        return np.ones_like(q_array, dtype=float)
    widths = np.empty_like(q_array, dtype=float)
    widths[0] = abs(float(q_array[1] - q_array[0]))
    widths[-1] = abs(float(q_array[-1] - q_array[-2]))
    if q_array.size > 2:
        widths[1:-1] = 0.5 * np.abs(q_array[2:] - q_array[:-2])
    return np.clip(widths, 1.0e-30, None)


def high_opacity_region_state(frame: dict[str, object]) -> tuple[float, float, float]:
    q_plot = np.asarray(frame["q_plot"], dtype=float)
    radius_plot = np.asarray(frame["radius_rsun_plot"], dtype=float)
    temperature_plot = np.asarray(frame["temperature_plot"], dtype=float)
    opacity_plot = np.asarray(frame["opacity_plot"], dtype=float)
    finite = (
        np.isfinite(q_plot)
        & np.isfinite(radius_plot)
        & np.isfinite(temperature_plot)
        & np.isfinite(opacity_plot)
    )
    if np.count_nonzero(finite) == 0:
        return (
            float(frame["photosphere_q_env"]) if "photosphere_q_env" in frame else float(np.nan),
            float(frame["photosphere_radius_rsun"]),
            float(frame["photosphere_temperature_K"]),
        )

    q_finite = q_plot[finite]
    radius_finite = radius_plot[finite]
    temperature_finite = temperature_plot[finite]
    opacity_finite = opacity_plot[finite]
    max_opacity = float(np.nanmax(opacity_finite))
    region_mask = opacity_finite >= HIGH_OPACITY_REGION_THRESHOLD * max_opacity
    if np.count_nonzero(region_mask) < 2:
        top_n = min(int(HIGH_OPACITY_REGION_FALLBACK_TOP_N), opacity_finite.size)
        top_indices = np.argsort(opacity_finite)[-top_n:]
        region_mask = np.zeros_like(opacity_finite, dtype=bool)
        region_mask[top_indices] = True

    q_region = q_finite[region_mask]
    radius_region = radius_finite[region_mask]
    temperature_region = temperature_finite[region_mask]
    opacity_region = np.clip(opacity_finite[region_mask], 0.0, None)
    q_width_region = q_cell_widths(q_finite)[region_mask]
    weights = opacity_region * q_width_region
    if not np.any(np.isfinite(weights)) or float(np.nansum(weights)) <= 0.0:
        weights = np.ones_like(q_region, dtype=float)

    return (
        float(np.average(q_region, weights=weights)),
        float(np.average(radius_region, weights=weights)),
        float(np.average(temperature_region, weights=weights)),
    )


def scaled_font(size: float) -> float:
    return float(size) * TEXT_SCALE_FACTOR


def photosphere_marker_size(radius_scale: float) -> float:
    scale = max(float(radius_scale), 1.0e-6)
    return float(PHOTOSPHERE_MARKER_BASE_SIZE * scale)


def rounded_temperature_text(temperature_K: float) -> str:
    rounded_temperature = int(100.0 * round(float(temperature_K) / 100.0))
    return f"{rounded_temperature:,.0f} K"


def radial_velocity_palette_rgb(velocity_km_per_s: float, velocity_scale_km_per_s: float) -> np.ndarray:
    neutral = np.asarray([0.24, 0.24, 0.27], dtype=float)
    red = np.asarray([0.92, 0.28, 0.24], dtype=float)
    blue = np.asarray([0.28, 0.50, 0.92], dtype=float)
    scale = max(float(velocity_scale_km_per_s), 1.0e-12)
    signed_unit = float(np.clip(float(velocity_km_per_s) / scale, -1.0, 1.0))
    target = blue if signed_unit >= 0.0 else red
    mix = abs(signed_unit)
    return (1.0 - mix) * neutral + mix * target


def theme_zone_colors(dark_mode: bool) -> dict[str, str]:
    if dark_mode:
        return dict(DARK_THEME["zone_reference_colors"])
    return dict(LIGHT_ZONE_REFERENCE_COLORS)


def finite_global_bounds(frame_data: list[dict[str, object]], field_name: str) -> tuple[float, float]:
    finite_values: list[np.ndarray] = []
    for frame in frame_data:
        values = np.asarray(frame[field_name], dtype=float)
        finite = np.isfinite(values)
        if np.any(finite):
            finite_values.append(values[finite])
    if not finite_values:
        return 0.0, 1.0
    concatenated = np.concatenate(finite_values)
    data_min = float(np.nanmin(concatenated))
    data_max = float(np.nanmax(concatenated))
    if not np.isfinite(data_min) or not np.isfinite(data_max):
        return 0.0, 1.0
    if data_max <= data_min:
        return data_min - 0.5, data_max + 0.5
    return data_min, data_max


def visible_window_mask(frame: dict[str, object], x_field_name: str, x_limits: tuple[float, float]) -> np.ndarray:
    x_values = np.asarray(frame[x_field_name], dtype=float)
    x0 = float(min(x_limits))
    x1 = float(max(x_limits))
    return np.isfinite(x_values) & (x_values >= x0) & (x_values <= x1)


def finite_visible_bounds(
    frame_data: list[dict[str, object]],
    field_name: str,
    x_field_name: str,
    x_limits: tuple[float, float],
) -> tuple[float, float]:
    finite_values: list[np.ndarray] = []
    for frame in frame_data:
        values = np.asarray(frame[field_name], dtype=float)
        mask = visible_window_mask(frame, x_field_name, x_limits) & np.isfinite(values)
        if np.any(mask):
            finite_values.append(values[mask])
    if not finite_values:
        return finite_global_bounds(frame_data, field_name)
    concatenated = np.concatenate(finite_values)
    data_min = float(np.nanmin(concatenated))
    data_max = float(np.nanmax(concatenated))
    if not np.isfinite(data_min) or not np.isfinite(data_max):
        return finite_global_bounds(frame_data, field_name)
    if data_max <= data_min:
        return data_min - 0.5, data_max + 0.5
    return data_min, data_max


def visible_power_series_names(main_terms_only: bool) -> tuple[str, ...]:
    if main_terms_only:
        return ("pdv_power", "heating_total")
    return ("pdv_power", "heating_total", "heating_radiative", "heating_convective")


def finite_visible_scaled_power_bounds(
    frame_data: list[dict[str, object]],
    series_names: tuple[str, ...],
    x_field_name: str,
    x_limits: tuple[float, float],
) -> tuple[float, float]:
    finite_values: list[np.ndarray] = []
    for frame in frame_data:
        mask = visible_window_mask(frame, x_field_name, x_limits)
        for series_name in series_names:
            values = np.asarray(frame[f"{series_name}_plot"], dtype=float)
            series_mask = mask & np.isfinite(values)
            if np.any(series_mask):
                scale = float(PLOT_SCALE_FACTORS[series_name]) / DISPLAY_POWER_SCALE
                finite_values.append(values[series_mask] * scale)
    if not finite_values:
        return -1.0, 1.0
    concatenated = np.concatenate(finite_values)
    data_min = float(np.nanmin(concatenated))
    data_max = float(np.nanmax(concatenated))
    if not np.isfinite(data_min) or not np.isfinite(data_max):
        return -1.0, 1.0
    if data_max <= data_min:
        pad = max(abs(data_min), POWER_PANEL_MIN_HALF_RANGE) * POWER_PANEL_PAD_FRACTION
        return data_min - pad, data_max + pad
    return data_min, data_max


def symmetric_power_panel_limits(data_bounds: tuple[float, float]) -> tuple[float, float]:
    max_abs = max(abs(float(data_bounds[0])), abs(float(data_bounds[1])), POWER_PANEL_MIN_HALF_RANGE)
    half_range = max_abs * (1.0 + POWER_PANEL_PAD_FRACTION)
    return -float(half_range), float(half_range)


def power_panel_limits_from_visible_bounds(data_bounds: tuple[float, float]) -> tuple[float, float]:
    return symmetric_power_panel_limits(data_bounds)


def opacity_display_scale(
    opacity_bounds: tuple[float, float],
    panel_limits: tuple[float, float],
) -> float:
    opacity_span = float(opacity_bounds[1]) - float(opacity_bounds[0])
    if not np.isfinite(opacity_span) or opacity_span <= 0.0:
        return 0.0
    panel_top = max(float(panel_limits[1]), 0.0)
    if panel_top <= 0.0:
        return 0.0
    return float(OPACITY_PANEL_TOP_FRACTION * panel_top / opacity_span)


def scale_diagnostic_to_panel(
    values: np.ndarray,
    data_bounds: tuple[float, float],
    panel_limits: tuple[float, float],
    pad_fraction: float = SCALED_DIAGNOSTIC_PAD_FRACTION,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    data_min = float(data_bounds[0])
    data_max = float(data_bounds[1])
    y_min = float(panel_limits[0])
    y_max = float(panel_limits[1])
    y_span = y_max - y_min
    if y_span <= 0.0:
        return np.full_like(array, np.nan, dtype=float)
    lower = y_min + float(pad_fraction) * y_span
    upper = y_max - float(pad_fraction) * y_span
    if data_max <= data_min:
        midpoint = 0.5 * (lower + upper)
        scaled = np.full_like(array, midpoint, dtype=float)
        scaled[~np.isfinite(array)] = np.nan
        return scaled
    scaled = lower + (array - data_min) * (upper - lower) / (data_max - data_min)
    scaled[~np.isfinite(array)] = np.nan
    return scaled


def clamp(value: float, lower: float, upper: float) -> float:
    return float(min(max(float(value), float(lower)), float(upper)))


def interpolate_on_coordinate(
    coordinate_values: np.ndarray,
    signal_values: np.ndarray,
    target_coordinate: float,
) -> float:
    coordinate = np.asarray(coordinate_values, dtype=float)
    signal = np.asarray(signal_values, dtype=float)
    finite = np.isfinite(coordinate) & np.isfinite(signal)
    if np.count_nonzero(finite) < 2:
        return float("nan")
    coordinate_finite = coordinate[finite]
    signal_finite = signal[finite]
    order = np.argsort(coordinate_finite)
    coordinate_sorted = coordinate_finite[order]
    signal_sorted = signal_finite[order]
    target = clamp(float(target_coordinate), float(coordinate_sorted[0]), float(coordinate_sorted[-1]))
    return float(np.interp(target, coordinate_sorted, signal_sorted))


def average_over_spans(
    coordinate_values: np.ndarray,
    signal_values: np.ndarray,
    spans: list[tuple[float, float]],
) -> float:
    coordinate = np.asarray(coordinate_values, dtype=float)
    signal = np.asarray(signal_values, dtype=float)
    finite = np.isfinite(coordinate) & np.isfinite(signal)
    if not spans or np.count_nonzero(finite) == 0:
        return float("nan")

    span_mask = np.zeros_like(coordinate, dtype=bool)
    for x0, x1 in spans:
        left = float(min(x0, x1))
        right = float(max(x0, x1))
        span_mask |= (coordinate >= left) & (coordinate <= right)

    selected = signal[finite & span_mask]
    if selected.size == 0:
        return float("nan")
    return float(np.nanmean(selected))


def style_axis_for_theme(ax: plt.Axes, dark_mode: bool) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_LINE_WIDTH)
    ax.tick_params(
        which="major",
        direction="out",
        top=False,
        right=False,
        length=5.0,
        width=AXIS_LINE_WIDTH,
    )
    ax.tick_params(
        which="minor",
        direction="out",
        top=False,
        right=False,
        length=3.0,
        width=0.9 * AXIS_LINE_WIDTH,
    )
    if not dark_mode:
        return
    ax.set_facecolor(str(DARK_THEME["axes_face"]))
    ax.tick_params(colors=str(DARK_THEME["text"]), which="both")
    ax.xaxis.label.set_color(str(DARK_THEME["text"]))
    ax.yaxis.label.set_color(str(DARK_THEME["text"]))
    for spine in ax.spines.values():
        spine.set_color(str(DARK_THEME["spine"]))


def configure_linear_axis_ticks(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(MaxNLocator(nbins=TARGET_MAJOR_TICKS))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=TARGET_MAJOR_TICKS))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))


def style_legend_for_theme(legend: plt.Legend | None, dark_mode: bool) -> None:
    if legend is None or not dark_mode:
        return
    for text in legend.get_texts():
        text.set_color(str(DARK_THEME["text"]))
    legend.get_title().set_color(str(DARK_THEME["text"]))


def ensure_blackbody_color_file(path: Path) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(BLACKBODY_COLOR_URL, timeout=30) as response:
        contents = response.read()
    path.write_bytes(contents)
    return path


def load_blackbody_color_table(path: Path | None) -> dict[str, np.ndarray]:
    table_path = ensure_blackbody_color_file(path or BLACKBODY_CACHE_PATH)
    color_rows_by_temperature: dict[float, tuple[float, float, float]] = {}
    for raw_line in table_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 13 or parts[1] != "K" or parts[2] != BLACKBODY_TABLE_CMF:
            continue
        color_rows_by_temperature[float(parts[0])] = (
            float(parts[6]),
            float(parts[7]),
            float(parts[8]),
        )
    for temperature_K, rgb255 in BLACKBODY_HIGH_T_ANCHORS_RGB255:
        color_rows_by_temperature[float(temperature_K)] = tuple(
            float(channel) / 255.0 for channel in rgb255
        )
    if not color_rows_by_temperature:
        raise RuntimeError(f"No blackbody color rows were parsed from {table_path}")
    sorted_temperatures = np.asarray(sorted(color_rows_by_temperature.keys()), dtype=float)
    rgb_rows = np.asarray([color_rows_by_temperature[float(temperature)] for temperature in sorted_temperatures], dtype=float)
    return {
        "path": np.asarray([str(table_path)], dtype=object),
        "temperature_K": sorted_temperatures,
        "rgb": rgb_rows,
    }


def interpolate_blackbody_rgb(
    temperature_K: float,
    color_table: dict[str, np.ndarray],
) -> np.ndarray:
    temperatures = np.asarray(color_table["temperature_K"], dtype=float)
    rgb_table = np.asarray(color_table["rgb"], dtype=float)
    temperature = float(np.clip(float(temperature_K), float(temperatures[0]), float(temperatures[-1])))
    return np.asarray(
        [np.interp(temperature, temperatures, rgb_table[:, channel]) for channel in range(3)],
        dtype=float,
    )


def infer_feh_from_initial_z(initial_z: float, solar_z_reference: float = SOLAR_Z_REFERENCE) -> float:
    z_value = max(float(initial_z), 1.0e-12)
    return float(math.log10(z_value / max(float(solar_z_reference), 1.0e-12)))


def planck_lambda_cgs(wavelength_cm: np.ndarray, temperature_K: float) -> np.ndarray:
    wavelength = np.asarray(wavelength_cm, dtype=float)
    temperature = max(float(temperature_K), 1.0)
    exponent = np.clip((H_CGS * C_CGS) / (wavelength * K_B_CGS * temperature), 1.0e-9, 700.0)
    numerator = 2.0 * H_CGS * C_CGS**2 / np.power(wavelength, 5)
    return numerator / np.expm1(exponent)


def v_band_flux_proxy(temperature_K: float, radius_rsun: float) -> float:
    spectral_radiance = planck_lambda_cgs(V_BAND_WAVELENGTHS_CM, float(temperature_K))
    band_flux = float(np.trapezoid(spectral_radiance * V_BAND_RESPONSE, V_BAND_WAVELENGTHS_CM))
    return float(max(radius_rsun, 0.0) ** 2 * band_flux)


def quadratic_limb_darkening_coefficients(
    temperature_K: float,
    logg_cgs: float,
    feh: float,
) -> tuple[float, float]:
    teff_term = np.clip((float(temperature_K) - 6500.0) / 2000.0, -1.2, 1.2)
    logg_term = np.clip(float(logg_cgs) - 2.8, -1.2, 1.2)
    feh_term = np.clip(float(feh), -2.5, 0.5)
    a_coeff = float(np.clip(0.55 - 0.12 * teff_term - 0.04 * logg_term + 0.02 * feh_term, 0.22, 0.82))
    b_coeff = float(np.clip(0.20 - 0.05 * teff_term - 0.02 * logg_term + 0.01 * feh_term, 0.02, 0.32))
    return a_coeff, b_coeff


def render_photosphere_sphere_rgba(
    rgb: np.ndarray,
    brightness_scale: float,
    limb_darkening_coefficients: tuple[float, float],
    size: int = SPHERE_IMAGE_SIZE,
) -> np.ndarray:
    axis = np.linspace(-1.0, 1.0, int(size), dtype=float)
    x_grid, y_grid = np.meshgrid(axis, axis)
    radius_squared = x_grid**2 + y_grid**2
    inside = radius_squared <= 1.0
    mu = np.zeros_like(x_grid)
    mu[inside] = np.sqrt(np.clip(1.0 - radius_squared[inside], 0.0, 1.0))

    a_coeff, b_coeff = limb_darkening_coefficients
    intensity_profile = np.zeros_like(x_grid)
    intensity_profile[inside] = (
        1.0
        - a_coeff * (1.0 - mu[inside])
        - b_coeff * np.square(1.0 - mu[inside])
    )
    intensity_profile = np.clip(intensity_profile, 0.0, None)
    intensity_profile *= float(brightness_scale)

    edge_radius = np.sqrt(np.clip(radius_squared, 0.0, None))
    edge_alpha = np.clip((1.0 - edge_radius) / SPHERE_EDGE_SOFTENING, 0.0, 1.0)

    rgba = np.zeros((int(size), int(size), 4), dtype=float)
    sphere_rgb = np.clip(np.asarray(rgb, dtype=float), 0.0, 1.0)
    rgba[..., :3] = sphere_rgb[None, None, :] * intensity_profile[..., None]
    rgba[..., 3] = edge_alpha * inside
    return rgba


def add_colored_phase_curve(
    ax: plt.Axes,
    x_values: np.ndarray,
    y_values: np.ndarray,
    point_rgb: np.ndarray,
    linewidth: float,
    zorder: float,
) -> LineCollection | None:
    x_array = np.asarray(x_values, dtype=float)
    y_array = np.asarray(y_values, dtype=float)
    rgb_points = np.asarray(point_rgb, dtype=float)
    finite = np.isfinite(x_array) & np.isfinite(y_array)
    if x_array.ndim != 1 or y_array.ndim != 1 or x_array.size != y_array.size or x_array.size < 2:
        return None
    if rgb_points.ndim != 2 or rgb_points.shape[0] != x_array.size or rgb_points.shape[1] != 3:
        return None
    if np.count_nonzero(finite) < 2:
        return None

    points = np.column_stack([x_array, y_array]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    segment_mask = finite[:-1] & finite[1:]
    if not np.any(segment_mask):
        return None
    segment_colors = 0.5 * (rgb_points[:-1] + rgb_points[1:])
    collection = LineCollection(
        segments[segment_mask],
        colors=np.clip(segment_colors[segment_mask], 0.0, 1.0),
        linewidths=float(linewidth),
        zorder=float(zorder),
        capstyle="round",
        joinstyle="round",
    )
    ax.add_collection(collection)
    return collection


def phase_order_age_wrap_breaks(
    cycle_records: list[dict[str, object]],
    phase_sorted: np.ndarray,
    period_days: float,
) -> list[float]:
    if len(cycle_records) < 2:
        return []
    phase = np.asarray(phase_sorted, dtype=float)
    ages = np.asarray([float(record["age_days"]) for record in cycle_records], dtype=float)
    if phase.size != ages.size:
        return []
    age_drops = np.flatnonzero(np.diff(ages) < -0.5 * float(period_days))
    breaks: list[float] = []
    for index in age_drops:
        left_phase = float(phase[int(index)])
        right_phase = float(phase[int(index) + 1])
        if right_phase > left_phase:
            breaks.append(0.5 * (left_phase + right_phase))
    return breaks


def add_phase_curve_breaks(
    x_values: np.ndarray,
    y_values: np.ndarray,
    break_phases: list[float],
    point_rgb: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    x_array = np.asarray(x_values, dtype=float)
    y_array = np.asarray(y_values, dtype=float)
    rgb_array = None if point_rgb is None else np.asarray(point_rgb, dtype=float)
    if not break_phases or x_array.size < 2:
        return x_array, y_array, rgb_array

    x_min = float(np.nanmin(x_array))
    x_max = float(np.nanmax(x_array))
    repeated_breaks: list[float] = []
    for phase in break_phases:
        phase_value = float(phase) % 1.0
        for offset in range(int(math.floor(x_min)) - 1, int(math.ceil(x_max)) + 2):
            break_position = phase_value + float(offset)
            if x_min < break_position < x_max:
                repeated_breaks.append(break_position)
    repeated_breaks = sorted(set(round(value, 12) for value in repeated_breaks))
    if not repeated_breaks:
        return x_array, y_array, rgb_array

    x_out: list[float] = []
    y_out: list[float] = []
    rgb_out: list[np.ndarray] = []
    break_index = 0
    for index in range(x_array.size):
        x_out.append(float(x_array[index]))
        y_out.append(float(y_array[index]))
        if rgb_array is not None:
            rgb_out.append(np.asarray(rgb_array[index], dtype=float))
        if index >= x_array.size - 1:
            continue
        x_left = float(x_array[index])
        x_right = float(x_array[index + 1])
        while break_index < len(repeated_breaks) and repeated_breaks[break_index] <= x_left:
            break_index += 1
        if break_index < len(repeated_breaks) and x_left < repeated_breaks[break_index] < x_right:
            x_out.append(float("nan"))
            y_out.append(float("nan"))
            if rgb_array is not None:
                rgb_out.append(np.full(3, np.nan, dtype=float))

    return (
        np.asarray(x_out, dtype=float),
        np.asarray(y_out, dtype=float),
        np.asarray(rgb_out, dtype=float) if rgb_array is not None else None,
    )


def add_photosphere_exterior_glow(
    ax: plt.Axes,
    radius_rsun_profile: np.ndarray,
    tau_profile: np.ndarray,
    photosphere_radius_rsun: float,
    photosphere_tau: float,
    x_right: float,
    y_limits: tuple[float, float],
    rgb: np.ndarray,
    brightness_scale: float,
) -> None:
    photosphere_radius = float(photosphere_radius_rsun)
    x_max = float(x_right)
    y_min = float(y_limits[0])
    y_max = float(y_limits[1])
    if not np.isfinite(photosphere_radius) or not np.isfinite(x_max) or x_max <= photosphere_radius:
        return
    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_max <= y_min:
        return

    sample_radii = np.linspace(photosphere_radius, x_max, EXTERIOR_GLOW_SAMPLES_X, dtype=float)
    radius_ratio = np.divide(
        photosphere_radius,
        np.clip(sample_radii, max(photosphere_radius, 1.0e-12), None),
        out=np.zeros_like(sample_radii),
        where=sample_radii > 0.0,
    )
    alpha_profile = (
        EXTERIOR_GLOW_MAX_ALPHA
        * float(brightness_scale)
        * np.power(radius_ratio, EXTERIOR_GLOW_FALLOFF_POWER)
    )
    rgba = np.zeros((EXTERIOR_GLOW_SAMPLES_Y, EXTERIOR_GLOW_SAMPLES_X, 4), dtype=float)
    rgba[..., :3] = np.clip(np.asarray(rgb, dtype=float), 0.0, 1.0)[None, None, :]
    rgba[..., 3] = alpha_profile[None, :]
    ax.imshow(
        rgba,
        extent=(photosphere_radius, x_max, y_min, y_max),
        origin="lower",
        aspect="auto",
        interpolation="bicubic",
        zorder=-9,
    )


def add_interior_temperature_shading(
    ax: plt.Axes,
    radius_rsun_profile: np.ndarray,
    temperature_profile_K: np.ndarray,
    tau_profile: np.ndarray,
    photosphere_radius_rsun: float,
    photosphere_tau: float,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    color_table: dict[str, np.ndarray] | None,
    dark_mode: bool,
    luminosity_unit: float = 1.0,
) -> None:
    if color_table is None:
        return

    radius = np.asarray(radius_rsun_profile, dtype=float)
    temperature = np.asarray(temperature_profile_K, dtype=float)
    tau = np.asarray(tau_profile, dtype=float)
    finite = np.isfinite(radius) & np.isfinite(temperature) & (temperature > 0.0) & np.isfinite(tau) & (tau >= 0.0)
    if np.count_nonzero(finite) < 2:
        return

    radius_sorted = radius[finite]
    temperature_sorted = temperature[finite]
    tau_sorted = tau[finite]
    order = np.argsort(radius_sorted)
    radius_sorted = radius_sorted[order]
    temperature_sorted = temperature_sorted[order]
    tau_sorted = tau_sorted[order]
    unique_radius, unique_indices = np.unique(radius_sorted, return_index=True)
    if unique_radius.size < 2:
        return
    temperature_unique = temperature_sorted[unique_indices]
    tau_unique = tau_sorted[unique_indices]

    x_left = max(float(x_limits[0]), float(unique_radius[0]))
    x_right = min(float(photosphere_radius_rsun), float(unique_radius[-1]))
    y_min = float(y_limits[0])
    y_max = float(y_limits[1])
    if not np.isfinite(x_left) or not np.isfinite(x_right) or x_right <= x_left:
        return
    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_max <= y_min:
        return

    sample_radii = np.linspace(x_left, x_right, INTERIOR_SHADE_SAMPLES_X, dtype=float)
    sample_temperatures = np.interp(sample_radii, unique_radius, temperature_unique)
    sample_tau = np.interp(sample_radii, unique_radius, tau_unique)
    log_temperature = np.log10(np.clip(sample_temperatures, 1.0, None))
    log_min = float(np.nanmin(log_temperature))
    log_max = float(np.nanmax(log_temperature))
    if log_max > log_min:
        normalized = (log_temperature - log_min) / (log_max - log_min)
    else:
        normalized = np.full_like(log_temperature, 0.5, dtype=float)

    rgb_profile = np.vstack(
        [interpolate_blackbody_rgb(float(temp), color_table) for temp in sample_temperatures]
    )
    alpha_base = INTERIOR_SHADE_ALPHA_DARK if dark_mode else INTERIOR_SHADE_ALPHA_LIGHT
    luminosity_scale = float(
        INTERIOR_SHADE_LUMINOSITY_FLOOR
        + (1.0 - INTERIOR_SHADE_LUMINOSITY_FLOOR) * np.clip(float(luminosity_unit), 0.0, 1.0)
    )
    optical_depth_below_photosphere = np.clip(sample_tau - float(photosphere_tau), 0.0, None)
    alpha_profile = (
        alpha_base
        * luminosity_scale
        * (0.45 + 0.55 * np.power(np.clip(normalized, 0.0, 1.0), 0.8))
        * np.exp(-optical_depth_below_photosphere)
    )

    rgba = np.zeros((INTERIOR_SHADE_SAMPLES_Y, INTERIOR_SHADE_SAMPLES_X, 4), dtype=float)
    rgba[..., :3] = np.clip(rgb_profile, 0.0, 1.0)[None, :, :]
    rgba[..., 3] = alpha_profile[None, :]
    ax.imshow(
        rgba,
        extent=(x_left, x_right, y_min, y_max),
        origin="lower",
        aspect="auto",
        interpolation="bicubic",
        zorder=-10,
    )


def add_zone_overlays_temperature(
    ax: plt.Axes,
    zone_spans: dict[str, list[tuple[float, float]]],
    label_positions: dict[str, tuple[float, str]],
    convection_profile: dict[str, object],
    dark_mode: bool = False,
) -> None:
    text_transform = blended_transform_factory(ax.transData, ax.transAxes)
    convection_temperature = np.asarray(convection_profile["temperature_K"], dtype=float)
    convection_strength = np.asarray(convection_profile["strength"], dtype=float)
    convection_strength_max = max(float(convection_profile["max_strength"]), 1.0e-12)
    reference_colors = theme_zone_colors(dark_mode)
    stroke_color = theme_value(dark_mode, "zone_label_stroke", "white")

    for x0, x1 in zone_spans.get("Convection", []):
        left = float(min(x0, x1))
        right = float(max(x0, x1))
        in_span = (
            np.isfinite(convection_temperature)
            & np.isfinite(convection_strength)
            & (convection_temperature >= left)
            & (convection_temperature <= right)
        )
        span_strength = (
            float(np.nanmean(convection_strength[in_span]) / convection_strength_max)
            if np.any(in_span)
            else 0.0
        )
        span_strength = min(max(span_strength, 0.0), 1.0)
        if dark_mode:
            edge_gray = float(DARK_THEME["convection_edge_base"]) + float(DARK_THEME["convection_edge_span"]) * span_strength
            edgecolor = (edge_gray, edge_gray, edge_gray, 0.32 + 0.16 * span_strength)
            facecolor = (1.0, 1.0, 1.0, 0.0)
        else:
            hatch_gray = 0.95 - 0.18 * span_strength
            edgecolor = (hatch_gray, hatch_gray, hatch_gray, 0.72)
            facecolor = (0.75, 0.75, 0.75, 0.012)
        ax.axvspan(
            x0,
            x1,
            facecolor=facecolor,
            edgecolor=edgecolor,
            hatch="/",
            linewidth=0.0,
            zorder=-6,
        )

    for name, spans in zone_spans.items():
        if name == "Convection" or not spans:
            continue
        color = reference_colors.get(name, "0.45")
        for x0, x1 in spans:
            ax.axvspan(
                x0,
                x1,
                facecolor=color,
                edgecolor="none",
                linewidth=0.0,
                alpha=0.26 if not dark_mode else 0.32,
                zorder=-4,
            )
    for name, (label_x, label_text) in label_positions.items():
        color = theme_value(dark_mode, "text", reference_colors.get(name, ZONE_LABEL_COLOR))
        ax.text(
            label_x,
            0.98,
            label_text,
            rotation=90,
            ha="center",
            va="top",
            fontsize=scaled_font(ZONE_LABEL_FONT_SIZE),
            color=color,
            transform=text_transform,
            path_effects=[pe.withStroke(linewidth=ZONE_LABEL_STROKE_WIDTH, foreground=stroke_color)],
        )


def add_photosphere_visual_state(
    frame_data: list[dict[str, object]],
    blackbody_color_table: dict[str, np.ndarray],
) -> dict[str, object]:
    if not frame_data:
        return {}

    initial_z = float(frame_data[0].get("initial_z", SOLAR_Z_REFERENCE))
    feh = infer_feh_from_initial_z(initial_z)
    radius_series = np.asarray([float(frame["photosphere_radius_rsun"]) for frame in frame_data], dtype=float)
    temperature_series = np.asarray([float(frame["photosphere_temperature_K"]) for frame in frame_data], dtype=float)
    logg_series = np.asarray([float(frame["photosphere_logg_cgs"]) for frame in frame_data], dtype=float)
    velocity_series = np.asarray([float(frame["photosphere_velocity_km_per_s"]) for frame in frame_data], dtype=float)
    v_flux_series = np.asarray(
        [v_band_flux_proxy(temperature, radius) for temperature, radius in zip(temperature_series, radius_series)],
        dtype=float,
    )
    normalized_flux = normalize_unit_interval(v_flux_series)
    normalized_radius = normalize_unit_interval(radius_series)
    normalized_temperature = normalize_unit_interval(temperature_series)
    radius_reference = float(np.nanmean(radius_series[np.isfinite(radius_series)]))
    radius_reference = max(radius_reference, 1.0e-12)
    flux_median = float(np.nanmedian(v_flux_series[np.isfinite(v_flux_series)]))
    flux_median = max(flux_median, 1.0e-30)
    velocity_scale = max(float(np.nanmax(np.abs(velocity_series))), 1.0e-6)

    for frame, temperature, radius, logg, velocity, v_flux, brightness_unit, radius_unit, temperature_unit in zip(
        frame_data,
        temperature_series,
        radius_series,
        logg_series,
        velocity_series,
        v_flux_series,
        normalized_flux,
        normalized_radius,
        normalized_temperature,
    ):
        rgb = interpolate_blackbody_rgb(float(temperature), blackbody_color_table)
        limb_darkening = quadratic_limb_darkening_coefficients(float(temperature), float(logg), feh)
        brightness_scale = 0.28 + 0.72 * float(np.nan_to_num(brightness_unit, nan=0.5))
        relative_v_mag = float(-2.5 * math.log10(max(float(v_flux) / flux_median, 1.0e-30)))
        frame["photosphere_blackbody_rgb"] = rgb
        frame["photosphere_luminosity_sphere_rgb"] = rgb
        frame["photosphere_rv_sphere_rgb"] = radial_velocity_palette_rgb(float(velocity), velocity_scale)
        frame["photosphere_limb_darkening"] = limb_darkening
        frame["photosphere_v_flux_proxy"] = float(v_flux)
        frame["photosphere_v_relative_mag"] = relative_v_mag
        frame["photosphere_luminosity_sphere_brightness"] = brightness_scale
        frame["photosphere_marker_rgb"] = rgb
        frame["photosphere_radius_axis_unit"] = float(np.nan_to_num(radius_unit, nan=0.5))
        frame["photosphere_radius_scale"] = float(radius) / radius_reference
        frame["photosphere_temperature_axis_unit"] = float(np.nan_to_num(temperature_unit, nan=0.5))
        frame["photosphere_marker_size_pts"] = photosphere_marker_size(frame["photosphere_radius_scale"])
        frame["photosphere_sphere_zoom"] = SPHERE_BASE_ZOOM * float(frame["photosphere_radius_scale"])

    return {
        "blackbody_color_file": str(np.asarray(blackbody_color_table["path"], dtype=object)[0]),
        "blackbody_color_url": BLACKBODY_COLOR_URL,
        "feh": float(feh),
        "sphere_center_axes_fraction_luminosity": [float(SPHERE_CENTER_LUM_PANEL[0]), float(SPHERE_CENTER_LUM_PANEL[1])],
        "sphere_center_axes_fraction_rv": [float(SPHERE_CENTER_RV_PANEL[0]), float(SPHERE_CENTER_RV_PANEL[1])],
        "sphere_radius_reference_rsun": float(radius_reference),
        "sphere_radius_min_rsun": float(np.nanmin(radius_series)),
        "sphere_radius_max_rsun": float(np.nanmax(radius_series)),
        "v_flux_proxy_min": float(np.nanmin(v_flux_series)),
        "v_flux_proxy_max": float(np.nanmax(v_flux_series)),
        "photosphere_velocity_min_km_per_s": float(np.nanmin(velocity_series)),
        "photosphere_velocity_max_km_per_s": float(np.nanmax(velocity_series)),
        "photosphere_velocity_color_scale_km_per_s": float(velocity_scale),
        "rv_sphere_brightness": float(np.nanmedian(0.28 + 0.72 * np.nan_to_num(normalized_flux, nan=0.5))),
    }


def interpolate_on_q(q_sorted: np.ndarray, values_sorted: np.ndarray, target_q: float) -> float:
    q = np.asarray(q_sorted, dtype=float)
    values = np.asarray(values_sorted, dtype=float)
    if q.size == 0:
        return float("nan")
    target = float(np.clip(float(target_q), float(q[0]), float(q[-1])))
    return float(np.interp(target, q, values))


def column_or_zeros(
    columns: dict[str, np.ndarray],
    name: str,
    reference: np.ndarray,
    *,
    required: bool = False,
) -> np.ndarray:
    if name in columns:
        values = np.asarray(columns[name], dtype=float)
        if values.shape != np.asarray(reference, dtype=float).shape:
            raise RuntimeError(f"Column {name} does not match the reference zone shape.")
        return values
    if required:
        raise RuntimeError(f"Requested column {name} is missing from the profile data.")
    return np.zeros_like(np.asarray(reference, dtype=float), dtype=float)


def effective_rsp_pressure(
    columns: dict[str, np.ndarray],
    *,
    pressure_work_mode: str,
) -> np.ndarray:
    base_pressure = np.asarray(columns["pressure"], dtype=float)
    if pressure_work_mode == "base":
        return base_pressure
    pressure_terms = base_pressure.copy()
    pressure_terms += column_or_zeros(columns, "rsp_Pvsc", base_pressure, required=True)
    if pressure_work_mode == "full_rsp":
        pressure_terms += column_or_zeros(columns, "rsp_Pt", base_pressure, required=True)
    return pressure_terms


def effective_rsp_eq(
    columns: dict[str, np.ndarray],
    reference: np.ndarray,
    *,
    subtract_rsp_eq: bool,
) -> np.ndarray:
    return column_or_zeros(columns, "rsp_Eq", reference, required=subtract_rsp_eq) if subtract_rsp_eq else np.zeros_like(
        np.asarray(reference, dtype=float),
        dtype=float,
    )


def phase_local_pdv_rate_surface_order(
    interpolated_profile: dict[str, object],
    period_days: float,
    *,
    pressure_work_mode: str = "base",
    subtract_rsp_eq: bool = False,
) -> np.ndarray:
    current_columns = interpolated_profile["columns"]
    left_columns = interpolated_profile["left_columns"]
    right_columns = interpolated_profile["right_columns"]

    q_current = np.asarray(current_columns["q"], dtype=float)
    if not np.array_equal(q_current, np.asarray(left_columns["q"], dtype=float)):
        raise RuntimeError("Left profile q grid does not match the interpolated profile q grid.")
    if not np.array_equal(q_current, np.asarray(right_columns["q"], dtype=float)):
        raise RuntimeError("Right profile q grid does not match the interpolated profile q grid.")

    left_age_days = float(interpolated_profile["left_age_days"])
    right_age_days = float(interpolated_profile["right_age_days"])
    if right_age_days <= left_age_days:
        right_age_days += float(period_days)
    delta_time_sec = max((right_age_days - left_age_days) * 86400.0, 1.0e-30)

    density_left = np.power(10.0, np.asarray(left_columns["logRho"], dtype=float))
    density_right = np.power(10.0, np.asarray(right_columns["logRho"], dtype=float))
    pressure_current = effective_rsp_pressure(
        current_columns,
        pressure_work_mode=pressure_work_mode,
    )
    eq_current = effective_rsp_eq(
        current_columns,
        q_current,
        subtract_rsp_eq=subtract_rsp_eq,
    )

    specific_volume_left = np.divide(
        1.0,
        density_left,
        out=np.full_like(density_left, np.nan, dtype=float),
        where=density_left > 0.0,
    )
    specific_volume_right = np.divide(
        1.0,
        density_right,
        out=np.full_like(density_right, np.nan, dtype=float),
        where=density_right > 0.0,
    )
    d_specific_volume_dt = (specific_volume_right - specific_volume_left) / delta_time_sec
    return pressure_current * d_specific_volume_dt - eq_current


def build_fourier_derivative_design_matrix(phase: np.ndarray, fit_harmonics: int) -> np.ndarray:
    phase_array = np.asarray(phase, dtype=float)
    columns = [np.zeros_like(phase_array)]
    for harmonic in range(1, fit_harmonics + 1):
        angle = harmonic * TWO_PI * phase_array
        factor = harmonic * TWO_PI
        columns.append(-factor * np.sin(angle))
        columns.append(factor * np.cos(angle))
    return np.column_stack(columns)


def periodic_phase_weights(phase_sorted: np.ndarray) -> np.ndarray:
    phase = np.asarray(phase_sorted, dtype=float)
    extended = np.concatenate(([phase[-1] - 1.0], phase, [phase[0] + 1.0]))
    return 0.5 * (extended[2:] - extended[:-2])


def fit_periodic_scalar_series(
    phase: np.ndarray,
    values: np.ndarray,
    fit_harmonics: int,
) -> tuple[np.ndarray, int]:
    phase_array = np.asarray(phase, dtype=float)
    value_array = np.asarray(values, dtype=float)
    if phase_array.ndim != 1 or value_array.ndim != 1 or phase_array.size != value_array.size:
        raise ValueError("phase and values must be one-dimensional arrays of the same length.")
    if phase_array.size < 3:
        raise ValueError("Need at least three samples for a periodic fit.")
    max_harmonics = max(1, (phase_array.size - 1) // 2)
    harmonic_count = int(max(1, min(int(fit_harmonics), max_harmonics)))
    design_matrix = build_fourier_design_matrix(phase_array, harmonic_count)
    weights = np.sqrt(np.clip(periodic_phase_weights(phase_array), 1.0e-12, None))
    weighted_design_matrix = design_matrix * weights[:, None]
    weighted_values = value_array * weights
    coefficients, *_ = np.linalg.lstsq(weighted_design_matrix, weighted_values, rcond=None)
    return np.asarray(coefficients, dtype=float), harmonic_count


def evaluate_periodic_scalar_series(
    phase: np.ndarray,
    coefficients: np.ndarray,
    fit_harmonics: int,
) -> np.ndarray:
    design_matrix = build_fourier_design_matrix(np.asarray(phase, dtype=float), int(fit_harmonics))
    return np.asarray(design_matrix @ np.asarray(coefficients, dtype=float), dtype=float)


def evaluate_periodic_scalar_derivative(
    phase: np.ndarray,
    coefficients: np.ndarray,
    fit_harmonics: int,
) -> np.ndarray:
    derivative_design_matrix = build_fourier_derivative_design_matrix(
        np.asarray(phase, dtype=float),
        int(fit_harmonics),
    )
    return np.asarray(derivative_design_matrix @ np.asarray(coefficients, dtype=float), dtype=float)


def prepare_periodic_pdv_model(
    cycle_records: list[dict[str, object]],
    cycle_phase: np.ndarray,
    profile_cache: dict[int, dict[str, object]],
    fit_harmonics: int = PDV_PERIODIC_FIT_HARMONICS,
    *,
    pressure_work_mode: str = "base",
    subtract_rsp_eq: bool = False,
) -> dict[str, object]:
    loaded_profiles = [
        load_profile_cached(cycle_records, profile_cache, index)
        for index in range(len(cycle_records))
    ]
    q_reference = np.asarray(loaded_profiles[0]["columns"]["q"], dtype=float)
    n_profiles = len(loaded_profiles)
    n_zones = q_reference.size

    specific_volume_matrix = np.empty((n_profiles, n_zones), dtype=float)
    pressure_matrix = np.empty((n_profiles, n_zones), dtype=float)
    eq_matrix = np.empty((n_profiles, n_zones), dtype=float)
    for index, profile in enumerate(loaded_profiles):
        columns = profile["columns"]
        q_current = np.asarray(columns["q"], dtype=float)
        if not np.array_equal(q_current, q_reference):
            raise RuntimeError("The q grid changed across the nonlinear hydro cycle; periodic pdV fitting is unsafe.")
        density = np.power(10.0, np.asarray(columns["logRho"], dtype=float))
        specific_volume_matrix[index] = np.divide(
            1.0,
            density,
            out=np.full_like(density, np.nan, dtype=float),
            where=density > 0.0,
        )
        pressure_matrix[index] = effective_rsp_pressure(
            columns,
            pressure_work_mode=pressure_work_mode,
        )
        eq_matrix[index] = effective_rsp_eq(
            columns,
            q_reference,
            subtract_rsp_eq=subtract_rsp_eq,
        )

    if not np.all(np.isfinite(specific_volume_matrix)):
        raise RuntimeError("Encountered non-finite specific volumes while building the periodic pdV model.")
    if not np.all(np.isfinite(pressure_matrix)):
        raise RuntimeError("Encountered non-finite pressures while building the periodic pdV model.")
    if not np.all(np.isfinite(eq_matrix)):
        raise RuntimeError("Encountered non-finite rsp_Eq values while building the periodic pdV model.")

    design_matrix = build_fourier_design_matrix(np.asarray(cycle_phase, dtype=float), fit_harmonics)
    weights = np.sqrt(np.clip(periodic_phase_weights(np.asarray(cycle_phase, dtype=float)), 1.0e-12, None))
    weighted_design_matrix = design_matrix * weights[:, None]

    specific_volume_coefficients, *_ = np.linalg.lstsq(
        weighted_design_matrix,
        specific_volume_matrix * weights[:, None],
        rcond=None,
    )
    pressure_coefficients, *_ = np.linalg.lstsq(
        weighted_design_matrix,
        pressure_matrix * weights[:, None],
        rcond=None,
    )
    eq_coefficients, *_ = np.linalg.lstsq(
        weighted_design_matrix,
        eq_matrix * weights[:, None],
        rcond=None,
    )

    return {
        "q_surface_order": q_reference,
        "fit_harmonics": int(fit_harmonics),
        "specific_volume_coefficients": specific_volume_coefficients,
        "pressure_coefficients": pressure_coefficients,
        "eq_coefficients": eq_coefficients,
        "pressure_work_mode": str(pressure_work_mode),
        "subtract_rsp_eq": bool(subtract_rsp_eq),
    }


def evaluate_periodic_pdv_surface_order(
    periodic_pdv_model: dict[str, object],
    phase_samples: np.ndarray,
    period_days: float,
) -> np.ndarray:
    phase_array = np.asarray(phase_samples, dtype=float)
    fit_harmonics = int(periodic_pdv_model["fit_harmonics"])
    design_matrix = build_fourier_design_matrix(phase_array, fit_harmonics)
    derivative_design_matrix = build_fourier_derivative_design_matrix(phase_array, fit_harmonics)
    pressure_matrix = design_matrix @ np.asarray(periodic_pdv_model["pressure_coefficients"], dtype=float)
    eq_matrix = design_matrix @ np.asarray(periodic_pdv_model["eq_coefficients"], dtype=float)
    d_specific_volume_dphase = derivative_design_matrix @ np.asarray(
        periodic_pdv_model["specific_volume_coefficients"],
        dtype=float,
    )
    d_specific_volume_dt = d_specific_volume_dphase / max(float(period_days) * SECONDS_PER_DAY, 1.0e-30)
    return pressure_matrix * d_specific_volume_dt - eq_matrix


def merge_spans(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not spans:
        return []
    sorted_spans = sorted((float(min(x0, x1)), float(max(x0, x1))) for x0, x1 in spans)
    merged: list[list[float]] = [[sorted_spans[0][0], sorted_spans[0][1]]]
    for start, end in sorted_spans[1:]:
        current = merged[-1]
        if start <= current[1]:
            current[1] = max(current[1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def add_zone_overlays_coordinate(
    ax: plt.Axes,
    zone_spans: dict[str, list[tuple[float, float]]],
    label_positions: dict[str, tuple[float, str]],
    coordinate_profile: dict[str, object],
    dark_mode: bool = False,
) -> None:
    text_transform = blended_transform_factory(ax.transData, ax.transAxes)
    coordinate = np.asarray(coordinate_profile["x"], dtype=float)
    strength = np.asarray(coordinate_profile["strength"], dtype=float)
    strength_max = max(float(coordinate_profile["max_strength"]), 1.0e-12)
    reference_colors = theme_zone_colors(dark_mode)
    stroke_color = theme_value(dark_mode, "zone_label_stroke", "white")

    for x0, x1 in zone_spans.get("Convection", []):
        left = float(min(x0, x1))
        right = float(max(x0, x1))
        in_span = (
            np.isfinite(coordinate)
            & np.isfinite(strength)
            & (coordinate >= left)
            & (coordinate <= right)
        )
        span_strength = (
            float(np.nanmean(strength[in_span]) / strength_max)
            if np.any(in_span)
            else 0.0
        )
        span_strength = min(max(span_strength, 0.0), 1.0)
        if dark_mode:
            edge_gray = float(DARK_THEME["convection_edge_base"]) + float(DARK_THEME["convection_edge_span"]) * span_strength
            edgecolor = (edge_gray, edge_gray, edge_gray, 0.32 + 0.16 * span_strength)
            facecolor = (1.0, 1.0, 1.0, 0.0)
        else:
            hatch_gray = 0.95 - 0.18 * span_strength
            edgecolor = (hatch_gray, hatch_gray, hatch_gray, 0.72)
            facecolor = (0.75, 0.75, 0.75, 0.012)
        ax.axvspan(
            x0,
            x1,
            facecolor=facecolor,
            edgecolor=edgecolor,
            hatch="/",
            linewidth=0.0,
            zorder=-6,
        )

    for name, spans in zone_spans.items():
        if name == "Convection" or not spans:
            continue
        color = reference_colors.get(name, "0.45")
        for x0, x1 in spans:
            ax.axvspan(
                x0,
                x1,
                facecolor=color,
                edgecolor="none",
                linewidth=0.0,
                alpha=0.34 if not dark_mode else 0.38,
                zorder=-4,
            )
    for name, (label_x, label_text) in label_positions.items():
        color = theme_value(dark_mode, "text", reference_colors.get(name, ZONE_LABEL_COLOR))
        ax.text(
            label_x,
            0.98,
            label_text,
            rotation=90,
            ha="center",
            va="top",
            fontsize=scaled_font(ZONE_LABEL_FONT_SIZE),
            color=color,
            transform=text_transform,
            path_effects=[pe.withStroke(linewidth=ZONE_LABEL_STROKE_WIDTH, foreground=stroke_color)],
        )


def map_display_spans_to_coordinate(
    display_zone_spans: dict[str, list[tuple[float, float]]],
    temperature_plot: np.ndarray,
    coordinate_plot: np.ndarray,
) -> dict[str, list[tuple[float, float]]]:
    temperature = np.asarray(temperature_plot, dtype=float)
    coordinate = np.asarray(coordinate_plot, dtype=float)
    mapped: dict[str, list[tuple[float, float]]] = {}
    for name, spans in display_zone_spans.items():
        mapped_spans: list[tuple[float, float]] = []
        for hot, cool in spans:
            low = float(min(hot, cool))
            high = float(max(hot, cool))
            in_span = (
                np.isfinite(temperature)
                & np.isfinite(coordinate)
                & (temperature >= low)
                & (temperature <= high)
            )
            if np.any(in_span):
                mapped_spans.append(
                    (
                        float(np.nanmin(coordinate[in_span])),
                        float(np.nanmax(coordinate[in_span])),
                    )
                )
        if mapped_spans:
            mapped[name] = merge_spans(mapped_spans)
    return mapped


def temperature_to_q_coordinate(
    temperature_plot: np.ndarray,
    q_plot: np.ndarray,
    target_temperature: float,
) -> float:
    temperature = np.asarray(temperature_plot, dtype=float)
    q_values = np.asarray(q_plot, dtype=float)
    finite = np.isfinite(temperature) & np.isfinite(q_values)
    if np.count_nonzero(finite) < 2:
        return float("nan")
    temperature_ascending = temperature[finite][::-1]
    q_ascending = q_values[finite][::-1]
    target = float(
        np.clip(
            float(target_temperature),
            float(np.nanmin(temperature_ascending)),
            float(np.nanmax(temperature_ascending)),
        )
    )
    return float(np.interp(target, temperature_ascending, q_ascending))


def spans_temperature_to_q(
    display_zone_spans: dict[str, list[tuple[float, float]]],
    temperature_plot: np.ndarray,
    q_plot: np.ndarray,
) -> dict[str, list[tuple[float, float]]]:
    q_spans: dict[str, list[tuple[float, float]]] = {}
    for name, spans in display_zone_spans.items():
        q_spans_for_name: list[tuple[float, float]] = []
        for x0, x1 in spans:
            cool_temperature = float(min(x0, x1))
            hot_temperature = float(max(x0, x1))
            q_hot = temperature_to_q_coordinate(temperature_plot, q_plot, hot_temperature)
            q_cool = temperature_to_q_coordinate(temperature_plot, q_plot, cool_temperature)
            if np.isfinite(q_hot) and np.isfinite(q_cool):
                q_spans_for_name.append((float(min(q_hot, q_cool)), float(max(q_hot, q_cool))))
        if q_spans_for_name:
            q_spans[name] = merge_spans(q_spans_for_name)
    return q_spans


def map_q_spans_to_coordinate(
    q_zone_spans: dict[str, list[tuple[float, float]]],
    q_plot: np.ndarray,
    coordinate_plot: np.ndarray,
) -> dict[str, list[tuple[float, float]]]:
    q_values = np.asarray(q_plot, dtype=float)
    coordinate = np.asarray(coordinate_plot, dtype=float)
    finite = np.isfinite(q_values) & np.isfinite(coordinate)
    if np.count_nonzero(finite) < 2:
        return {}

    q_finite = q_values[finite]
    coordinate_finite = coordinate[finite]
    mapped: dict[str, list[tuple[float, float]]] = {}
    for name, spans in q_zone_spans.items():
        mapped_spans: list[tuple[float, float]] = []
        for q0, q1 in spans:
            left_q = float(min(q0, q1))
            right_q = float(max(q0, q1))
            coordinate_left = float(np.interp(left_q, q_finite, coordinate_finite))
            coordinate_right = float(np.interp(right_q, q_finite, coordinate_finite))
            mapped_spans.append(
                (
                    float(min(coordinate_left, coordinate_right)),
                    float(max(coordinate_left, coordinate_right)),
                )
            )
        if mapped_spans:
            mapped[name] = merge_spans(mapped_spans)
    return mapped


def compute_main_radius_xlim(
    frame_data: list[dict[str, object]],
    radius_rsun_xlim: tuple[float, float],
    radius_xmin: float | None,
    radius_xmax: float | None,
) -> tuple[tuple[float, float], dict[str, object]]:
    if (radius_xmin is None) != (radius_xmax is None):
        raise RuntimeError("Please provide both --radius-xmin and --radius-xmax when overriding radius limits.")

    available_left = float(min(radius_rsun_xlim))
    available_right = float(max(radius_rsun_xlim))
    if not np.isfinite(available_left) or not np.isfinite(available_right) or available_right <= available_left:
        raise RuntimeError("Could not determine a finite plotted radius range.")

    if radius_xmin is not None and radius_xmax is not None:
        x0 = float(radius_xmin)
        x1 = float(radius_xmax)
        if not np.isfinite(x0) or not np.isfinite(x1) or x1 <= x0:
            raise RuntimeError("--radius-xmin and --radius-xmax must define an increasing finite interval.")
        return (x0, x1), {
            "mode": "manual_override",
            "available_radius_rsun": [available_left, available_right],
        }

    photosphere_radii = np.asarray(
        [float(frame["photosphere_radius_rsun"]) for frame in frame_data],
        dtype=float,
    )
    finite_photosphere = photosphere_radii[np.isfinite(photosphere_radii)]
    if finite_photosphere.size == 0:
        raise RuntimeError("Could not determine finite photosphere radii for dynamic radius limits.")

    photo_left = float(np.nanmin(finite_photosphere))
    photo_right = float(np.nanmax(finite_photosphere))
    zone_edges: list[float] = []
    for frame in frame_data:
        mapped_spans = map_q_spans_to_coordinate(
            frame.get("display_zone_spans_q", frame["display_zone_spans"]),
            np.asarray(frame["q_plot"], dtype=float),
            np.asarray(frame["radius_rsun_plot"], dtype=float),
        )
        for zone_name in ("He II Ionization", "H/He I Ionization", "H Ionization", "He I Ionization"):
            for span in mapped_spans.get(zone_name, []):
                zone_edges.extend([float(span[0]), float(span[1])])

    finite_zone_edges = np.asarray(zone_edges, dtype=float)
    finite_zone_edges = finite_zone_edges[np.isfinite(finite_zone_edges)]
    wanted_left = photo_left - MAIN_RADIUS_INNER_MARGIN_RSUN
    wanted_right = photo_right + MAIN_RADIUS_OUTER_MARGIN_RSUN
    if finite_zone_edges.size:
        wanted_left = min(wanted_left, float(np.nanmin(finite_zone_edges)) - MAIN_RADIUS_ZONE_PADDING_RSUN)
        wanted_right = max(wanted_right, float(np.nanmax(finite_zone_edges)) + MAIN_RADIUS_ZONE_PADDING_RSUN)

    x0 = max(available_left, wanted_left)
    x1 = max(wanted_right, photo_right + MAIN_RADIUS_OUTER_MARGIN_RSUN)
    if not np.isfinite(x0) or not np.isfinite(x1) or x1 <= x0:
        x0 = available_left
        x1 = max(available_right, photo_right + MAIN_RADIUS_OUTER_MARGIN_RSUN)

    if x1 - x0 < MAIN_RADIUS_MIN_SPAN_RSUN:
        center = 0.5 * (photo_left + photo_right)
        half_span = 0.5 * MAIN_RADIUS_MIN_SPAN_RSUN
        x0 = max(available_left, center - half_span)
        x1 = max(center + half_span, photo_right + MAIN_RADIUS_OUTER_MARGIN_RSUN)
        if x1 - x0 < MAIN_RADIUS_MIN_SPAN_RSUN:
            x1 = x0 + MAIN_RADIUS_MIN_SPAN_RSUN

    return (float(x0), float(x1)), {
        "mode": "dynamic_photosphere_and_ionization_window",
        "available_radius_rsun": [available_left, available_right],
        "photosphere_radius_rsun": [photo_left, photo_right],
        "ionization_radius_rsun": (
            [float(np.nanmin(finite_zone_edges)), float(np.nanmax(finite_zone_edges))]
            if finite_zone_edges.size
            else None
        ),
        "inner_margin_rsun": float(MAIN_RADIUS_INNER_MARGIN_RSUN),
        "outer_margin_rsun": float(MAIN_RADIUS_OUTER_MARGIN_RSUN),
        "ionization_padding_rsun": float(MAIN_RADIUS_ZONE_PADDING_RSUN),
        "minimum_span_rsun": float(MAIN_RADIUS_MIN_SPAN_RSUN),
    }


def smooth_zone_boundaries_in_q(
    frame_data: list[dict[str, object]],
    sampled_phase: np.ndarray,
    zone_names: tuple[str, ...] = ("He II Ionization", "H/He I Ionization"),
    fit_harmonics: int = ZONE_BOUNDARY_FIT_HARMONICS,
) -> None:
    phase = np.asarray(sampled_phase, dtype=float)
    if not frame_data or phase.size != len(frame_data):
        return

    for frame in frame_data:
        frame["display_zone_spans_q"] = spans_temperature_to_q(
            frame["display_zone_spans"],
            np.asarray(frame["temperature_plot"], dtype=float),
            np.asarray(frame["q_plot"], dtype=float),
        )

    for zone_name in zone_names:
        q_left_edges: list[float] = []
        q_right_edges: list[float] = []
        for frame in frame_data:
            spans = frame["display_zone_spans_q"].get(zone_name, [])
            if len(spans) != 1:
                q_left_edges = []
                q_right_edges = []
                break
            q_left_edges.append(float(spans[0][0]))
            q_right_edges.append(float(spans[0][1]))

        if not q_left_edges:
            continue

        q_left_array = np.asarray(q_left_edges, dtype=float)
        q_right_array = np.asarray(q_right_edges, dtype=float)
        q_center_array = 0.5 * (q_left_array + q_right_array)
        q_width_array = np.clip(q_right_array - q_left_array, 1.0e-12, None)

        center_coefficients, harmonic_count = fit_periodic_scalar_series(
            phase,
            q_center_array,
            fit_harmonics,
        )
        log_width_coefficients, _ = fit_periodic_scalar_series(
            phase,
            np.log(q_width_array),
            harmonic_count,
        )

        center_smoothed = evaluate_periodic_scalar_series(phase, center_coefficients, harmonic_count)
        width_smoothed = np.exp(
            evaluate_periodic_scalar_series(phase, log_width_coefficients, harmonic_count)
        )
        for frame, q_center, q_width in zip(frame_data, center_smoothed, width_smoothed):
            q_values = np.asarray(frame["q_plot"], dtype=float)
            q_min = float(np.nanmin(q_values))
            q_max = float(np.nanmax(q_values))
            q_half_width = 0.5 * float(max(q_width, 1.0e-12))
            q0 = float(np.clip(float(q_center) - q_half_width, q_min, q_max))
            q1 = float(np.clip(float(q_center) + q_half_width, q_min, q_max))
            frame["display_zone_spans_q"][zone_name] = [(q0, q1)]


def label_positions_from_spans(
    display_zone_spans: dict[str, list[tuple[float, float]]],
    coordinate: str,
) -> dict[str, tuple[float, str]]:
    positions: dict[str, tuple[float, str]] = {}
    if display_zone_spans.get("He II Ionization"):
        span = display_zone_spans["He II Ionization"][0]
        if coordinate == "temperature":
            xpos = float(np.sqrt(float(span[0]) * float(span[1])))
        else:
            xpos = 0.5 * (float(span[0]) + float(span[1]))
        positions["He II Ionization"] = (float(xpos), "He II")
    if display_zone_spans.get("H/He I Ionization"):
        span = display_zone_spans["H/He I Ionization"][0]
        if coordinate == "temperature":
            xpos = float(np.sqrt(float(span[0]) * float(span[1])))
        else:
            xpos = 0.5 * (float(span[0]) + float(span[1]))
        positions["H/He I Ionization"] = (float(xpos), "H / He I")
    return positions


def span_midpoint(spans: list[tuple[float, float]]) -> float:
    if not spans:
        return float("nan")
    first_span = spans[0]
    return 0.5 * (float(first_span[0]) + float(first_span[1]))


def transition_center_q(
    q_values: np.ndarray,
    transition_values: np.ndarray,
) -> float:
    q_array = np.asarray(q_values, dtype=float)
    transition_array = np.asarray(transition_values, dtype=float)
    valid_mask = np.isfinite(q_array) & np.isfinite(transition_array)
    if np.count_nonzero(valid_mask) < 2:
        return float("nan")
    transition_mask, details = normalized_transition_mask(transition_array, valid_mask)
    if not bool(details.get("detected", False)) or not np.any(transition_mask):
        return float("nan")
    q_selected = q_array[transition_mask]
    return 0.5 * (float(np.nanmin(q_selected)) + float(np.nanmax(q_selected)))


def q_cell_widths(q_values: np.ndarray) -> np.ndarray:
    q_array = np.asarray(q_values, dtype=float)
    if q_array.ndim != 1 or q_array.size == 0:
        return np.asarray([], dtype=float)
    if q_array.size == 1:
        return np.asarray([1.0], dtype=float)
    edges = np.empty(q_array.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (q_array[:-1] + q_array[1:])
    edges[0] = q_array[0] - 0.5 * (q_array[1] - q_array[0])
    edges[-1] = q_array[-1] + 0.5 * (q_array[-1] - q_array[-2])
    edges = np.clip(edges, 0.0, 1.0)
    widths = np.diff(edges)
    return np.clip(widths, 0.0, None)


def mass_weighted_average_on_mask(
    q_values: np.ndarray,
    signal_values: np.ndarray,
    mask: np.ndarray,
) -> float:
    q_array = np.asarray(q_values, dtype=float)
    signal_array = np.asarray(signal_values, dtype=float)
    mask_array = np.asarray(mask, dtype=bool)
    if q_array.size == 0 or q_array.size != signal_array.size or q_array.size != mask_array.size:
        return float("nan")
    weights = q_cell_widths(q_array)
    finite = np.isfinite(q_array) & np.isfinite(signal_array) & mask_array
    if not np.any(finite):
        return float("nan")
    selected_weights = weights[finite]
    weight_sum = float(np.nansum(selected_weights))
    if weight_sum <= 0.0:
        return float("nan")
    return float(np.nansum(selected_weights * signal_array[finite]) / weight_sum)


def mass_weighted_average_over_spans(
    q_values: np.ndarray,
    signal_values: np.ndarray,
    spans: list[tuple[float, float]],
) -> float:
    if not spans:
        return float("nan")
    q_array = np.asarray(q_values, dtype=float)
    span_mask = np.zeros(q_array.shape, dtype=bool)
    for q0, q1 in spans:
        left = float(min(q0, q1))
        right = float(max(q0, q1))
        span_mask |= (q_array >= left) & (q_array <= right)
    return mass_weighted_average_on_mask(q_array, np.asarray(signal_values, dtype=float), span_mask)


def mass_weighted_transition_average(
    q_values: np.ndarray,
    transition_values: np.ndarray,
    signal_values: np.ndarray,
) -> tuple[float, bool]:
    q_array = np.asarray(q_values, dtype=float)
    transition_array = np.asarray(transition_values, dtype=float)
    signal_array = np.asarray(signal_values, dtype=float)
    valid_mask = np.isfinite(q_array) & np.isfinite(transition_array) & np.isfinite(signal_array)
    if np.count_nonzero(valid_mask) < 2:
        return float("nan"), False
    transition_mask, details = normalized_transition_mask(transition_array, valid_mask)
    if not bool(details.get("detected", False)) or not np.any(transition_mask):
        return float("nan"), False
    return mass_weighted_average_on_mask(q_array, signal_array, transition_mask), True


def instantaneous_power_structure(
    header: dict[str, float],
    columns: dict[str, np.ndarray],
    hot_limit: float,
    pdv_rate_surface_order: np.ndarray | None = None,
    heating_mode: str = "dLdm",
) -> dict[str, object]:
    frame = instantaneous_zone_structure(header, columns, hot_limit)
    photosphere = profile_photosphere_state(header, columns)

    q_surface_order = np.asarray(columns["q"], dtype=float)
    q_order = np.argsort(q_surface_order)
    q_sorted = q_surface_order[q_order]
    temperature_sorted = np.power(10.0, np.asarray(columns["logT"], dtype=float)[q_order])
    radius_rsun_sorted = np.asarray(columns["radius"], dtype=float)[q_order]
    log_radius_rsun_sorted = np.log10(
        np.clip(radius_rsun_sorted, 1.0e-30, None)
    )
    radius_fraction_sorted = radius_rsun_sorted / max(float(photosphere["radius_rsun"]), 1.0e-30)
    log_one_minus_r_over_R_sorted = np.full_like(radius_fraction_sorted, np.nan, dtype=float)
    interior_mask = np.isfinite(radius_fraction_sorted) & (radius_fraction_sorted < 1.0)
    log_one_minus_r_over_R_sorted[interior_mask] = np.log10(
        np.clip(1.0 - radius_fraction_sorted[interior_mask], 1.0e-12, None)
    )
    luminosity_total_sorted = np.asarray(columns["luminosity"], dtype=float)[q_order]

    lsun_cgs = float(header["lsun"])
    rsun_cgs = float(header["rsun"])
    star_mass_g = float(header["star_mass"]) * float(header["msun"])
    photosphere_radius_cm = max(float(photosphere["radius_rsun"]) * rsun_cgs, 1.0e-30)
    photosphere_logg_cgs = math.log10(max(G_CGS * star_mass_g / photosphere_radius_cm**2, 1.0e-30))
    envelope_mass_g = star_mass_g - float(header["M_center"])
    if envelope_mass_g <= 0.0:
        raise RuntimeError("Encountered a non-positive envelope mass while computing the power diagnostics.")
    dldm_scale = lsun_cgs / envelope_mass_g

    if "rsp_Lr" in columns:
        luminosity_radiative_sorted = np.asarray(columns["rsp_Lr"], dtype=float)[q_order] / lsun_cgs
    else:
        luminosity_radiative_sorted = luminosity_total_sorted.copy()
    if "rsp_Lc" in columns:
        luminosity_convective_sorted = np.asarray(columns["rsp_Lc"], dtype=float)[q_order] / lsun_cgs
    else:
        luminosity_convective_sorted = np.zeros_like(luminosity_total_sorted)
    if "rsp_Lc_div_L" in columns:
        convective_strength_sorted = np.abs(np.asarray(columns["rsp_Lc_div_L"], dtype=float)[q_order])
    else:
        convective_strength_sorted = np.abs(
            np.divide(
                luminosity_convective_sorted,
                luminosity_total_sorted,
                out=np.full_like(luminosity_total_sorted, np.nan, dtype=float),
                where=np.abs(luminosity_total_sorted) > 0.0,
            )
        )

    heating_total_sorted = -np.gradient(luminosity_total_sorted, q_sorted) * dldm_scale
    heating_radiative_sorted = -np.gradient(luminosity_radiative_sorted, q_sorted) * dldm_scale
    heating_convective_sorted = -np.gradient(luminosity_convective_sorted, q_sorted) * dldm_scale
    if heating_mode == "dLdm_plus_eq":
        heating_total_sorted = heating_total_sorted + column_or_zeros(
            columns,
            "rsp_Eq",
            q_surface_order,
            required=True,
        )[q_order]
    elif heating_mode == "gas_minus_c":
        coupling_sorted = column_or_zeros(
            columns,
            "rsp_src_snk",
            q_surface_order,
            required=True,
        )[q_order]
        heating_total_sorted = (
            -np.gradient(luminosity_radiative_sorted + luminosity_convective_sorted, q_sorted) * dldm_scale
            - coupling_sorted
        )
    if pdv_rate_surface_order is None:
        pdv_power_sorted = np.zeros_like(heating_total_sorted)
    else:
        pdv_power_sorted = np.asarray(pdv_rate_surface_order, dtype=float)[q_order]

    plot_mask = (
        np.isfinite(temperature_sorted)
        & np.isfinite(heating_total_sorted)
        & np.isfinite(heating_radiative_sorted)
        & np.isfinite(heating_convective_sorted)
        & np.isfinite(pdv_power_sorted)
        & (temperature_sorted <= float(hot_limit))
    )
    photosphere_q = float(photosphere["q_env"])

    frame.update(
        {
            "temperature_plot": temperature_sorted[plot_mask],
            "q_plot": q_sorted[plot_mask],
            "radius_rsun_plot": radius_rsun_sorted[plot_mask],
            "log_radius_rsun_plot": log_radius_rsun_sorted[plot_mask],
            "log_one_minus_r_over_R_plot": log_one_minus_r_over_R_sorted[plot_mask],
            "tau_plot": np.asarray(columns["tau"], dtype=float)[q_order][plot_mask],
            "opacity_plot": np.asarray(columns["opacity"], dtype=float)[q_order][plot_mask],
            "convection_strength_plot": convective_strength_sorted[plot_mask],
            "heating_total_plot": heating_total_sorted[plot_mask],
            "heating_radiative_plot": heating_radiative_sorted[plot_mask],
            "heating_convective_plot": heating_convective_sorted[plot_mask],
            "pdv_power_plot": pdv_power_sorted[plot_mask],
            "photosphere_q_env": float(photosphere_q),
            "photosphere_radius_rsun": float(photosphere["radius_rsun"]),
            "photosphere_tau": float(photosphere["tau"]),
            "photosphere_logg_cgs": float(photosphere_logg_cgs),
            "photosphere_velocity_km_per_s": float(photosphere["velocity_km_per_s"]),
            "photosphere_opacity": interpolate_on_q(
                q_sorted,
                np.asarray(columns["opacity"], dtype=float)[q_order],
                photosphere_q,
            ),
            "photosphere_heating_total": interpolate_on_q(q_sorted, heating_total_sorted, photosphere_q),
            "photosphere_heating_radiative": interpolate_on_q(q_sorted, heating_radiative_sorted, photosphere_q),
            "photosphere_heating_convective": interpolate_on_q(q_sorted, heating_convective_sorted, photosphere_q),
            "photosphere_pdv_power": interpolate_on_q(q_sorted, pdv_power_sorted, photosphere_q),
            "initial_z": float(header.get("initial_z", SOLAR_Z_REFERENCE)),
        }
    )
    if "ionization_he4" in columns:
        frame["ionization_he4_plot"] = np.asarray(columns["ionization_he4"], dtype=float)[q_order][plot_mask]
    return frame


def main() -> None:
    args = parse_args()

    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix
    dark_mode = bool(args.dark_mode)
    main_terms_only = bool(args.main_terms_only)
    pressure_work_mode = pressure_work_mode_from_args(args)
    pdv_subtract_rsp_eq = bool(args.pdv_subtract_rsp_eq)
    heating_mode = str(args.heating_mode)

    phase_curve_color = theme_value(dark_mode, "phase_curve", PHASE_CURVE_COLOR)
    photosphere_color = theme_value(dark_mode, "photosphere", PHOTOSPHERE_COLOR)
    shell_curve_color = theme_value(dark_mode, "shell_curve", NET_HEATING_COLOR)
    radiative_color = theme_value(dark_mode, "radiative", RADIATIVE_COLOR)
    convective_color = theme_value(dark_mode, "convective", CONVECTIVE_COLOR)
    pdv_color = theme_value(dark_mode, "pdv", PDV_COLOR)
    zero_line_color = theme_value(dark_mode, "spine", "k")
    marker_edge_color = theme_value(dark_mode, "marker_edge", "white")

    coordinate = str(args.coordinate)
    if coordinate == "temperature":
        stem = f"{prefix}_work_logT_phase_cycle"
    else:
        stem = f"{prefix}_work_r_over_R_phase_cycle"
    if dark_mode:
        stem = f"{stem}_dark"
    if main_terms_only:
        stem = f"{stem}_main_terms"
    stem = f"{stem}{heating_mode_stem_suffix(heating_mode)}"
    stem = f"{stem}{pressure_work_mode_stem_suffix(pressure_work_mode, pdv_subtract_rsp_eq)}"
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
    phase_curve_break_phases = phase_order_age_wrap_breaks(cycle_records, cycle_phase, period_days)

    sampled_phase = uniform_phase_grid(len(cycle_records), int(args.max_frames))
    profile_cache: dict[int, dict[str, object]] = {}
    periodic_pdv_model = prepare_periodic_pdv_model(
        cycle_records,
        cycle_phase,
        profile_cache,
        pressure_work_mode=pressure_work_mode,
        subtract_rsp_eq=pdv_subtract_rsp_eq,
    )
    sampled_pdv_surface_order = evaluate_periodic_pdv_surface_order(
        periodic_pdv_model,
        sampled_phase,
        period_days,
    )
    frame_data: list[dict[str, object]] = []
    for frame_index, phase_target in enumerate(sampled_phase):
        interpolated_profile = interpolate_profile_at_phase(
            cycle_records,
            cycle_phase,
            float(phase_target),
            profile_cache,
        )
        frame = instantaneous_power_structure(
            interpolated_profile["header"],
            interpolated_profile["columns"],
            float(args.hot_limit),
            pdv_rate_surface_order=sampled_pdv_surface_order[frame_index],
            heating_mode=heating_mode,
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
    sampled_photosphere_rv = np.asarray(
        [float(frame["photosphere_velocity_km_per_s"]) for frame in frame_data],
        dtype=float,
    )
    normalized_photosphere_l = normalize_unit_interval(sampled_photosphere_l)
    for frame, luminosity_unit in zip(frame_data, normalized_photosphere_l):
        frame["photosphere_luminosity_unit"] = float(np.nan_to_num(luminosity_unit, nan=0.5))

    mean_light_profile = load_mean_light_profile(run_dir, final_cycle_summary_path)
    mean_light_frame = instantaneous_power_structure(
        mean_light_profile["header"],
        mean_light_profile["columns"],
        float(args.hot_limit),
        heating_mode=heating_mode,
    )
    if coordinate == "temperature":
        label_positions = mean_light_zone_label_positions(mean_light_frame["display_zone_spans"])
    else:
        label_positions = {}
    smooth_display_zone_spans(frame_data)
    smooth_zone_boundaries_in_q(frame_data, sampled_phase)

    blackbody_color_table: dict[str, np.ndarray] | None = None
    photosphere_visual_metadata: dict[str, object] = {}
    if dark_mode or coordinate != "temperature":
        blackbody_color_table = load_blackbody_color_table(
            args.blackbody_color_file.resolve() if args.blackbody_color_file is not None else None
        )
    if dark_mode and blackbody_color_table is not None:
        photosphere_visual_metadata = add_photosphere_visual_state(frame_data, blackbody_color_table)
    star_mass_msun = float(mean_light_profile["header"]["star_mass"])
    initial_z_model = float(frame_data[0].get("initial_z", SOLAR_Z_REFERENCE))
    left_panel_caption = rf"RR Lyrae  {star_mass_msun:.3f} M$_\odot$, Z = {initial_z_model:.2f}"
    heating_total_legend_expression = heating_total_expression(heating_mode)

    outermost_temperature = float(min(float(frame["outermost_temperature_K"]) for frame in frame_data))
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
    if coordinate == "temperature":
        left_panel_x_field = "temperature_plot"
        left_panel_x_limits = (float(args.hot_limit), outermost_temperature)
    else:
        left_panel_x_field = "radius_rsun_plot"
        left_panel_x_limits = main_radius_xlim
    visible_power_bounds = finite_visible_scaled_power_bounds(
        frame_data,
        visible_power_series_names(main_terms_only),
        left_panel_x_field,
        left_panel_x_limits,
    )
    if args.ymin is not None and args.ymax is not None:
        left_power_ylim_raw = (float(args.ymin), float(args.ymax))
        left_power_ylim = (
            float(left_power_ylim_raw[0]) / DISPLAY_POWER_SCALE,
            float(left_power_ylim_raw[1]) / DISPLAY_POWER_SCALE,
        )
    elif args.ymin is not None or args.ymax is not None:
        raise RuntimeError("Please provide both --ymin and --ymax when overriding the left-panel y-axis limits.")
    else:
        left_power_ylim = power_panel_limits_from_visible_bounds(visible_power_bounds)
        left_power_ylim_raw = (
            float(left_power_ylim[0]) * DISPLAY_POWER_SCALE,
            float(left_power_ylim[1]) * DISPLAY_POWER_SCALE,
        )
    scaled_diagnostic_bounds = {
        "opacity": finite_visible_bounds(frame_data, "opacity_plot", left_panel_x_field, left_panel_x_limits),
    }
    opacity_scale_display = opacity_display_scale(scaled_diagnostic_bounds["opacity"], left_power_ylim)
    luminosity_ylim = fractional_padding(sampled_photosphere_l, fraction=0.08)
    rv_ylim = fractional_padding(sampled_photosphere_rv, fraction=0.08)
    luminosity_span = float(luminosity_ylim[1] - luminosity_ylim[0])
    rv_span = float(rv_ylim[1] - rv_ylim[0])
    luminosity_sphere_center = (
        float(LUM_SPHERE_PHASE),
        float(luminosity_ylim[0] + LUM_SPHERE_Y_FRACTION * luminosity_span - LUM_SPHERE_DROP_LSUN),
    )
    luminosity_radius_visual_center = (
        float(luminosity_sphere_center[0] + LUM_THERMOMETER_PHASE_SHIFT + LUM_RADIUS_THERMOMETER_PHASE_EXTRA_SHIFT),
        float(luminosity_sphere_center[1] + LUM_VISUAL_STACK_OFFSET_LSUN + LUM_RADIUS_THERMOMETER_DY_LSUN),
    )
    luminosity_teff_visual_center = (
        float(luminosity_sphere_center[0] + LUM_THERMOMETER_PHASE_SHIFT),
        float(luminosity_sphere_center[1] - LUM_VISUAL_STACK_OFFSET_LSUN + LUM_TEFF_THERMOMETER_DY_LSUN),
    )

    phase_curve_repeats = 3
    cycle_luminosity_phase_three, cycle_luminosity_curve_three = repeated_phase_curve(
        sampled_phase,
        sampled_photosphere_l,
        phase_curve_repeats,
    )
    cycle_rv_phase_three, cycle_rv_curve_three = repeated_phase_curve(
        sampled_phase,
        sampled_photosphere_rv,
        phase_curve_repeats,
    )
    photosphere_radius_series = np.asarray(
        [float(frame["photosphere_radius_rsun"]) for frame in frame_data],
        dtype=float,
    )
    photosphere_temperature_series = np.asarray(
        [float(frame["photosphere_temperature_K"]) for frame in frame_data],
        dtype=float,
    )
    min_radius_index = int(np.nanargmin(photosphere_radius_series))
    max_radius_index = int(np.nanargmax(photosphere_radius_series))
    min_light_index = int(np.nanargmin(sampled_photosphere_l))
    max_light_index = int(np.nanargmax(sampled_photosphere_l))
    min_teff_index = int(np.nanargmin(photosphere_temperature_series))
    max_teff_index = int(np.nanargmax(photosphere_temperature_series))
    max_rv_index = int(np.nanargmax(sampled_photosphere_rv))
    rv_sphere_center = (
        float(sampled_phase[max_rv_index] + 0.5),
        float(rv_ylim[0] + RV_SPHERE_Y_FRACTION * rv_span),
    )

    trend_fit_harmonics = min(12, max(1, (len(sampled_phase) - 1) // 2))
    radius_coefficients, trend_fit_harmonics = fit_periodic_scalar_series(
        sampled_phase,
        photosphere_radius_series,
        trend_fit_harmonics,
    )
    teff_coefficients, _ = fit_periodic_scalar_series(
        sampled_phase,
        photosphere_temperature_series,
        trend_fit_harmonics,
    )
    radius_dphase = evaluate_periodic_scalar_derivative(sampled_phase, radius_coefficients, trend_fit_harmonics)
    teff_dphase = evaluate_periodic_scalar_derivative(sampled_phase, teff_coefficients, trend_fit_harmonics)

    def classify_trend(derivative: np.ndarray) -> np.ndarray:
        derivative_array = np.asarray(derivative, dtype=float)
        scale = max(float(np.nanmax(np.abs(derivative_array))), 1.0e-12)
        threshold = 0.015 * scale
        trend = np.zeros(derivative_array.shape, dtype=int)
        trend[derivative_array > threshold] = 1
        trend[derivative_array < -threshold] = -1
        return trend

    radius_trend = classify_trend(radius_dphase)
    teff_trend = classify_trend(teff_dphase)

    plt.rcParams.update(
        {
            "font.size": scaled_font(9),
            "axes.labelsize": scaled_font(11),
            "xtick.labelsize": scaled_font(8),
            "ytick.labelsize": scaled_font(8),
        }
    )

    figure_facecolor = theme_value(dark_mode, "figure_face", "white")
    figure_size_inches = (FIGURE_WIDTH_PX / FIGURE_DPI, FIGURE_HEIGHT_PX / FIGURE_DPI)
    fig = plt.figure(
        figsize=figure_size_inches,
        dpi=FIGURE_DPI,
        constrained_layout=True,
        facecolor=figure_facecolor,
    )
    grid = fig.add_gridspec(2, 2, width_ratios=[1.55, 1.0], height_ratios=[1.0, 1.0])
    ax_left = fig.add_subplot(grid[:, 0])
    ax_left_inset = None
    ax_rv = fig.add_subplot(grid[0, 1])
    ax_lum = fig.add_subplot(grid[1, 1], sharex=ax_rv)
    fig.patch.set_facecolor(figure_facecolor)
    style_axis_for_theme(ax_left, dark_mode)
    if ax_left_inset is not None:
        style_axis_for_theme(ax_left_inset, dark_mode)
    style_axis_for_theme(ax_rv, dark_mode)
    style_axis_for_theme(ax_lum, dark_mode)

    annotation_style = {
        "color": theme_value(dark_mode, "text", "black"),
        "fontsize": scaled_font(7.2),
        "zorder": 4,
        "path_effects": [pe.withStroke(linewidth=2.0, foreground=theme_value(dark_mode, "axes_face", "white"))],
    }

    def setup_phase_panel(
        ax: plt.Axes,
        y_series: np.ndarray,
        x_curve_three: np.ndarray,
        y_curve_three: np.ndarray,
        panel_ylim: tuple[float, float],
        y_label: str,
        visual_mode: str = "none",
        show_x_label: bool = True,
        min_light_label_y_shift: float = 0.0,
    ) -> dict[str, object]:
        panel_handles: dict[str, object] = {
            "phase_curve": None,
            "phase_vline_primary": None,
            "phase_vline_secondary": None,
            "sphere_image": None,
            "sphere_box": None,
            "radius_axis_line": None,
            "radius_axis_dot": None,
            "radius_text": None,
            "teff_axis_line": None,
            "teff_axis_dot": None,
            "teff_text": None,
            "sphere_mode": visual_mode,
        }

        if dark_mode and visual_mode in {"luminosity", "rv"}:
            if visual_mode == "luminosity":
                phase_rgb_cycle = np.asarray(
                    [
                        np.clip(
                            np.asarray(frame["photosphere_luminosity_sphere_rgb"], dtype=float)
                            * float(frame["photosphere_luminosity_sphere_brightness"]),
                            0.0,
                            1.0,
                        )
                        for frame in frame_data
                    ],
                    dtype=float,
                )
            else:
                phase_rgb_cycle = np.asarray(
                    [frame["photosphere_rv_sphere_rgb"] for frame in frame_data],
                    dtype=float,
                )
            cycle_point_rgb_three = repeated_phase_colors(phase_rgb_cycle, phase_curve_repeats)
            phase_curve_x, phase_curve_y, phase_curve_rgb = add_phase_curve_breaks(
                x_curve_three,
                y_curve_three,
                phase_curve_break_phases,
                cycle_point_rgb_three,
            )
            panel_handles["phase_curve"] = add_colored_phase_curve(
                ax,
                phase_curve_x,
                phase_curve_y,
                phase_curve_rgb if phase_curve_rgb is not None else cycle_point_rgb_three,
                linewidth=1.8,
                zorder=1.1,
            )
        else:
            phase_curve_x, phase_curve_y, _ = add_phase_curve_breaks(
                x_curve_three,
                y_curve_three,
                phase_curve_break_phases,
            )
            (phase_curve_line,) = ax.plot(
                phase_curve_x,
                phase_curve_y,
                color=phase_curve_color,
                linewidth=1.55,
                zorder=1,
            )
            panel_handles["phase_curve"] = phase_curve_line

        phase_vline_primary = ax.axvline(
            float(sampled_phase[0]),
            color="0.28",
            linewidth=1.0,
            linestyle="--",
            alpha=0.9,
            zorder=0.2,
        )
        phase_vline_secondary = ax.axvline(
            float(sampled_phase[0] + 1.0),
            color="0.28",
            linewidth=1.0,
            linestyle="--",
            alpha=0.9,
            zorder=0.2,
        )
        panel_handles["phase_vline_primary"] = phase_vline_primary
        panel_handles["phase_vline_secondary"] = phase_vline_secondary

        if visual_mode in {"luminosity", "rv"} and dark_mode:
            initial_frame = frame_data[0]
            if visual_mode == "luminosity":
                initial_rgb = np.asarray(initial_frame["photosphere_luminosity_sphere_rgb"], dtype=float)
                initial_brightness = float(initial_frame["photosphere_luminosity_sphere_brightness"])
                sphere_center = luminosity_sphere_center
                sphere_xycoords = "data"
            else:
                initial_rgb = np.asarray(initial_frame["photosphere_rv_sphere_rgb"], dtype=float)
                initial_brightness = float(photosphere_visual_metadata.get("rv_sphere_brightness", 0.72))
                sphere_center = rv_sphere_center
                sphere_xycoords = "data"
            initial_rgba = render_photosphere_sphere_rgba(
                initial_rgb,
                initial_brightness,
                tuple(initial_frame["photosphere_limb_darkening"]),
            )
            sphere_image = OffsetImage(initial_rgba, zoom=float(initial_frame["photosphere_sphere_zoom"]))
            sphere_box = AnnotationBbox(
                sphere_image,
                sphere_center,
                xycoords=sphere_xycoords,
                box_alignment=(0.5, 0.5),
                frameon=False,
                pad=0.0,
                zorder=2,
            )
            ax.add_artist(sphere_box)
            panel_handles.update({"sphere_image": sphere_image, "sphere_box": sphere_box})

            if visual_mode == "luminosity":
                radius_gauge_x = float(luminosity_radius_visual_center[0] + LUM_GAUGE_X_OFFSET_PHASE)
                radius_text_x = float(luminosity_radius_visual_center[0] + LUM_TEXT_X_OFFSET_PHASE)
                gauge_y0 = float(luminosity_radius_visual_center[1] - LUM_GAUGE_HALF_HEIGHT_LSUN)
                gauge_y1 = float(luminosity_radius_visual_center[1] + LUM_GAUGE_HALF_HEIGHT_LSUN)
                radius_axis_line, = ax.plot(
                    [radius_gauge_x, radius_gauge_x],
                    [gauge_y0, gauge_y1],
                    color=theme_value(dark_mode, "text", "black"),
                    linewidth=AXIS_LINE_WIDTH,
                    zorder=3,
                )
                initial_radius_dot_y = gauge_y0 + float(initial_frame["photosphere_radius_axis_unit"]) * (gauge_y1 - gauge_y0)
                radius_axis_dot, = ax.plot(
                    [radius_gauge_x],
                    [initial_radius_dot_y],
                    marker="o",
                    markersize=6.0,
                    color=tuple(np.asarray(initial_frame["photosphere_marker_rgb"], dtype=float)),
                    markeredgecolor=marker_edge_color,
                    markeredgewidth=0.6,
                    linestyle="None",
                    zorder=4,
                )
                radius_text = ax.text(
                    radius_text_x,
                    float(luminosity_radius_visual_center[1]),
                    rf"{float(initial_frame['photosphere_radius_rsun']):.2f} R$_\odot$",
                    color=theme_value(dark_mode, "text", "black"),
                    fontsize=scaled_font(7.8),
                    ha="left",
                    va="center",
                    zorder=4,
                    path_effects=[pe.withStroke(linewidth=2.0, foreground=theme_value(dark_mode, "axes_face", "white"))],
                )
                teff_gauge_x = float(luminosity_teff_visual_center[0] + LUM_GAUGE_X_OFFSET_PHASE)
                teff_text_x = float(luminosity_teff_visual_center[0] + LUM_TEXT_X_OFFSET_PHASE)
                teff_gauge_y0 = float(luminosity_teff_visual_center[1] - LUM_GAUGE_HALF_HEIGHT_LSUN)
                teff_gauge_y1 = float(luminosity_teff_visual_center[1] + LUM_GAUGE_HALF_HEIGHT_LSUN)
                teff_axis_line, = ax.plot(
                    [teff_gauge_x, teff_gauge_x],
                    [teff_gauge_y0, teff_gauge_y1],
                    color=theme_value(dark_mode, "text", "black"),
                    linewidth=AXIS_LINE_WIDTH,
                    zorder=3,
                )
                initial_teff_dot_y = teff_gauge_y0 + float(initial_frame["photosphere_temperature_axis_unit"]) * (
                    teff_gauge_y1 - teff_gauge_y0
                )
                teff_axis_dot, = ax.plot(
                    [teff_gauge_x],
                    [initial_teff_dot_y],
                    marker="o",
                    markersize=6.0,
                    color=theme_value(dark_mode, "text", "black"),
                    markeredgecolor=marker_edge_color,
                    markeredgewidth=0.6,
                    linestyle="None",
                    zorder=4,
                )
                teff_text = ax.text(
                    teff_text_x,
                    float(luminosity_teff_visual_center[1]),
                    rounded_temperature_text(float(initial_frame["photosphere_temperature_K"])),
                    color=theme_value(dark_mode, "text", "black"),
                    fontsize=scaled_font(7.8),
                    ha="left",
                    va="center",
                    zorder=4,
                    path_effects=[pe.withStroke(linewidth=2.0, foreground=theme_value(dark_mode, "axes_face", "white"))],
                )
                panel_handles.update(
                    {
                        "radius_axis_line": radius_axis_line,
                        "radius_axis_dot": radius_axis_dot,
                        "radius_text": radius_text,
                        "teff_axis_line": teff_axis_line,
                        "teff_axis_dot": teff_axis_dot,
                        "teff_text": teff_text,
                    }
                )

        for highlight_index in (min_radius_index, max_radius_index):
            highlight_frame = frame_data[highlight_index]
            highlight_phase = float(sampled_phase[highlight_index])
            highlight_value = float(y_series[highlight_index])
            highlight_color = (
                tuple(np.asarray(highlight_frame["photosphere_marker_rgb"], dtype=float))
                if dark_mode
                else photosphere_color
            )
            ax.plot(
                phase_reference_positions(highlight_phase),
                np.full(2, highlight_value, dtype=float),
                marker="o",
                markersize=float(highlight_frame.get("photosphere_marker_size_pts", PHOTOSPHERE_MARKER_BASE_SIZE)),
                markerfacecolor="none",
                markeredgecolor=highlight_color,
                markeredgewidth=1.2,
                linestyle="None",
                zorder=2.6,
            )

        for light_index in (min_light_index, max_light_index):
            light_phase = float(sampled_phase[light_index])
            light_value = float(y_series[light_index])
            ax.plot(
                phase_reference_positions(light_phase),
                np.full(2, light_value, dtype=float),
                marker="o",
                markersize=PHASE_REFERENCE_DOT_SIZE,
                color=theme_value(dark_mode, "text", "white"),
                markeredgecolor=theme_value(dark_mode, "text", "white"),
                markeredgewidth=0.0,
                linestyle="None",
                zorder=4.2,
            )
        max_velocity_phase = float(sampled_phase[max_rv_index])
        max_velocity_value = float(y_series[max_rv_index])
        max_velocity_positions = phase_reference_positions(max_velocity_phase)
        ax.plot(
            max_velocity_positions,
            np.full(2, max_velocity_value, dtype=float),
            marker="x",
            markersize=6.4,
            color=theme_value(dark_mode, "text", "white"),
            markeredgecolor=theme_value(dark_mode, "text", "white"),
            markeredgewidth=1.0,
            linestyle="None",
            zorder=4.35,
        )
        center_phase_index = int(np.argmin(np.abs(max_velocity_positions - 1.0)))
        ax.text(
            float(max_velocity_positions[center_phase_index]) + 0.04,
            max_velocity_value,
            "max velocity",
            ha="left",
            va="center",
            **annotation_style,
        )

        ax.text(
            float(sampled_phase[max_radius_index]) - 0.045,
            float(y_series[max_radius_index]),
            "max radius",
            ha="right",
            va="center",
            **annotation_style,
        )
        ax.text(
            (
                float(sampled_phase[min_radius_index]) + 0.045
                if visual_mode == "rv"
                else float(sampled_phase[min_radius_index] + 1.0) - 0.045
            ),
            float(y_series[min_radius_index]),
            "min radius",
            ha="left" if visual_mode == "rv" else "right",
            va="center",
            **annotation_style,
        )
        ax.text(
            float(sampled_phase[min_light_index]) + 0.045,
            float(y_series[min_light_index]) + float(min_light_label_y_shift),
            "min light",
            ha="left",
            va="center",
            **annotation_style,
        )
        ax.text(
            float(sampled_phase[max_light_index]) + 0.04,
            float(y_series[max_light_index]),
            "max light",
            ha="left",
            va="center",
            **annotation_style,
        )
        for teff_index, marker_text in ((max_teff_index, "↑"), (min_teff_index, "↓")):
            ax.plot(
                [float(sampled_phase[teff_index])],
                [float(y_series[teff_index])],
                marker=rf"${marker_text}$",
                markersize=7.0,
                color=theme_value(dark_mode, "text", "white"),
                markeredgecolor=theme_value(dark_mode, "text", "white"),
                markeredgewidth=0.0,
                linestyle="None",
                zorder=4.4,
            )
        ax.text(
            (
                float(sampled_phase[min_teff_index]) - 0.03
                if visual_mode == "rv"
                else float(sampled_phase[min_teff_index]) + 0.03
            ),
            float(y_series[min_teff_index]),
            "min Teff",
            ha="right" if visual_mode == "rv" else "left",
            va="center",
            **annotation_style,
        )
        if visual_mode == "rv":
            ax.text(
                float(sampled_phase[max_teff_index]) - 0.03,
                float(y_series[max_teff_index]),
                "max Teff",
                ha="right",
                va="center",
                **annotation_style,
            )

        phase_dot, = ax.plot(
            phase_moving_positions(float(sampled_phase[0])),
            np.full(4, float(y_series[0]), dtype=float),
            marker="o",
            markersize=float(frame_data[0].get("photosphere_marker_size_pts", PHOTOSPHERE_MARKER_BASE_SIZE)),
            color=theme_value(dark_mode, "text", photosphere_color),
            markeredgecolor=marker_edge_color,
            markeredgewidth=0.65,
            linestyle="None",
            zorder=5,
        )
        if show_x_label:
            ax.set_xlabel("Pulsation phase")
        else:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)
        ax.set_ylabel(y_label)
        ax.set_xlim(0.0, 2.0)
        ax.set_ylim(*panel_ylim)
        ax.grid(False)
        configure_linear_axis_ticks(ax)
        style_axis_for_theme(ax, dark_mode)
        panel_handles["phase_dot"] = phase_dot
        return panel_handles

    luminosity_panel = setup_phase_panel(
        ax_lum,
        sampled_photosphere_l,
        cycle_luminosity_phase_three,
        cycle_luminosity_curve_three,
        luminosity_ylim,
        r"Photosphere $L\ [L_\odot]$",
        visual_mode="luminosity",
        show_x_label=True,
    )
    rv_panel = setup_phase_panel(
        ax_rv,
        sampled_photosphere_rv,
        cycle_rv_phase_three,
        cycle_rv_curve_three,
        rv_ylim,
        r"Photosphere $v_r\ [{\rm km}\,{\rm s}^{-1}]$",
        visual_mode="rv",
        show_x_label=False,
        min_light_label_y_shift=-1.0,
    )

    def draw_left_panel(frame_index: int) -> float:
        nonlocal ax_left_inset
        frame = frame_data[frame_index]
        if coordinate == "temperature":
            x_plot = np.asarray(frame["temperature_plot"], dtype=float)
            photosphere_x = float(frame["photosphere_temperature_K"])
            zone_spans = frame["display_zone_spans"]
            current_label_positions = label_positions
            coordinate_profile = {
                "x": np.asarray(frame["convection_profile"]["temperature_K"], dtype=float),
                "strength": np.asarray(frame["convection_profile"]["strength"], dtype=float),
                "max_strength": float(frame["convection_profile"]["max_strength"]),
            }
        else:
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
                "pdv_power",
                np.asarray(frame["pdv_power_plot"], dtype=float),
                float(frame["photosphere_pdv_power"]),
                pdv_color,
                2.1,
                build_series_label(
                    "Pressure-Volume Work",
                    mechanical_work_legend_expression(
                        pressure_work_mode=pressure_work_mode,
                        subtract_rsp_eq=pdv_subtract_rsp_eq,
                    ),
                ),
            ),
            (
                "heating_total",
                np.asarray(frame["heating_total_plot"], dtype=float),
                float(frame["photosphere_heating_total"]),
                shell_curve_color,
                2.25,
                build_series_label("Net Heating", f"  {heating_total_legend_expression}"),
            ),
        ]
        if not main_terms_only:
            series_definitions.extend(
                [
                    (
                        "heating_radiative",
                        np.asarray(frame["heating_radiative_plot"], dtype=float),
                        float(frame["photosphere_heating_radiative"]),
                        radiative_color,
                        0.95,
                        build_series_label("Radiative Heating", "-dLr/dm", "0.05"),
                    ),
                    (
                        "heating_convective",
                        np.asarray(frame["heating_convective_plot"], dtype=float),
                        float(frame["photosphere_heating_convective"]),
                        convective_color,
                        0.95,
                        build_series_label("Convective Heating", "-dLc/dm", "0.05"),
                    ),
                ]
            )
        inset_series_names = {"pdv_power", "heating_total"}
        scaled_series_lookup: dict[str, np.ndarray] = {}
        scaled_kappa_handle = None

        ax_left.cla()
        style_axis_for_theme(ax_left, dark_mode)
        if dark_mode and coordinate != "temperature":
            add_photosphere_exterior_glow(
                ax_left,
                np.asarray(frame["radius_rsun_plot"], dtype=float),
                np.asarray(frame["tau_plot"], dtype=float),
                photosphere_x,
                float(frame.get("photosphere_tau", 2.0 / 3.0)),
                float(main_radius_xlim[1]),
                left_power_ylim,
                np.asarray(frame["photosphere_blackbody_rgb"], dtype=float),
                float(frame["photosphere_luminosity_sphere_brightness"]),
            )
        if coordinate == "temperature":
            add_zone_overlays_temperature(
                ax_left,
                zone_spans,
                current_label_positions,
                frame["convection_profile"],
                dark_mode=dark_mode,
            )
        else:
            add_zone_overlays_coordinate(
                ax_left,
                zone_spans,
                current_label_positions,
                coordinate_profile,
                dark_mode=dark_mode,
            )
        axis_linewidth = float(ax_left.spines["left"].get_linewidth())
        ax_left.axhline(0.0, color=zero_line_color, linewidth=axis_linewidth, linestyle="--", alpha=1.0, zorder=-3)
        if ax_left_inset is not None:
            add_zone_overlays_coordinate(
                ax_left_inset,
                zone_spans,
                {},
                coordinate_profile,
                dark_mode=dark_mode,
            )
            inset_axis_linewidth = float(ax_left_inset.spines["left"].get_linewidth())
            ax_left_inset.axhline(
                0.0,
                color=zero_line_color,
                linewidth=inset_axis_linewidth,
                linestyle="--",
                alpha=1.0,
                zorder=-3,
            )

        if coordinate != "temperature":
            luminosity_link_color = np.clip(
                np.asarray(frame["photosphere_luminosity_sphere_rgb"], dtype=float)
                * float(frame["photosphere_luminosity_sphere_brightness"]),
                0.0,
                1.0,
            )
            rv_link_color = np.clip(np.asarray(frame["photosphere_rv_sphere_rgb"], dtype=float), 0.0, 1.0)
            ax_left.plot(
                [photosphere_x, photosphere_x],
                [float(left_power_ylim[0]), 0.0],
                color=luminosity_link_color,
                linewidth=PHOTOSPHERE_LINK_LINEWIDTH,
                zorder=4.6,
            )
            ax_left.plot(
                [photosphere_x, photosphere_x],
                [0.0, float(left_power_ylim[1])],
                color=rv_link_color,
                linewidth=PHOTOSPHERE_LINK_LINEWIDTH,
                zorder=4.6,
            )
            photosphere_label_transform = blended_transform_factory(ax_left.transData, ax_left.transAxes)
            ax_left.text(
                float(photosphere_x + PHOTOSPHERE_LABEL_X_OFFSET_RSUN),
                0.98,
                "photosphere",
                rotation=90,
                ha="center",
                va="top",
                fontsize=scaled_font(ZONE_LABEL_FONT_SIZE),
                color=theme_value(dark_mode, "text", "black"),
                transform=photosphere_label_transform,
                path_effects=[
                    pe.withStroke(
                        linewidth=ZONE_LABEL_STROKE_WIDTH,
                        foreground=theme_value(dark_mode, "zone_label_stroke", "white"),
                    )
                ],
                zorder=4.7,
            )

        for series_name, values, photosphere_value, color, linewidth, label in series_definitions:
            plot_scale = float(PLOT_SCALE_FACTORS[series_name])
            scaled_values = plot_scale * values / DISPLAY_POWER_SCALE
            scaled_series_lookup[series_name] = np.asarray(scaled_values, dtype=float)
            ax_left.plot(
                x_plot,
                scaled_values,
                color=color,
                linewidth=linewidth,
                zorder=2,
                label=label,
            )
            if ax_left_inset is not None and series_name in inset_series_names:
                inset_values = scaled_values
                finite_inset = np.isfinite(x_plot) & np.isfinite(inset_values)
                ax_left_inset.plot(
                    x_plot[finite_inset],
                    inset_values[finite_inset],
                    color=color,
                    linewidth=linewidth,
                    zorder=2,
                )

        if coordinate != "temperature":
            opacity_min = float(scaled_diagnostic_bounds["opacity"][0])
            scaled_opacity = (
                np.asarray(frame["opacity_plot"], dtype=float) - opacity_min
            ) * opacity_scale_display
            (scaled_kappa_handle,) = ax_left.plot(
                x_plot,
                scaled_opacity,
                color=SCALED_KAPPA_COLOR,
                linewidth=1.0,
                linestyle=":",
                alpha=0.9,
                zorder=1,
                label="_nolegend_",
            )

        if coordinate == "temperature":
            configure_temperature_axis([ax_left], x_plot)
            ax_left.set_xlim(float(args.hot_limit), outermost_temperature)
            ax_left.set_xlabel("T [K]")
            style_axis_for_theme(ax_left, dark_mode)
        else:
            ax_left.set_xlim(*main_radius_xlim)
            ax_left.set_xlabel(r"r / R$_\odot$")
        ax_left.set_ylim(*left_power_ylim)
        ax_left.set_ylabel(r"Specific Power [$10^9$ erg g$^{-1}$ s$^{-1}$]")
        ax_left.grid(False)
        if coordinate != "temperature":
            active_states = {
                "heating": teff_trend[frame_index] > 0,
                "cooling": teff_trend[frame_index] < 0,
                "expanding": radius_trend[frame_index] > 0,
                "contracting": radius_trend[frame_index] < 0,
            }
            power_guide_x = float(main_radius_xlim[0]) + POWER_GUIDE_X_FRACTION_FROM_LEFT * (
                float(main_radius_xlim[1]) - float(main_radius_xlim[0])
            )
            power_guide_transform = blended_transform_factory(ax_left.transData, ax_left.transAxes)
            for label_text, y_position, active_color in POWER_GUIDE_LABELS:
                ax_left.text(
                    power_guide_x,
                    float(y_position),
                    label_text,
                    color=active_color if active_states[label_text] else POWER_GUIDE_INACTIVE,
                    fontsize=scaled_font(8.6),
                    fontweight="bold",
                    ha="left",
                    va="center",
                    transform=power_guide_transform,
                    path_effects=[pe.withStroke(linewidth=1.0, foreground="black")],
                    zorder=4.1,
                )
        if coordinate != "temperature":
            configure_linear_axis_ticks(ax_left)
        if ax_left_inset is not None:
            configure_linear_axis_ticks(ax_left_inset)
        legend_handles, legend_labels = ax_left.get_legend_handles_labels()
        convection_legend_handle = Patch(
            facecolor=(0.88, 0.88, 0.88, 0.14) if dark_mode else (0.55, 0.55, 0.55, 0.16),
            edgecolor=(0.95, 0.95, 0.95, 0.95) if dark_mode else (0.20, 0.20, 0.20, 0.90),
            hatch="////////",
            linewidth=0.9,
            label="Convection",
        )
        if scaled_kappa_handle is not None:
            legend_handles.append(scaled_kappa_handle)
            legend_labels.append("Scaled opacity")
        legend_handles.append(convection_legend_handle)
        legend_labels.append("Convection")
        legend = ax_left.legend(
            legend_handles,
            legend_labels,
            loc="upper left",
            frameon=False,
            fontsize=scaled_font(8),
            ncol=1,
            handlelength=1.8,
            handleheight=1.1,
            columnspacing=1.0,
            prop={"family": "monospace", "size": scaled_font(8)},
        )
        legend.set_title(
            left_panel_caption,
            prop={"family": "monospace", "size": scaled_font(8)},
        )
        legend._legend_box.align = "left"
        legend.get_title().set_ha("left")
        legend.get_title().set_multialignment("left")
        style_legend_for_theme(legend, dark_mode)
        return float(frame["photosphere_luminosity_lsun"])

    def update(frame_index: int) -> tuple[object, ...]:
        photosphere_luminosity = draw_left_panel(frame_index)
        current_phase = float(sampled_phase[frame_index])
        frame = frame_data[frame_index]
        panel_values = {
            "luminosity": float(photosphere_luminosity),
            "rv": float(frame["photosphere_velocity_km_per_s"]),
        }
        panel_handles_map = {
            "luminosity": luminosity_panel,
            "rv": rv_panel,
        }
        artists: list[object] = []
        for panel_name, panel_handles in panel_handles_map.items():
            panel_value = panel_values[panel_name]
            phase_dot = panel_handles["phase_dot"]
            phase_dot.set_data(phase_moving_positions(current_phase), np.full(4, panel_value, dtype=float))
            phase_dot.set_markersize(float(frame.get("photosphere_marker_size_pts", PHOTOSPHERE_MARKER_BASE_SIZE)))
            if dark_mode:
                phase_dot.set_color(theme_value(dark_mode, "text", photosphere_color))
            artists.append(phase_dot)

            phase_vline_primary = panel_handles["phase_vline_primary"]
            phase_vline_secondary = panel_handles["phase_vline_secondary"]
            if phase_vline_primary is not None:
                phase_vline_primary.set_xdata([current_phase, current_phase])
                artists.append(phase_vline_primary)
            if phase_vline_secondary is not None:
                phase_vline_secondary.set_xdata([current_phase + 1.0, current_phase + 1.0])
                artists.append(phase_vline_secondary)

            sphere_image = panel_handles["sphere_image"]
            sphere_box = panel_handles["sphere_box"]
            if sphere_image is not None and sphere_box is not None:
                sphere_mode = str(panel_handles.get("sphere_mode", "none"))
                if sphere_mode == "luminosity":
                    sphere_rgb = np.asarray(frame["photosphere_luminosity_sphere_rgb"], dtype=float)
                    sphere_brightness = float(frame["photosphere_luminosity_sphere_brightness"])
                else:
                    sphere_rgb = np.asarray(frame["photosphere_rv_sphere_rgb"], dtype=float)
                    sphere_brightness = float(photosphere_visual_metadata.get("rv_sphere_brightness", 0.72))
                sphere_image.set_data(
                    render_photosphere_sphere_rgba(
                        sphere_rgb,
                        sphere_brightness,
                        tuple(frame["photosphere_limb_darkening"]),
                    )
                )
                sphere_image.set_zoom(float(frame["photosphere_sphere_zoom"]))
                artists.append(sphere_box)

            radius_axis_dot = panel_handles["radius_axis_dot"]
            radius_text = panel_handles["radius_text"]
            if radius_axis_dot is not None and radius_text is not None:
                gauge_y0 = float(luminosity_radius_visual_center[1] - LUM_GAUGE_HALF_HEIGHT_LSUN)
                gauge_y1 = float(luminosity_radius_visual_center[1] + LUM_GAUGE_HALF_HEIGHT_LSUN)
                radius_dot_y = gauge_y0 + float(frame["photosphere_radius_axis_unit"]) * (gauge_y1 - gauge_y0)
                radius_axis_dot.set_data(
                    [float(luminosity_radius_visual_center[0] + LUM_GAUGE_X_OFFSET_PHASE)],
                    [radius_dot_y],
                )
                radius_axis_dot.set_color(tuple(np.asarray(frame["photosphere_marker_rgb"], dtype=float)))
                radius_text.set_text(rf"{float(frame['photosphere_radius_rsun']):.2f} R$_\odot$")
                artists.extend([radius_axis_dot, radius_text])

            teff_axis_dot = panel_handles["teff_axis_dot"]
            teff_text = panel_handles["teff_text"]
            if teff_axis_dot is not None and teff_text is not None:
                teff_gauge_y0 = float(luminosity_teff_visual_center[1] - LUM_GAUGE_HALF_HEIGHT_LSUN)
                teff_gauge_y1 = float(luminosity_teff_visual_center[1] + LUM_GAUGE_HALF_HEIGHT_LSUN)
                teff_dot_y = teff_gauge_y0 + float(frame["photosphere_temperature_axis_unit"]) * (
                    teff_gauge_y1 - teff_gauge_y0
                )
                teff_axis_dot.set_data(
                    [float(luminosity_teff_visual_center[0] + LUM_GAUGE_X_OFFSET_PHASE)],
                    [teff_dot_y],
                )
                teff_axis_dot.set_color(theme_value(dark_mode, "text", "black"))
                teff_text.set_text(rounded_temperature_text(float(frame["photosphere_temperature_K"])))
                artists.extend([teff_axis_dot, teff_text])

            radius_axis_line = panel_handles["radius_axis_line"]
            teff_axis_line = panel_handles["teff_axis_line"]
            if radius_axis_line is not None:
                artists.append(radius_axis_line)
            if teff_axis_line is not None:
                artists.append(teff_axis_line)

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
    fig.savefig(png_path, dpi=FIGURE_DPI, facecolor=fig.get_facecolor())
    plt.close(fig)

    summary = {
        "prefix": prefix,
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "gif_path": str(gif_path),
        "png_path": str(png_path),
        "cycle_source": cycle_source,
        "frame_generation": "fixed-q linear interpolation between bracketing nonlinear profiles on a uniform phase grid",
        "coordinate": coordinate,
        "main_terms_only": main_terms_only,
        "scaling_method_version": ANIMATION_SCALING_VERSION,
        "layout": (
            "left panel: phase-local power diagnostics versus temperature; right column top: photosphere radial velocity curve; right column bottom: photosphere luminosity curve"
            if coordinate == "temperature"
            else "left panel: phase-local power diagnostics versus r/R_sun; right column top: photosphere radial velocity curve; right column bottom: photosphere luminosity curve"
        ),
        "zone_boundary_smoothing": (
            "ionization-zone boundaries are converted from temperature spans to q spans, "
            "then the q-space zone centers and widths are smoothed over phase with a low-order "
            "periodic Fourier fit before being mapped to the displayed coordinate"
        ),
        "left_panel_quantity": "phase-local specific power diagnostics",
        "left_panel_units": "10^9 erg g^-1 s^-1",
        "left_panel_display_scale": float(DISPLAY_POWER_SCALE),
        "left_panel_caption": left_panel_caption,
        "heating_mode": heating_mode,
        "guide_label_source": {
            "shell": "photosphere",
            "heating_cooling_trigger": "sign of dT/dphase for the photosphere",
            "expanding_contracting_trigger": "sign of dR/dphase for the photosphere",
        },
        "component_definitions": {
            "heating_total": heating_total_description(heating_mode),
            "heating_radiative": "radiative contribution to the heating rate, -dLr/dm, with positive meaning radiative heating",
            "heating_convective": "convective contribution to the heating rate, -dLc/dm, with positive meaning convective heating",
            "pdv_power": (
                "phase-local specific mechanical-work rate, "
                f"{mechanical_work_expression(pressure_work_mode=pressure_work_mode, subtract_rsp_eq=pdv_subtract_rsp_eq).replace('dV/dt', 'd(1/rho)/dt')}, "
                "with positive meaning expansion work done by the gas and negative meaning contraction/compression"
            ),
        },
        "visible_series": (
            ["pdv_power", "heating_total"]
            if main_terms_only
            else ["pdv_power", "heating_total", "heating_radiative", "heating_convective"]
        ),
        "pdv_generation": {
            "method": "phase-weighted periodic Fourier fit of shell specific volume, effective pressure, and rsp_Eq across the nonlinear cycle, with analytic dV/dt",
            "fit_harmonics": int(periodic_pdv_model["fit_harmonics"]),
            "pressure_work_mode": pressure_work_mode,
            "subtract_rsp_eq": bool(pdv_subtract_rsp_eq),
            "pressure_definition": (
                "pressure + rsp_Pt + rsp_Pvsc"
                if pressure_work_mode == "full_rsp"
                else ("pressure + rsp_Pvsc" if pressure_work_mode == "gas_plus_pav" else "pressure")
            ),
            "eq_definition": "rsp_Eq subtracted from the mechanical work term" if pdv_subtract_rsp_eq else "not subtracted",
        },
        "plot_scale_factors": {name: float(value) for name, value in PLOT_SCALE_FACTORS.items()},
        "cycle_profile_count": len(cycle_records),
        "frame_count": len(frame_data),
        "closing_frame_added_for_seamless_loop": False,
        "phase_panel_repeat_mode": (
            "repeated cycle copies are separated by NaN gaps; the plot does not connect the last "
            "sample of one cycle to the first sample of the next"
        ),
        "phase_panel_repeat_count": int(phase_curve_repeats),
        "fps": int(args.fps),
        "period_days_used": float(period_days),
        "phase_reference_days_used": float(phase_reference_days),
        "phase_curve_break_phases": [float(value) for value in phase_curve_break_phases],
        "phase_curve_break_reason": (
            "phase-ordered profiles cross the chronological cycle boundary here; the plotted phase curves "
            "are broken at these phases so an unconverged cycle is not connected by a synthetic spike"
            if phase_curve_break_phases
            else None
        ),
        "hot_temperature_limit_K": float(args.hot_limit),
        "outermost_temperature_limit_K": outermost_temperature,
        "left_panel_x_field_for_scaling": left_panel_x_field,
        "left_panel_x_limits_for_scaling": [
            float(left_panel_x_limits[0]),
            float(left_panel_x_limits[1]),
        ],
        "left_panel_scaling_method": (
            "per-panel visible-window extrema; power y-limits use the maximum absolute "
            "specific-power value between the minimum and maximum shown x coordinate, "
            "matching the model_000 reference convention"
        ),
        "panel_y_ranges": {
            "left_power": {
                "limits": [float(left_power_ylim[0]), float(left_power_ylim[1])],
                "visible_data_bounds": [float(visible_power_bounds[0]), float(visible_power_bounds[1])],
                "x_field_for_scaling": left_panel_x_field,
                "x_limits_for_scaling": [float(left_panel_x_limits[0]), float(left_panel_x_limits[1])],
                "method": (
                    "minimum and maximum of the plotted power curves within this panel's shown x range; "
                    "the axis is padded symmetrically about zero using the larger absolute extremum"
                ),
            },
            "photosphere_radial_velocity": {
                "limits": [float(rv_ylim[0]), float(rv_ylim[1])],
                "visible_data_bounds": [
                    float(np.nanmin(sampled_photosphere_rv)),
                    float(np.nanmax(sampled_photosphere_rv)),
                ],
                "method": "panel-local fractional padding around the displayed photosphere radial-velocity curve",
            },
            "photosphere_luminosity": {
                "limits": [float(luminosity_ylim[0]), float(luminosity_ylim[1])],
                "visible_data_bounds": [
                    float(np.nanmin(sampled_photosphere_l)),
                    float(np.nanmax(sampled_photosphere_l)),
                ],
                "method": "panel-local fractional padding around the displayed photosphere luminosity curve",
            },
        },
        "left_power_visible_data_bounds": [
            float(visible_power_bounds[0]),
            float(visible_power_bounds[1]),
        ],
        "left_power_ylim_raw": [float(left_power_ylim_raw[0]), float(left_power_ylim_raw[1])],
        "left_power_ylim": [float(left_power_ylim[0]), float(left_power_ylim[1])],
        "scaled_diagnostic_bounds": {
            "opacity": [
                float(scaled_diagnostic_bounds["opacity"][0]),
                float(scaled_diagnostic_bounds["opacity"][1]),
            ],
        },
        "opacity_scaling": {
            "method": "left-panel visible-window min opacity maps to zero; left-panel visible-window max opacity maps to a fixed fraction of that panel's positive y-limit",
            "panel": "left_power",
            "panel_top_fraction": float(OPACITY_PANEL_TOP_FRACTION),
            "display_units_per_opacity_unit": float(opacity_scale_display),
            "opacity_min_baseline": float(scaled_diagnostic_bounds["opacity"][0]),
            "opacity_max_display_value": float(
                (float(scaled_diagnostic_bounds["opacity"][1]) - float(scaled_diagnostic_bounds["opacity"][0]))
                * opacity_scale_display
            ),
        },
        "radius_rsun_xlim": [float(radius_rsun_xlim[0]), float(radius_rsun_xlim[1])],
        "main_radius_xlim_used": [float(main_radius_xlim[0]), float(main_radius_xlim[1])],
        "main_radius_xlim_selection": main_radius_xlim_metadata,
        "inset_radius_window_relative_to_photosphere_rsun": None,
        "figure_size_px": [int(FIGURE_WIDTH_PX), int(FIGURE_HEIGHT_PX)],
        "luminosity_panel_ylim": [float(luminosity_ylim[0]), float(luminosity_ylim[1])],
        "rv_panel_ylim": [float(rv_ylim[0]), float(rv_ylim[1])],
        "mean_light_label_positions_x": {
            name: float(position[0]) for name, position in label_positions.items()
        },
        "sampled_frame_paths": [str(frame["path"]) for frame in frame_data],
        "sampled_frame_left_paths": [str(frame["left_path"]) for frame in frame_data],
        "sampled_frame_right_paths": [str(frame["right_path"]) for frame in frame_data],
        "sampled_frame_interpolation_weights": [float(frame["interpolation_weight"]) for frame in frame_data],
        "sampled_phases": [float(value) for value in sampled_phase],
        "dark_mode": dark_mode,
        "photosphere_visualization": (
            {
                "style": "dark-mode limb-darkened spheres with a luminosity/temperature-colored sphere on the luminosity panel and a radial-velocity-colored sphere on the radial-velocity panel",
                "anchor_data_luminosity": [float(luminosity_sphere_center[0]), float(luminosity_sphere_center[1])],
                "anchor_data_rv": [float(rv_sphere_center[0]), float(rv_sphere_center[1])],
                "radius_text_anchor_data_luminosity": [
                    float(luminosity_radius_visual_center[0] + LUM_TEXT_X_OFFSET_PHASE),
                    float(luminosity_radius_visual_center[1]),
                ],
                "teff_text_anchor_data_luminosity": [
                    float(luminosity_teff_visual_center[0] + LUM_TEXT_X_OFFSET_PHASE),
                    float(luminosity_teff_visual_center[1]),
                ],
                "radius_gauge_x_phase": float(luminosity_radius_visual_center[0] + LUM_GAUGE_X_OFFSET_PHASE),
                "radius_gauge_half_height_lsun": float(LUM_GAUGE_HALF_HEIGHT_LSUN),
                "sphere_image_size_px": int(SPHERE_IMAGE_SIZE),
                "sphere_base_zoom": float(SPHERE_BASE_ZOOM),
                "limb_darkening_law": "phase-dependent quadratic limb darkening with coefficients parameterized by photosphere Teff, log g, and inferred [Fe/H]",
                "phase_marker_base_size_pts": float(PHOTOSPHERE_MARKER_BASE_SIZE),
                "phase_marker_scaling": "marker diameter scales with the same radius_scale = R/R_ref factor used for the sphere zoom",
                "radius_extrema_markers": {
                    "min_radius": {
                        "frame_index": int(min_radius_index),
                        "phase": float(sampled_phase[min_radius_index]),
                        "radius_rsun": float(frame_data[min_radius_index]["photosphere_radius_rsun"]),
                        "luminosity_lsun": float(frame_data[min_radius_index]["photosphere_luminosity_lsun"]),
                    },
                    "max_radius": {
                        "frame_index": int(max_radius_index),
                        "phase": float(sampled_phase[max_radius_index]),
                        "radius_rsun": float(frame_data[max_radius_index]["photosphere_radius_rsun"]),
                        "luminosity_lsun": float(frame_data[max_radius_index]["photosphere_luminosity_lsun"]),
                    },
                },
                **photosphere_visual_metadata,
            }
            if dark_mode
            else None
        ),
        "interior_temperature_shading": (
            {
                "enabled": False,
                "blackbody_color_file": (
                    str(np.asarray(blackbody_color_table["path"], dtype=object)[0])
                    if blackbody_color_table is not None
                    else None
                ),
                "style": "disabled",
                "phase_luminosity_scaling": None,
                "luminosity_floor": None,
            }
            if coordinate != "temperature"
            else None
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
