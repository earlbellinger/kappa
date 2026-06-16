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
from matplotlib.transforms import blended_transform_factory
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

from plot_fourier_vs_logT import (
    COMPLEX_TRANSFER_REFERENCE_COLORS,
    configure_temperature_axis,
    geometric_midpoint,
    load_mean_light_profile,
)
from plot_fourier_vs_massdepth_profiles import build_zone_spans, parse_profile, profile_photosphere_state
from plot_mean_light_work_terms_vs_logT import collect_profile_records, load_json, select_last_complete_cycle

PHOTOSPHERE_COLOR = "#B24A3A"
SHELL_CURVE_COLOR = "#355C7D"
RADIATIVE_COLOR = "#C17B2C"
CONVECTIVE_COLOR = "#2F8F83"
PHASE_CURVE_COLOR = "#8C857E"
CONNECTOR_COLOR = "#6F6963"
ZONE_LABEL_COLOR = "#2B2B2B"
DEFAULT_FPS = 12
DEFAULT_MAX_FRAMES = 180
HOT_TEMPERATURE_LIMIT = 2.0e5
HEII_WINDOW_HOT_LIMIT = 1.0e5
HEII_WINDOW_COOL_LIMIT = 2.0e4
HEII_HALFMAX_FRACTION = 0.5
INTERPOLATED_PROFILE_COLUMNS = (
    "q",
    "tau",
    "radius",
    "luminosity",
    "vel_km_per_s",
    "logT",
    "logRho",
    "pressure",
    "opacity",
    "cp",
    "mu",
    "gamma1",
    "rsp_Lc_div_L",
    "rsp_Lc",
    "rsp_Lr",
    "rsp_Lt",
    "rsp_Eq",
    "rsp_src_snk",
    "rsp_src",
    "rsp_damp",
    "rsp_dampR",
    "rsp_Pt",
    "rsp_Pvsc",
    "ionization_h1",
    "ionization_he4",
    "typical_charge_h1",
    "typical_charge_he4",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Animate the nonlinear luminosity profile versus logT across one pulsation cycle."
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
        help="Hot-side temperature limit in K for the luminosity profile panel",
    )
    return parser.parse_args()


def fractional_padding(values: np.ndarray, fraction: float = 0.06) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    vrange = vmax - vmin
    if vrange <= 0.0:
        pad = max(abs(vmin), 1.0) * fraction
        return vmin - pad, vmax + pad
    pad = vrange * fraction
    return vmin - pad, vmax + pad


def normalize_unit_interval(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    normalized = np.full_like(array, np.nan, dtype=float)
    finite = np.isfinite(array)
    if not np.any(finite):
        return normalized
    vmin = float(np.min(array[finite]))
    vmax = float(np.max(array[finite]))
    vrange = vmax - vmin
    if vrange <= 0.0:
        normalized[finite] = 0.0
        return normalized
    normalized[finite] = (array[finite] - vmin) / vrange
    return normalized


def halfmax_span_descending(
    temperature_desc: np.ndarray,
    series_desc: np.ndarray,
    peak_index: int,
    fraction: float,
) -> tuple[float, float]:
    temperature = np.asarray(temperature_desc, dtype=float)
    series = np.asarray(series_desc, dtype=float)
    peak_index = int(peak_index)
    threshold = float(fraction) * float(series[peak_index])

    hot_index = peak_index
    while hot_index > 0 and series[hot_index - 1] >= threshold:
        hot_index -= 1

    cool_index = peak_index
    while cool_index < series.size - 1 and series[cool_index + 1] >= threshold:
        cool_index += 1

    return float(temperature[hot_index]), float(temperature[cool_index])


def heii_ionization_transition_span(
    temperature_sorted: np.ndarray,
    ionization_he4_sorted: np.ndarray | None,
    visible_mask: np.ndarray,
    cp_sorted: np.ndarray | None = None,
    gamma1_sorted: np.ndarray | None = None,
) -> tuple[tuple[float, float] | None, dict[str, object]]:
    details: dict[str, object] = {"detected": False}
    if ionization_he4_sorted is None:
        details["message"] = "ionization_he4 not present"
        return None, details

    analysis_mask = visible_mask & np.isfinite(temperature_sorted) & np.isfinite(ionization_he4_sorted)
    if np.count_nonzero(analysis_mask) < 2:
        details["message"] = "Insufficient cells to locate the He II ionization transition"
        return None, details

    temperature = np.asarray(temperature_sorted[analysis_mask], dtype=float)
    ionization = np.asarray(ionization_he4_sorted[analysis_mask], dtype=float)

    partial_mask = np.isfinite(ionization) & (ionization > 1.0) & (ionization < 2.0)
    if np.count_nonzero(partial_mask) >= 2:
        hot_temperature = float(np.nanmax(temperature[partial_mask]))
        cool_temperature = float(np.nanmin(temperature[partial_mask]))
        details.update(
            {
                "detected": True,
                "source": "cells with 1 < X_He,ion < 2 on the profile mesh",
                "partial_cell_count": int(np.count_nonzero(partial_mask)),
                "transition_temperatures_K": [hot_temperature, cool_temperature],
            }
        )
        return (hot_temperature, cool_temperature), details

    transition_indices = np.flatnonzero(
        np.isfinite(ionization[:-1])
        & np.isfinite(ionization[1:])
        & (ionization[:-1] >= 1.5)
        & (ionization[1:] <= 1.5)
        & (np.abs(np.diff(ionization)) > 0.5)
    )
    if transition_indices.size == 0:
        transition_indices = np.flatnonzero(np.abs(np.diff(ionization)) > 0.5)
    if transition_indices.size == 0:
        details["message"] = "No X_He,ion 2 -> 1 transition was found on the profile mesh"
        return None, details

    transition_index = int(transition_indices[0])
    hot_temperature = float(temperature[transition_index])
    cool_temperature = float(temperature[transition_index + 1])

    if cp_sorted is not None and gamma1_sorted is not None:
        cp = np.asarray(cp_sorted[analysis_mask], dtype=float)
        gamma1 = np.asarray(gamma1_sorted[analysis_mask], dtype=float)
        heii_window_mask = (
            np.isfinite(cp)
            & np.isfinite(gamma1)
            & (temperature <= HEII_WINDOW_HOT_LIMIT)
            & (temperature >= HEII_WINDOW_COOL_LIMIT)
        )
        if np.count_nonzero(heii_window_mask) >= 5:
            temperature_window = temperature[heii_window_mask]
            cp_window = cp[heii_window_mask]
            gamma1_window = gamma1[heii_window_mask]
            log_cp_window = np.log10(np.clip(cp_window, 1.0e-99, None))
            cp_norm = normalize_unit_interval(log_cp_window)
            gamma1_deficit = np.nanmax(gamma1_window) - gamma1_window
            gamma1_norm = normalize_unit_interval(gamma1_deficit)
            susceptibility = np.nan_to_num(cp_norm, nan=0.0) + np.nan_to_num(gamma1_norm, nan=0.0)

            local_maxima = np.flatnonzero(
                (susceptibility[1:-1] > susceptibility[:-2]) & (susceptibility[1:-1] >= susceptibility[2:])
            ) + 1
            candidate_indices = local_maxima if local_maxima.size else np.flatnonzero(np.isfinite(susceptibility))
            if candidate_indices.size:
                transition_center_temperature = 0.5 * (hot_temperature + cool_temperature)
                peak_index = int(
                    candidate_indices[
                        np.argmin(np.abs(temperature_window[candidate_indices] - transition_center_temperature))
                    ]
                )
                broad_hot_temperature, broad_cool_temperature = halfmax_span_descending(
                    temperature_window,
                    susceptibility,
                    peak_index,
                    HEII_HALFMAX_FRACTION,
                )
                hot_temperature = max(hot_temperature, broad_hot_temperature)
                cool_temperature = min(cool_temperature, broad_cool_temperature)
                details.update(
                    {
                        "detected": True,
                        "source": "thermodynamic He II partial-ionization band centered on the X_He,ion 2 -> 1 transition",
                        "transition_profile_indices": [transition_index, transition_index + 1],
                        "transition_temperatures_K": [float(temperature[transition_index]), float(temperature[transition_index + 1])],
                        "transition_values": [float(ionization[transition_index]), float(ionization[transition_index + 1])],
                        "thermodynamic_peak_temperature_K": float(temperature_window[peak_index]),
                        "thermodynamic_halfmax_fraction": float(HEII_HALFMAX_FRACTION),
                        "thermodynamic_span_temperatures_K": [float(broad_hot_temperature), float(broad_cool_temperature)],
                    }
                )
                return (hot_temperature, cool_temperature), details

    details.update(
        {
            "detected": True,
            "source": "adjacent shells bracketing the X_He,ion 2 -> 1 transition on the profile mesh",
            "transition_profile_indices": [transition_index, transition_index + 1],
            "transition_temperatures_K": [hot_temperature, cool_temperature],
            "transition_values": [float(ionization[transition_index]), float(ionization[transition_index + 1])],
        }
    )
    return (hot_temperature, cool_temperature), details


def instantaneous_zone_structure(
    header: dict[str, float],
    columns: dict[str, np.ndarray],
    hot_limit: float,
) -> dict[str, object]:
    photosphere = profile_photosphere_state(header, columns)
    q_surface_order = np.asarray(columns["q"], dtype=float)
    q_order = np.argsort(q_surface_order)
    q_sorted = q_surface_order[q_order]
    temperature_sorted = np.power(10.0, np.asarray(columns["logT"], dtype=float)[q_order])
    luminosity_sorted = np.asarray(columns["luminosity"], dtype=float)[q_order]
    cp_sorted = np.asarray(columns["cp"], dtype=float)[q_order]
    gamma1_sorted = np.asarray(columns["gamma1"], dtype=float)[q_order]
    lsun_cgs = float(header["lsun"])

    if "rsp_Lc_div_L" in columns:
        convective_fraction_sorted = np.asarray(columns["rsp_Lc_div_L"], dtype=float)[q_order]
    else:
        luminosity_cgs = np.asarray(columns["luminosity"], dtype=float) * lsun_cgs
        convective_fraction_sorted = np.divide(
            np.asarray(columns["rsp_Lc"], dtype=float),
            luminosity_cgs,
            out=np.full(q_sorted.size, np.nan, dtype=float),
            where=np.abs(luminosity_cgs) > 0.0,
        )[q_order]
    if "rsp_Lr" in columns:
        luminosity_radiative_sorted = np.asarray(columns["rsp_Lr"], dtype=float)[q_order] / lsun_cgs
    else:
        luminosity_radiative_sorted = luminosity_sorted - (
            np.asarray(columns["rsp_Lc"], dtype=float)[q_order] / lsun_cgs
        )
    if "rsp_Lc" in columns:
        luminosity_convective_sorted = np.asarray(columns["rsp_Lc"], dtype=float)[q_order] / lsun_cgs
    else:
        luminosity_convective_sorted = convective_fraction_sorted * luminosity_sorted

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

    heii_span, heii_details = heii_ionization_transition_span(
        temperature_sorted=temperature_sorted,
        ionization_he4_sorted=ionization_he4_sorted,
        visible_mask=visible_mask,
        cp_sorted=cp_sorted,
        gamma1_sorted=gamma1_sorted,
    )
    if heii_span is not None:
        zone_spans["He II Ionization"] = [tuple(sorted(heii_span))]
        zone_details["He II Ionization"] = heii_details

    display_zone_spans = canonicalize_display_zone_spans(zone_spans)
    plot_mask = np.isfinite(temperature_sorted) & np.isfinite(luminosity_sorted) & (temperature_sorted <= float(hot_limit))
    convection_profile = {
        "temperature_K": temperature_sorted[visible_mask],
        "strength": np.abs(convective_fraction_sorted[visible_mask]),
        "max_strength": float(np.nanmax(np.abs(convective_fraction_sorted[visible_mask])))
        if np.any(np.isfinite(convective_fraction_sorted[visible_mask]))
        else 0.0,
    }
    return {
        "temperature_plot": temperature_sorted[plot_mask],
        "luminosity_plot": luminosity_sorted[plot_mask],
        "luminosity_radiative_plot": luminosity_radiative_sorted[plot_mask],
        "luminosity_convective_plot": luminosity_convective_sorted[plot_mask],
        "outermost_temperature_K": float(np.nanmin(temperature_sorted)),
        "photosphere_temperature_K": float(10.0 ** float(photosphere["logT"])),
        "photosphere_luminosity_lsun": float(photosphere["luminosity_lsun"]),
        "zone_spans": zone_spans,
        "display_zone_spans": display_zone_spans,
        "zone_details": zone_details,
        "convection_profile": convection_profile,
    }


def uniform_phase_grid(total_profiles: int, max_frames: int) -> np.ndarray:
    frame_count = min(int(total_profiles), int(max_frames))
    return np.linspace(0.0, 1.0, frame_count, endpoint=False, dtype=float)


def load_profile_cached(
    cycle_records: list[dict[str, object]],
    profile_cache: dict[int, dict[str, object]],
    index: int,
) -> dict[str, object]:
    cached = profile_cache.get(int(index))
    if cached is not None:
        return cached

    record = cycle_records[int(index)]
    header, columns = parse_profile(Path(record["path"]))
    cached = {
        "path": str(record["path"]),
        "age_days": float(record["age_days"]),
        "header": header,
        "columns": columns,
    }
    profile_cache[int(index)] = cached
    return cached


def phase_bracket(
    phase_sorted: np.ndarray,
    target_phase: float,
) -> tuple[int, int, float, float]:
    phase = np.asarray(phase_sorted, dtype=float)
    target = float(target_phase)
    if target < float(phase[0]):
        return phase.size - 1, 0, float(phase[-1] - 1.0), float(phase[0])

    right_index = int(np.searchsorted(phase, target, side="right"))
    if right_index == 0:
        return phase.size - 1, 0, float(phase[-1] - 1.0), float(phase[0])
    if right_index == phase.size:
        return phase.size - 1, 0, float(phase[-1]), float(phase[0] + 1.0)
    left_index = right_index - 1
    return left_index, right_index, float(phase[left_index]), float(phase[right_index])


def interpolated_header(
    left_header: dict[str, float],
    right_header: dict[str, float],
    weight: float,
) -> dict[str, float]:
    header = dict(left_header)
    header.pop("photosphere_r", None)
    header.pop("photosphere_L", None)
    header.pop("photosphere_Teff", None)
    if "star_age" in left_header and "star_age" in right_header:
        header["star_age"] = (1.0 - float(weight)) * float(left_header["star_age"]) + float(weight) * float(
            right_header["star_age"]
        )
    return header


def interpolated_columns(
    left_columns: dict[str, np.ndarray],
    right_columns: dict[str, np.ndarray],
    weight: float,
) -> dict[str, np.ndarray]:
    columns: dict[str, np.ndarray] = {}
    for key in INTERPOLATED_PROFILE_COLUMNS:
        if key not in left_columns or key not in right_columns:
            continue
        left = np.asarray(left_columns[key], dtype=float)
        right = np.asarray(right_columns[key], dtype=float)
        if key == "q":
            if not np.array_equal(left, right):
                raise RuntimeError("The q grid changed across the nonlinear hydro cycle; phase interpolation is unsafe.")
            columns[key] = left.copy()
        else:
            columns[key] = (1.0 - float(weight)) * left + float(weight) * right
    return columns


def interpolate_profile_at_phase(
    cycle_records: list[dict[str, object]],
    phase_sorted: np.ndarray,
    target_phase: float,
    profile_cache: dict[int, dict[str, object]],
) -> dict[str, object]:
    left_index, right_index, left_phase, right_phase = phase_bracket(phase_sorted, target_phase)
    left_profile = load_profile_cached(cycle_records, profile_cache, left_index)
    right_profile = load_profile_cached(cycle_records, profile_cache, right_index)

    denominator = right_phase - left_phase
    if denominator <= 0.0:
        weight = 0.0
    else:
        weight = (float(target_phase) - float(left_phase)) / denominator
    weight = min(max(float(weight), 0.0), 1.0)

    header = interpolated_header(
        left_profile["header"],
        right_profile["header"],
        weight,
    )
    columns = interpolated_columns(
        left_profile["columns"],
        right_profile["columns"],
        weight,
    )
    interpolated_age = (1.0 - weight) * float(left_profile["age_days"]) + weight * float(right_profile["age_days"])
    return {
        "header": header,
        "columns": columns,
        "age_days": interpolated_age,
        "left_age_days": float(left_profile["age_days"]),
        "right_age_days": float(right_profile["age_days"]),
        "left_columns": left_profile["columns"],
        "right_columns": right_profile["columns"],
        "left_path": str(left_profile["path"]),
        "right_path": str(right_profile["path"]),
        "left_phase": float(np.mod(left_phase, 1.0)),
        "right_phase": float(np.mod(right_phase, 1.0)),
        "weight": weight,
    }


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


def canonicalize_display_zone_spans(
    zone_spans: dict[str, list[tuple[float, float]]],
) -> dict[str, list[tuple[float, float]]]:
    display_zone_spans: dict[str, list[tuple[float, float]]] = {}
    if zone_spans.get("He II Ionization"):
        display_zone_spans["He II Ionization"] = merge_spans(zone_spans["He II Ionization"])

    outer_spans: list[tuple[float, float]] = []
    for name in ("H Ionization", "He I Ionization", "H/He I Ionization"):
        outer_spans.extend(zone_spans.get(name, []))
    if outer_spans:
        display_zone_spans["H/He I Ionization"] = merge_spans(outer_spans)

    if zone_spans.get("Convection"):
        display_zone_spans["Convection"] = merge_spans(zone_spans["Convection"])
    return display_zone_spans


def mean_light_zone_label_positions(
    mean_light_display_spans: dict[str, list[tuple[float, float]]],
) -> dict[str, tuple[float, str]]:
    positions: dict[str, tuple[float, str]] = {}
    if mean_light_display_spans.get("He II Ionization"):
        span = mean_light_display_spans["He II Ionization"][0]
        positions["He II Ionization"] = (geometric_midpoint(float(span[0]), float(span[1])), "He II")
    if mean_light_display_spans.get("H/He I Ionization"):
        span = mean_light_display_spans["H/He I Ionization"][0]
        positions["H/He I Ionization"] = (geometric_midpoint(float(span[0]), float(span[1])), "H / He I")
    return positions


def circular_weighted_average(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    kernel = np.asarray(weights, dtype=float)
    if array.ndim != 1:
        raise ValueError("circular_weighted_average expects a one-dimensional array.")
    if kernel.ndim != 1 or kernel.size % 2 == 0:
        raise ValueError("weights must be a one-dimensional odd-length kernel.")
    if array.size == 0:
        return array.copy()
    kernel = kernel / np.sum(kernel)
    radius = kernel.size // 2
    smoothed = np.zeros_like(array, dtype=float)
    for offset, weight in enumerate(kernel):
        shift = offset - radius
        smoothed += weight * np.roll(array, -shift)
    return smoothed


def smooth_display_zone_spans(
    frame_data: list[dict[str, object]],
    zone_names: tuple[str, ...] = ("He II Ionization", "H/He I Ionization", "Convection"),
) -> None:
    if not frame_data:
        return
    weights = np.ones(5, dtype=float)
    for zone_name in zone_names:
        if any(len(frame["display_zone_spans"].get(zone_name, [])) != 1 for frame in frame_data):
            continue
        hot_edges_logT = np.asarray(
            [math.log10(float(frame["display_zone_spans"][zone_name][0][0])) for frame in frame_data],
            dtype=float,
        )
        cool_edges_logT = np.asarray(
            [math.log10(float(frame["display_zone_spans"][zone_name][0][1])) for frame in frame_data],
            dtype=float,
        )
        hot_smoothed = circular_weighted_average(hot_edges_logT, weights)
        cool_smoothed = circular_weighted_average(cool_edges_logT, weights)
        for frame, hot_logT, cool_logT in zip(frame_data, hot_smoothed, cool_smoothed):
            hot_temperature = float(10.0 ** hot_logT)
            cool_temperature = float(10.0 ** cool_logT)
            frame["display_zone_spans"][zone_name] = [tuple(sorted((hot_temperature, cool_temperature)))]


def add_zone_overlays_single_axis(
    ax: plt.Axes,
    zone_spans: dict[str, list[tuple[float, float]]],
    label_positions: dict[str, tuple[float, str]],
    convection_profile: dict[str, object],
) -> None:
    text_transform = blended_transform_factory(ax.transData, ax.transAxes)
    convection_temperature = np.asarray(convection_profile["temperature_K"], dtype=float)
    convection_strength = np.asarray(convection_profile["strength"], dtype=float)
    convection_strength_max = max(float(convection_profile["max_strength"]), 1.0e-12)

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
        hatch_gray = 0.95 - 0.18 * span_strength
        ax.axvspan(
            x0,
            x1,
            facecolor=(0.75, 0.75, 0.75, 0.012),
            edgecolor=(hatch_gray, hatch_gray, hatch_gray, 0.72),
            hatch="/",
            linewidth=0.0,
            zorder=-6,
        )

    for name, spans in zone_spans.items():
        if name == "Convection" or not spans:
            continue
        color = COMPLEX_TRANSFER_REFERENCE_COLORS.get(name, "0.45")
        for x0, x1 in spans:
            ax.axvspan(
                x0,
                x1,
                facecolor=color,
                edgecolor="none",
                linewidth=0.0,
                alpha=0.26,
                zorder=-4,
            )
    for name, (label_x, label_text) in label_positions.items():
        color = COMPLEX_TRANSFER_REFERENCE_COLORS.get(name, ZONE_LABEL_COLOR)
        ax.text(
            label_x,
            0.98,
            label_text,
            rotation=90,
            ha="center",
            va="top",
            fontsize=7.5,
            color=color,
            transform=text_transform,
            path_effects=[pe.withStroke(linewidth=2.4, foreground="white")],
        )


def main() -> None:
    args = parse_args()

    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix

    gif_path = output_dir / f"{prefix}_luminosity_logT_phase_cycle.gif"
    png_path = output_dir / f"{prefix}_luminosity_logT_phase_cycle.png"
    summary_path = output_dir / f"{prefix}_luminosity_logT_phase_cycle_summary.json"
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
    cycle_photosphere_l = np.asarray(
        [float(record["photosphere_l_lsun"]) for record in cycle_records],
        dtype=float,
    )

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
        frame = instantaneous_zone_structure(
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

    mean_light_profile = load_mean_light_profile(run_dir, final_cycle_summary_path)
    mean_light_frame = instantaneous_zone_structure(
        mean_light_profile["header"],
        mean_light_profile["columns"],
        float(args.hot_limit),
    )
    label_positions = mean_light_zone_label_positions(mean_light_frame["display_zone_spans"])
    smooth_display_zone_spans(frame_data)

    outermost_temperature = float(
        min(float(frame["outermost_temperature_K"]) for frame in frame_data)
    )
    all_shell_luminosity = np.concatenate(
        [
            np.asarray(frame["luminosity_plot"], dtype=float)
            for frame in frame_data
        ]
        + [
            np.asarray(frame["luminosity_radiative_plot"], dtype=float)
            for frame in frame_data
        ]
        + [
            np.asarray(frame["luminosity_convective_plot"], dtype=float)
            for frame in frame_data
        ]
    )
    left_luminosity_ylim = fractional_padding(all_shell_luminosity, fraction=0.07)
    right_luminosity_ylim = fractional_padding(cycle_photosphere_l, fraction=0.08)

    cycle_phase_curve = np.append(cycle_phase, 1.0)
    cycle_luminosity_curve = np.append(cycle_photosphere_l, cycle_photosphere_l[0])
    cycle_phase_curve_two = np.concatenate([cycle_phase_curve, cycle_phase_curve[1:] + 1.0])
    cycle_luminosity_curve_two = np.concatenate([cycle_luminosity_curve, cycle_luminosity_curve[1:]])

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 11,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )

    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(13.0, 5.6),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.45, 1.0]},
    )

    ax_right.plot(
        cycle_phase_curve_two,
        cycle_luminosity_curve_two,
        color=PHASE_CURVE_COLOR,
        linewidth=1.55,
        zorder=1,
    )
    phase_dots, = ax_right.plot(
        [sampled_phase[0], sampled_phase[0] + 1.0],
        [float(frame_data[0]["photosphere_luminosity_lsun"]), float(frame_data[0]["photosphere_luminosity_lsun"])],
        marker="o",
        markersize=6.5,
        color=PHOTOSPHERE_COLOR,
        markeredgecolor="white",
        markeredgewidth=0.65,
        linestyle="None",
        zorder=3,
    )
    ax_right.set_xlabel("Pulsation phase")
    ax_right.set_ylabel(r"Photosphere $L\ [L_\odot]$")
    ax_right.set_xlim(0.0, 2.0)
    ax_right.set_ylim(*right_luminosity_ylim)
    ax_right.grid(False)

    def draw_left_panel(frame_index: int) -> tuple[float, float]:
        frame = frame_data[frame_index]
        temperature_plot = np.asarray(frame["temperature_plot"], dtype=float)
        luminosity_plot = np.asarray(frame["luminosity_plot"], dtype=float)
        luminosity_radiative_plot = np.asarray(frame["luminosity_radiative_plot"], dtype=float)
        luminosity_convective_plot = np.asarray(frame["luminosity_convective_plot"], dtype=float)
        photosphere_temperature = float(frame["photosphere_temperature_K"])
        photosphere_luminosity = float(frame["photosphere_luminosity_lsun"])

        ax_left.cla()
        add_zone_overlays_single_axis(
            ax_left,
            frame["display_zone_spans"],
            label_positions,
            frame["convection_profile"],
        )
        ax_left.plot(
            temperature_plot,
            luminosity_radiative_plot,
            color=RADIATIVE_COLOR,
            linewidth=1.35,
            alpha=0.95,
            zorder=1,
            label=r"$L_{\rm rad}$",
        )
        ax_left.plot(
            temperature_plot,
            luminosity_convective_plot,
            color=CONVECTIVE_COLOR,
            linewidth=1.35,
            alpha=0.95,
            zorder=1,
            label=r"$L_{\rm conv}$",
        )
        ax_left.plot(
            temperature_plot,
            luminosity_plot,
            color=SHELL_CURVE_COLOR,
            linewidth=1.8,
            zorder=2,
            label=r"$L$",
        )
        ax_left.axhline(0.0, color="k", linewidth=0.9, linestyle="--", alpha=0.75, zorder=-7)
        ax_left.plot(
            [photosphere_temperature],
            [photosphere_luminosity],
            marker="o",
            markersize=6.6,
            color=PHOTOSPHERE_COLOR,
            markeredgecolor="white",
            markeredgewidth=0.65,
            linestyle="None",
            zorder=4,
        )
        configure_temperature_axis([ax_left], temperature_plot)
        ax_left.set_xlim(float(args.hot_limit), outermost_temperature)
        ax_left.set_ylim(*left_luminosity_ylim)
        ax_left.set_xlabel("T [K]")
        ax_left.set_ylabel(r"$L\ [L_\odot]$")
        ax_left.grid(False)
        ax_left.legend(
            loc="upper left",
            frameon=False,
            fontsize=8,
            ncol=3,
            handlelength=1.8,
            columnspacing=1.0,
        )
        return photosphere_temperature, photosphere_luminosity

    def update(frame_index: int) -> tuple[object, ...]:
        _photosphere_temperature, photosphere_luminosity = draw_left_panel(frame_index)
        current_phase = float(sampled_phase[frame_index])
        phase_dots.set_data([current_phase, current_phase + 1.0], [photosphere_luminosity, photosphere_luminosity])
        return (phase_dots,)

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frame_data),
        interval=1000.0 / max(int(args.fps), 1),
        blit=False,
    )
    animation.save(gif_path, writer=PillowWriter(fps=max(int(args.fps), 1)))

    update(len(frame_data) - 1)
    fig.savefig(png_path, dpi=220)
    plt.close(fig)

    summary = {
        "prefix": prefix,
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "gif_path": str(gif_path),
        "png_path": str(png_path),
        "cycle_source": cycle_source,
        "frame_generation": "fixed-q linear interpolation between bracketing nonlinear profiles on a uniform phase grid",
        "cycle_profile_count": len(cycle_records),
        "frame_count": len(frame_data),
        "fps": int(args.fps),
        "period_days_used": float(period_days),
        "phase_reference_days_used": float(phase_reference_days),
        "hot_temperature_limit_K": float(args.hot_limit),
        "outermost_temperature_limit_K": outermost_temperature,
        "left_luminosity_ylim": [float(left_luminosity_ylim[0]), float(left_luminosity_ylim[1])],
        "right_luminosity_ylim": [float(right_luminosity_ylim[0]), float(right_luminosity_ylim[1])],
        "mean_light_label_positions_temperature_K": {
            name: float(position[0]) for name, position in label_positions.items()
        },
        "sampled_frame_paths": [str(frame["path"]) for frame in frame_data],
        "sampled_frame_left_paths": [str(frame["left_path"]) for frame in frame_data],
        "sampled_frame_right_paths": [str(frame["right_path"]) for frame in frame_data],
        "sampled_frame_interpolation_weights": [float(frame["interpolation_weight"]) for frame in frame_data],
        "sampled_phases": [float(value) for value in sampled_phase],
    }
    summary_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
