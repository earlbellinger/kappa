from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shlex
import sys
from pathlib import Path

WORKSPACE_VENDOR = Path(__file__).resolve().parent / ".vendor"
if WORKSPACE_VENDOR.exists():
    sys.path.insert(0, str(WORKSPACE_VENDOR))

from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory
import matplotlib.pyplot as plt
import numpy as np

TWOPI = 2.0 * math.pi
PROFILE_RE = re.compile(r"profile(\d+)\.data$")
PHOTOSPHERE_TAU = 2.0 / 3.0
FIT_PHASE_SAMPLES = 512

LINE_COLORS = {
    "A1": "#1f77b4",
    "A2": "#ff7f0e",
    "A3": "#2ca02c",
    "phi21": "#9467bd",
    "phi31": "#d62728",
}

ZONE_STYLES = {
    "He II Ionization": {"color": "#55a868", "alpha": 0.14},
    "He I Ionization": {"color": "#dd8452", "alpha": 0.14},
    "H Ionization": {"color": "#c44e52", "alpha": 0.14},
    "H/He I Ionization": {"color": "#8172b3", "alpha": 0.14},
    "Convection": {"color": "0.65", "alpha": 0.20},
}

DEFAULT_FIT_HARMONICS = 14


def numeric_value(text: str) -> float:
    return float(text.replace("D", "E").replace("d", "e"))


def profile_key(path: Path) -> int:
    match = PROFILE_RE.match(path.name)
    if not match:
        raise ValueError(f"Unexpected profile filename: {path}")
    return int(match.group(1))


def parse_profile(path: Path) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    lines = path.read_text().splitlines()

    header_names_idx = next(i for i, line in enumerate(lines) if line.strip().startswith("model_number"))
    header_names = lines[header_names_idx].split()
    header_values = shlex.split(lines[header_names_idx + 1])
    header: dict[str, float] = {}
    for name, value in zip(header_names, header_values):
        try:
            header[name] = numeric_value(value)
        except ValueError:
            continue

    zone_names_idx = next(i for i in range(header_names_idx + 1, len(lines)) if lines[i].strip().startswith("zone"))
    column_names = lines[zone_names_idx].split()

    rows: list[list[float]] = []
    for line in lines[zone_names_idx + 1 :]:
        parts = line.split()
        if len(parts) != len(column_names):
            continue
        rows.append([numeric_value(part) for part in parts])

    data = np.asarray(rows, dtype=float)
    columns = {name: data[:, i] for i, name in enumerate(column_names)}
    return header, columns


def interpolate_at_coordinate(coordinate: np.ndarray, values: np.ndarray, target_coordinate: float) -> float:
    order = np.argsort(coordinate)
    coordinate_sorted = coordinate[order]
    values_sorted = values[order]
    return float(np.interp(target_coordinate, coordinate_sorted, values_sorted))


def profile_photosphere_state(
    header: dict[str, float],
    columns: dict[str, np.ndarray],
    target_tau: float = PHOTOSPHERE_TAU,
) -> dict[str, float]:
    tau = columns["tau"]
    return {
        "tau": float(target_tau),
        "q_env": interpolate_at_coordinate(tau, columns["q"], target_tau),
        "radius_rsun": float(header.get("photosphere_r", interpolate_at_coordinate(tau, columns["radius"], target_tau))),
        "luminosity_lsun": float(
            header.get("photosphere_L", interpolate_at_coordinate(tau, columns["luminosity"], target_tau))
        ),
        "velocity_km_per_s": interpolate_at_coordinate(tau, columns["vel_km_per_s"], target_tau),
        "logT": interpolate_at_coordinate(tau, columns["logT"], target_tau),
    }


def find_local_maxima(values: np.ndarray) -> np.ndarray:
    return np.where((values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:]))[0] + 1


def find_local_minima(values: np.ndarray) -> np.ndarray:
    return np.where((values[1:-1] < values[:-2]) & (values[1:-1] <= values[2:]))[0] + 1


def load_period_and_phase_reference(
    summary_path: Path,
    explicit_period_days: float | None,
    absolute_time_days: np.ndarray,
    photosphere_radius: np.ndarray,
) -> tuple[float, float]:
    if explicit_period_days is not None:
        if summary_path.exists():
            summary = load_json(summary_path)
            phase_reference_days = float(summary.get("max_light_age_days", absolute_time_days[0]))
        else:
            phase_reference_days = float(absolute_time_days[0])
        return float(explicit_period_days), phase_reference_days

    if summary_path.exists():
        summary = load_json(summary_path)
        return float(summary["period_days"]), float(summary.get("max_light_age_days", absolute_time_days[0]))

    maxima = find_local_maxima(photosphere_radius)
    if maxima.size < 2:
        raise RuntimeError("Could not infer the pulsation period from the photosphere radius maxima.")
    period_days = float(absolute_time_days[maxima[-1]] - absolute_time_days[maxima[-2]])
    phase_reference_days = float(absolute_time_days[maxima[-1]])
    return period_days, phase_reference_days


def build_fourier_design_matrix(phase: np.ndarray, fit_harmonics: int) -> np.ndarray:
    columns = [np.ones_like(phase)]
    for harmonic in range(1, fit_harmonics + 1):
        angle = harmonic * TWOPI * phase
        columns.append(np.cos(angle))
        columns.append(np.sin(angle))
    return np.column_stack(columns)


def wrap_2pi(values: np.ndarray) -> np.ndarray:
    return np.mod(values, TWOPI)


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.full_like(numerator, np.nan, dtype=float)
    good = np.abs(denominator) > 0.0
    out[good] = numerator[good] / denominator[good]
    return out


def analyze_signal_stack(
    signal_stack: np.ndarray,
    absolute_time_days: np.ndarray,
    phase_reference_days: float,
    period_days: float,
    fit_harmonics: int,
) -> dict[str, object]:
    phase = np.mod((absolute_time_days - phase_reference_days) / period_days, 1.0)
    design_matrix = build_fourier_design_matrix(phase, fit_harmonics)
    coefficients, *_ = np.linalg.lstsq(design_matrix, signal_stack.T, rcond=None)

    cos_coefficients = coefficients[1::2, :]
    sin_coefficients = coefficients[2::2, :]
    harmonic_amplitudes = np.hypot(cos_coefficients, sin_coefficients)
    harmonic_phases = wrap_2pi(np.arctan2(cos_coefficients, sin_coefficients))

    fitted_signal = (design_matrix @ coefficients).T

    A1 = harmonic_amplitudes[0]
    A2 = harmonic_amplitudes[1] if fit_harmonics >= 2 else np.full_like(A1, np.nan)
    A3 = harmonic_amplitudes[2] if fit_harmonics >= 3 else np.full_like(A1, np.nan)
    phi1 = harmonic_phases[0]
    phi2 = harmonic_phases[1] if fit_harmonics >= 2 else np.full_like(phi1, np.nan)
    phi3 = harmonic_phases[2] if fit_harmonics >= 3 else np.full_like(phi1, np.nan)
    R21 = safe_ratio(A2, A1)
    R31 = safe_ratio(A3, A1)
    phi21 = wrap_2pi(phi2 - 2.0 * phi1)
    phi31 = wrap_2pi(phi3 - 3.0 * phi1)

    rms_residual = np.sqrt(np.mean(np.square(signal_stack - fitted_signal), axis=1))
    signal_rms = np.sqrt(np.mean(np.square(signal_stack), axis=1))
    fit_r2 = 1.0 - safe_ratio(
        np.mean(np.square(signal_stack - fitted_signal), axis=1),
        np.mean(np.square(signal_stack), axis=1),
    )

    fit_phase = np.linspace(0.0, 1.0, FIT_PHASE_SAMPLES, endpoint=False)
    return {
        "fit_phase": fit_phase,
        "parameters": {
            "A1": A1,
            "A2": A2,
            "A3": A3,
            "phi1": phi1,
            "phi2": phi2,
            "phi3": phi3,
            "R21": R21,
            "R31": R31,
            "phi21": phi21,
            "phi31": phi31,
            "rms_residual": rms_residual,
            "signal_rms": signal_rms,
            "fit_r2": fit_r2,
            "fit_harmonics": np.full_like(A1, fit_harmonics, dtype=float),
        },
    }


def cell_edges(coordinate: np.ndarray) -> np.ndarray:
    if coordinate.size < 2:
        raise ValueError("At least two coordinate values are required.")
    edges = np.empty(coordinate.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (coordinate[:-1] + coordinate[1:])
    edges[0] = coordinate[0] - 0.5 * (coordinate[1] - coordinate[0])
    edges[-1] = coordinate[-1] + 0.5 * (coordinate[-1] - coordinate[-2])
    return edges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot shell Fourier amplitudes and phases against log10(1-m/M) using only RSP profiles."
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory containing LOGS/profile*.data")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for the figure and table outputs")
    parser.add_argument("--prefix", required=True, help="Output filename prefix, e.g. mesa_rsp_combined_14507")
    parser.add_argument(
        "--period-days",
        type=float,
        default=None,
        help="Override the pulsation period in days. Defaults to the saved cycle summary or an inferred value.",
    )
    parser.add_argument(
        "--fit-harmonics",
        type=int,
        default=None,
        help="Fourier order to fit. Defaults to the existing depth-scan summary or 14.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def break_wrapped_series(x_values: np.ndarray, y_values: np.ndarray, wrap: float = TWOPI) -> tuple[np.ndarray, np.ndarray]:
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


def contiguous_spans(edges: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []

    spans: list[tuple[float, float]] = []
    split_points = np.where(np.diff(indices) > 1)[0] + 1
    for group in np.split(indices, split_points):
        spans.append((float(edges[group[0]]), float(edges[group[-1] + 1])))
    return spans


def spans_from_mask(x_descending: np.ndarray, mask_descending: np.ndarray) -> list[tuple[float, float]]:
    x_desc = np.asarray(x_descending, dtype=float)
    mask_desc = np.asarray(mask_descending, dtype=bool)
    if x_desc.size != mask_desc.size:
        raise ValueError("Coordinate and mask arrays must have the same length.")

    x_ascending = x_desc[::-1]
    mask_ascending = mask_desc[::-1]
    edges = cell_edges(x_ascending)
    return [(min(x0, x1), max(x0, x1)) for x0, x1 in contiguous_spans(edges, mask_ascending)]


def normalize_visible(values: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, dict[str, float | None]]:
    normalized = np.full_like(values, np.nan, dtype=float)
    finite_mask = valid_mask & np.isfinite(values)
    if not np.any(finite_mask):
        return normalized, {"min": None, "max": None, "range": None}

    vmin = float(np.nanmin(values[finite_mask]))
    vmax = float(np.nanmax(values[finite_mask]))
    vrange = float(vmax - vmin)
    if vrange <= 0.0:
        return normalized, {"min": vmin, "max": vmax, "range": vrange}

    normalized[finite_mask] = (values[finite_mask] - vmin) / vrange
    return normalized, {"min": vmin, "max": vmax, "range": vrange}


def normalized_transition_mask(
    values: np.ndarray,
    valid_mask: np.ndarray,
    low_fraction: float = 0.05,
    high_fraction: float = 0.95,
    min_range: float = 1.0e-6,
) -> tuple[np.ndarray, dict[str, object]]:
    normalized, stats = normalize_visible(values, valid_mask)
    mask = np.zeros_like(valid_mask, dtype=bool)
    detected = stats["range"] is not None and float(stats["range"]) > min_range
    if detected:
        mask = valid_mask & np.isfinite(normalized) & (normalized >= low_fraction) & (normalized <= high_fraction)

    details = {
        "min": stats["min"],
        "max": stats["max"],
        "range": stats["range"],
        "fraction_bounds": [float(low_fraction), float(high_fraction)],
        "detected": bool(detected and np.any(mask)),
    }
    return mask, details


def build_mask_from_local_slice(global_indices: np.ndarray, start_local: int, end_local: int, size: int) -> np.ndarray:
    mask = np.zeros(size, dtype=bool)
    mask[global_indices[start_local : end_local + 1]] = True
    return mask


def infer_outer_partial_ionization_spans(
    massdepth_sorted: np.ndarray,
    mean_cp_sorted: np.ndarray,
    mean_gamma1_sorted: np.ndarray,
    visible_mask: np.ndarray,
    excluded_mask: np.ndarray,
) -> tuple[dict[str, list[tuple[float, float]]], dict[str, object]]:
    thermodynamic_mask = (
        visible_mask
        & ~excluded_mask
        & np.isfinite(mean_cp_sorted)
        & np.isfinite(mean_gamma1_sorted)
        & (mean_cp_sorted > 0.0)
    )
    if np.count_nonzero(thermodynamic_mask) < 5:
        return {}, {"detected_labels": [], "message": "Insufficient thermodynamic samples outside the He II zone."}

    global_indices = np.flatnonzero(thermodynamic_mask)
    log_cp = np.log10(np.clip(mean_cp_sorted[thermodynamic_mask], 1.0e-99, None))
    cp_norm, cp_stats = normalize_visible(log_cp, np.ones(log_cp.size, dtype=bool))
    gamma_deficit = np.nanmax(mean_gamma1_sorted[thermodynamic_mask]) - mean_gamma1_sorted[thermodynamic_mask]
    gamma_norm, gamma_stats = normalize_visible(gamma_deficit, np.ones(gamma_deficit.size, dtype=bool))
    susceptibility = cp_norm + gamma_norm

    peak_indices = find_local_maxima(susceptibility)
    if peak_indices.size == 0:
        peak_indices = np.asarray([int(np.nanargmax(susceptibility))], dtype=int)

    peak_strength = susceptibility[peak_indices]
    strength_floor = 0.5 * float(np.nanmax(susceptibility))
    selected_peaks = peak_indices[peak_strength >= strength_floor]
    if selected_peaks.size == 0:
        selected_peaks = np.asarray([int(peak_indices[np.nanargmax(peak_strength)])], dtype=int)
    elif selected_peaks.size > 2:
        order = np.argsort(susceptibility[selected_peaks])[::-1][:2]
        selected_peaks = selected_peaks[order]

    selected_peaks = np.sort(selected_peaks)[::-1]
    minima_indices = find_local_minima(susceptibility)

    spans: dict[str, list[tuple[float, float]]] = {}
    peak_records: list[dict[str, float]] = []
    if selected_peaks.size == 1:
        labels = ["H/He I Ionization"]
    else:
        labels = ["H Ionization", "He I Ionization"][: selected_peaks.size]

    for label, peak in zip(labels, selected_peaks):
        left_candidates = minima_indices[minima_indices < peak]
        right_candidates = minima_indices[minima_indices > peak]
        left_bound = int(left_candidates[-1]) if left_candidates.size else 0
        right_bound = int(right_candidates[0]) if right_candidates.size else susceptibility.size - 1
        zone_mask = build_mask_from_local_slice(global_indices, left_bound, right_bound, massdepth_sorted.size)
        spans[label] = spans_from_mask(massdepth_sorted, zone_mask)
        peak_records.append(
            {
                "label": label,
                "peak_massdepth": float(massdepth_sorted[global_indices[peak]]),
                "peak_susceptibility": float(susceptibility[peak]),
                "left_massdepth": float(massdepth_sorted[global_indices[left_bound]]),
                "right_massdepth": float(massdepth_sorted[global_indices[right_bound]]),
            }
        )

    details = {
        "detected_labels": labels,
        "selection_threshold": float(strength_floor),
        "log10_cp_range": cp_stats["range"],
        "gamma1_deficit_range": gamma_stats["range"],
        "peak_records": peak_records,
        "message": "Outer partial-ionization spans inferred from phase-averaged cp peaks and gamma1 dips because the explicit H/He I charge fields are flat in these RSP profiles.",
    }
    return spans, details


def load_profile_cycle(run_dir: Path) -> dict[str, np.ndarray | float | Path]:
    logs_dir = run_dir / "LOGS"
    profile_paths = sorted(logs_dir.glob("profile*.data"), key=profile_key)
    if not profile_paths:
        raise RuntimeError(f"No profile files found in {logs_dir}")

    first_header, first_columns = parse_profile(profile_paths[0])
    q_grid = np.asarray(first_columns["q"], dtype=float)
    n_profiles = len(profile_paths)
    n_zones = q_grid.size

    absolute_time_days = np.empty(n_profiles, dtype=float)
    luminosity = np.empty((n_profiles, n_zones), dtype=float)
    log_temperature = np.empty((n_profiles, n_zones), dtype=float)
    pressure = np.empty((n_profiles, n_zones), dtype=float)
    density = np.empty((n_profiles, n_zones), dtype=float)
    convective_fraction = np.empty((n_profiles, n_zones), dtype=float)
    gamma1 = np.empty((n_profiles, n_zones), dtype=float)
    cp = np.empty((n_profiles, n_zones), dtype=float)

    optional_field_names = [
        name
        for name in ("typical_charge_h1", "ionization_h1", "typical_charge_he4", "ionization_he4")
        if name in first_columns
    ]
    optional_matrices = {name: np.empty((n_profiles, n_zones), dtype=float) for name in optional_field_names}

    photosphere_q_env = np.empty(n_profiles, dtype=float)
    photosphere_radius = np.empty(n_profiles, dtype=float)

    total_mass_msun = float(first_header["star_mass"])
    core_mass_msun = float(first_header["M_center"] / first_header["msun"])
    envelope_mass_msun = total_mass_msun - core_mass_msun

    for i, path in enumerate(profile_paths):
        header, columns = parse_profile(path)
        if not np.array_equal(columns["q"], q_grid):
            raise RuntimeError(f"The q grid changed in {path.name}; this plot assumes a fixed Lagrangian mesh.")

        lsun_cgs = float(header["lsun"])
        absolute_time_days[i] = float(header["star_age"]) * 365.25
        luminosity[i] = np.asarray(columns["luminosity"], dtype=float)
        log_temperature[i] = np.asarray(columns["logT"], dtype=float)
        pressure[i] = np.asarray(columns["pressure"], dtype=float)
        density[i] = np.power(10.0, np.asarray(columns["logRho"], dtype=float))
        gamma1[i] = np.asarray(columns["gamma1"], dtype=float)
        cp[i] = np.asarray(columns["cp"], dtype=float)
        if "rsp_Lc_div_L" in columns:
            convective_fraction[i] = np.asarray(columns["rsp_Lc_div_L"], dtype=float)
        else:
            convective_fraction[i] = np.divide(
                np.asarray(columns["rsp_Lc"], dtype=float),
                np.asarray(columns["luminosity"], dtype=float) * lsun_cgs,
                out=np.full(n_zones, np.nan, dtype=float),
                where=np.abs(np.asarray(columns["luminosity"], dtype=float)) > 0.0,
            )

        for name, matrix in optional_matrices.items():
            matrix[i] = np.asarray(columns[name], dtype=float)

        photosphere = profile_photosphere_state(header, columns)
        photosphere_q_env[i] = float(photosphere["q_env"])
        photosphere_radius[i] = float(photosphere["radius_rsun"])

    q_order = np.argsort(q_grid)
    q_env_sorted = q_grid[q_order]
    q_full_sorted = (core_mass_msun + q_env_sorted * envelope_mass_msun) / total_mass_msun
    massdepth_sorted = np.log10(np.clip(1.0 - q_full_sorted, 1.0e-99, None))

    photosphere_q_full = (core_mass_msun + photosphere_q_env * envelope_mass_msun) / total_mass_msun
    photosphere_massdepth = np.log10(np.clip(1.0 - photosphere_q_full, 1.0e-99, None))

    result: dict[str, np.ndarray | float | Path] = {
        "run_dir": run_dir,
        "logs_dir": logs_dir,
        "first_profile_path": profile_paths[0],
        "absolute_time_days": absolute_time_days,
        "luminosity_sorted": luminosity[:, q_order],
        "log_temperature_sorted": log_temperature[:, q_order],
        "pressure_sorted": pressure[:, q_order],
        "density_sorted": density[:, q_order],
        "convective_fraction_sorted": convective_fraction[:, q_order],
        "gamma1_sorted": gamma1[:, q_order],
        "cp_sorted": cp[:, q_order],
        "q_env_sorted": q_env_sorted,
        "q_full_sorted": q_full_sorted,
        "massdepth_sorted": massdepth_sorted,
        "photosphere_radius": photosphere_radius,
        "photosphere_q_full": photosphere_q_full,
        "photosphere_massdepth": photosphere_massdepth,
        "total_mass_msun": total_mass_msun,
        "core_mass_msun": core_mass_msun,
        "envelope_mass_msun": envelope_mass_msun,
    }
    for name, matrix in optional_matrices.items():
        result[f"{name}_sorted"] = matrix[:, q_order]
    return result


def determine_fit_harmonics(output_dir: Path, prefix: str, explicit_fit_harmonics: int | None) -> int:
    if explicit_fit_harmonics is not None:
        return int(explicit_fit_harmonics)

    fourier_summary_path = output_dir / f"{prefix}_fourier_depth_scan_summary.json"
    if fourier_summary_path.exists():
        summary = load_json(fourier_summary_path)
        fit_harmonics = summary.get("fit_harmonics")
        if fit_harmonics is not None:
            return int(fit_harmonics)
    return DEFAULT_FIT_HARMONICS


def build_zone_spans(
    massdepth_sorted: np.ndarray,
    mean_cp_sorted: np.ndarray,
    mean_gamma1_sorted: np.ndarray,
    mean_abs_convective_fraction_sorted: np.ndarray,
    visible_mask: np.ndarray,
    mean_typical_charge_h1_sorted: np.ndarray | None = None,
    mean_ionization_h1_sorted: np.ndarray | None = None,
    mean_typical_charge_he4_sorted: np.ndarray | None = None,
    mean_ionization_he4_sorted: np.ndarray | None = None,
) -> tuple[dict[str, list[tuple[float, float]]], float, dict[str, object]]:
    spans: dict[str, list[tuple[float, float]]] = {}
    zone_details: dict[str, object] = {}

    heii_mask = np.zeros_like(visible_mask, dtype=bool)
    field_details: dict[str, object] = {}
    for field_name, mean_field in (
        ("typical_charge_h1", mean_typical_charge_h1_sorted),
        ("ionization_h1", mean_ionization_h1_sorted),
        ("typical_charge_he4", mean_typical_charge_he4_sorted),
        ("ionization_he4", mean_ionization_he4_sorted),
    ):
        if mean_field is None:
            field_details[field_name] = {"present": False, "detected": False}
            continue
        _, details = normalized_transition_mask(mean_field, visible_mask)
        field_details[field_name] = {"present": True, **details}

    if mean_ionization_he4_sorted is not None:
        heii_mask, heii_details = normalized_transition_mask(mean_ionization_he4_sorted, visible_mask)
        spans["He II Ionization"] = spans_from_mask(massdepth_sorted, heii_mask)
        zone_details["He II Ionization"] = {
            "source": "phase-averaged ionization_he4 on the profile mesh",
            **heii_details,
        }
    else:
        zone_details["He II Ionization"] = {"source": "ionization_he4 not present", "detected": False}

    outer_spans, outer_details = infer_outer_partial_ionization_spans(
        massdepth_sorted=massdepth_sorted,
        mean_cp_sorted=mean_cp_sorted,
        mean_gamma1_sorted=mean_gamma1_sorted,
        visible_mask=visible_mask,
        excluded_mask=heii_mask,
    )
    spans.update(outer_spans)
    for label in outer_details.get("detected_labels", []):
        zone_details[label] = {
            "source": "phase-averaged cp and gamma1 on the profile mesh",
            **outer_details,
        }
    zone_details["profile_field_ranges"] = field_details

    visible_conv = mean_abs_convective_fraction_sorted[visible_mask]
    if visible_conv.size == 0 or not np.any(np.isfinite(visible_conv)):
        convective_threshold = math.nan
        spans["Convection"] = []
    else:
        convective_peak = float(np.nanmax(visible_conv))
        convective_threshold = max(1.0e-3, 0.05 * convective_peak)
        convection_mask = (
            visible_mask
            & np.isfinite(mean_abs_convective_fraction_sorted)
            & (mean_abs_convective_fraction_sorted >= convective_threshold)
        )
        spans["Convection"] = spans_from_mask(massdepth_sorted, convection_mask)

    zone_details["Convection"] = {
        "source": "phase-averaged |Lc|/L on the profile mesh",
        "threshold_abs_Lc_over_L": None if math.isnan(convective_threshold) else float(convective_threshold),
    }

    return spans, convective_threshold, zone_details


def write_csv(
    path: Path,
    x_plot: np.ndarray,
    q_env_plot: np.ndarray,
    q_full_plot: np.ndarray,
    mean_logT_plot: np.ndarray,
    mean_abs_conv_plot: np.ndarray,
    parameters: dict[str, np.ndarray],
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "log10_one_minus_m_over_M",
                "q_env",
                "q_full",
                "mean_log10_T",
                "mean_abs_Lc_over_L",
                "A1",
                "A2",
                "A3",
                "phi21_rad",
                "phi31_rad",
                "signal_rms",
                "fit_r2",
            ]
        )
        for row in zip(
            x_plot,
            q_env_plot,
            q_full_plot,
            mean_logT_plot,
            mean_abs_conv_plot,
            parameters["A1"],
            parameters["A2"],
            parameters["A3"],
            parameters["phi21"],
            parameters["phi31"],
            parameters["signal_rms"],
            parameters["fit_r2"],
        ):
            writer.writerow([f"{float(value):.12g}" for value in row])


def main() -> None:
    args = parse_args()

    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix

    png_path = output_dir / f"{prefix}_fourier_vs_massdepth_profiles.png"
    csv_path = output_dir / f"{prefix}_fourier_vs_massdepth_profiles.csv"
    summary_path = output_dir / f"{prefix}_fourier_vs_massdepth_profiles_summary.json"
    final_cycle_summary_path = output_dir / f"{prefix}_final_cycle_summary.json"

    cycle = load_profile_cycle(run_dir)
    fit_harmonics = determine_fit_harmonics(output_dir, prefix, args.fit_harmonics)
    period_days, phase_reference_days = load_period_and_phase_reference(
        final_cycle_summary_path,
        args.period_days,
        np.asarray(cycle["absolute_time_days"], dtype=float),
        np.asarray(cycle["photosphere_radius"], dtype=float),
    )

    shell_luminosity = np.asarray(cycle["luminosity_sorted"], dtype=float).T
    shell_mean_luminosity = np.mean(shell_luminosity, axis=1, keepdims=True)
    shell_signal = shell_luminosity / shell_mean_luminosity - 1.0
    full_analysis = analyze_signal_stack(
        signal_stack=shell_signal,
        absolute_time_days=np.asarray(cycle["absolute_time_days"], dtype=float),
        phase_reference_days=phase_reference_days,
        period_days=period_days,
        fit_harmonics=fit_harmonics,
    )
    parameters_all = {name: np.asarray(values, dtype=float) for name, values in full_analysis["parameters"].items()}

    massdepth_sorted = np.asarray(cycle["massdepth_sorted"], dtype=float)
    q_env_sorted = np.asarray(cycle["q_env_sorted"], dtype=float)
    q_full_sorted = np.asarray(cycle["q_full_sorted"], dtype=float)
    mean_logT_sorted = np.mean(np.asarray(cycle["log_temperature_sorted"], dtype=float), axis=0)
    mean_abs_conv_sorted = np.mean(np.abs(np.asarray(cycle["convective_fraction_sorted"], dtype=float)), axis=0)
    mean_gamma1_sorted = np.mean(np.asarray(cycle["gamma1_sorted"], dtype=float), axis=0)
    mean_cp_sorted = np.mean(np.asarray(cycle["cp_sorted"], dtype=float), axis=0)
    mean_typical_charge_h1_sorted = (
        np.mean(np.asarray(cycle["typical_charge_h1_sorted"], dtype=float), axis=0)
        if "typical_charge_h1_sorted" in cycle
        else None
    )
    mean_ionization_h1_sorted = (
        np.mean(np.asarray(cycle["ionization_h1_sorted"], dtype=float), axis=0)
        if "ionization_h1_sorted" in cycle
        else None
    )
    mean_typical_charge_he4_sorted = (
        np.mean(np.asarray(cycle["typical_charge_he4_sorted"], dtype=float), axis=0)
        if "typical_charge_he4_sorted" in cycle
        else None
    )
    mean_ionization_he4_sorted = (
        np.mean(np.asarray(cycle["ionization_he4_sorted"], dtype=float), axis=0)
        if "ionization_he4_sorted" in cycle
        else None
    )

    photosphere_massdepth_mean = float(np.mean(np.asarray(cycle["photosphere_massdepth"], dtype=float)))
    photosphere_q_full_mean = float(np.mean(np.asarray(cycle["photosphere_q_full"], dtype=float)))
    visible_mask = np.isfinite(massdepth_sorted) & (q_full_sorted <= photosphere_q_full_mean)
    if not np.any(visible_mask):
        raise RuntimeError("No interior zones were found beneath the mean photosphere.")

    zone_spans, convective_threshold, zone_details = build_zone_spans(
        massdepth_sorted=massdepth_sorted,
        mean_cp_sorted=mean_cp_sorted,
        mean_gamma1_sorted=mean_gamma1_sorted,
        mean_abs_convective_fraction_sorted=mean_abs_conv_sorted,
        visible_mask=visible_mask,
        mean_typical_charge_h1_sorted=mean_typical_charge_h1_sorted,
        mean_ionization_h1_sorted=mean_ionization_h1_sorted,
        mean_typical_charge_he4_sorted=mean_typical_charge_he4_sorted,
        mean_ionization_he4_sorted=mean_ionization_he4_sorted,
    )

    x_plot = massdepth_sorted[visible_mask]
    q_env_plot = q_env_sorted[visible_mask]
    q_full_plot = q_full_sorted[visible_mask]
    mean_logT_plot = mean_logT_sorted[visible_mask]
    mean_abs_conv_plot = mean_abs_conv_sorted[visible_mask]
    parameters_plot = {name: values[visible_mask] for name, values in parameters_all.items()}

    phi21_x, phi21_y = break_wrapped_series(x_plot, parameters_plot["phi21"])
    phi31_x, phi31_y = break_wrapped_series(x_plot, parameters_plot["phi31"])

    write_csv(
        path=csv_path,
        x_plot=x_plot,
        q_env_plot=q_env_plot,
        q_full_plot=q_full_plot,
        mean_logT_plot=mean_logT_plot,
        mean_abs_conv_plot=mean_abs_conv_plot,
        parameters=parameters_plot,
    )

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

    fig, (ax_amp, ax_phase) = plt.subplots(2, 1, figsize=(9.2, 6.9), sharex=True, constrained_layout=True)
    axes = [ax_amp, ax_phase]

    zone_handles: list[Patch] = []
    for name in ("He II Ionization", "He I Ionization", "H Ionization", "H/He I Ionization", "Convection"):
        style = ZONE_STYLES[name]
        spans = zone_spans.get(name, [])
        if not spans:
            continue
        for x0, x1 in spans:
            for ax in axes:
                ax.axvspan(x0, x1, color=style["color"], alpha=style["alpha"], zorder=-3)
        zone_handles.append(Patch(facecolor=style["color"], edgecolor="none", alpha=style["alpha"], label=name))

    ax_amp.plot(x_plot, parameters_plot["A1"], color=LINE_COLORS["A1"], linewidth=1.6, label=r"$A_1$")
    ax_amp.plot(x_plot, parameters_plot["A2"], color=LINE_COLORS["A2"], linewidth=1.4, label=r"$A_2$")
    ax_amp.plot(x_plot, parameters_plot["A3"], color=LINE_COLORS["A3"], linewidth=1.4, label=r"$A_3$")
    ax_amp.set_ylabel("Amplitude")
    ax_amp.set_title("Fourier Amplitudes")
    ax_amp.grid(alpha=0.2)
    line_legend = ax_amp.legend(loc="upper left", ncol=3, frameon=True, framealpha=0.9)
    if zone_handles:
        zone_legend = ax_amp.legend(handles=zone_handles, loc="upper center", ncol=2, frameon=True, framealpha=0.9)
        ax_amp.add_artist(line_legend)
        zone_legend.set_title("Phase-Averaged Zones")

    ax_phase.plot(phi21_x, phi21_y, color=LINE_COLORS["phi21"], linewidth=1.5, label=r"$\phi_{21}$")
    ax_phase.plot(phi31_x, phi31_y, color=LINE_COLORS["phi31"], linewidth=1.5, label=r"$\phi_{31}$")
    ax_phase.set_ylabel("Phase [rad]")
    ax_phase.set_xlabel(r"$\log_{10}(1 - m/M)$")
    ax_phase.set_title("Fourier Phase Differences")
    ax_phase.set_ylim(0.0, TWOPI)
    ax_phase.grid(alpha=0.2)
    ax_phase.legend(loc="upper left", ncol=2, frameon=True, framealpha=0.9)

    for ax in axes:
        ax.axvline(photosphere_massdepth_mean, color="0.2", linewidth=1.2, linestyle=":", zorder=-2)
    text_transform = blended_transform_factory(ax_amp.transData, ax_amp.transAxes)
    ax_amp.text(
        photosphere_massdepth_mean,
        0.98,
        "Photosphere",
        rotation=90,
        ha="left",
        va="top",
        color="0.2",
        fontsize=8,
        transform=text_transform,
    )

    ax_phase.set_xlim(float(np.max(x_plot)), photosphere_massdepth_mean)

    fig.suptitle(f"{prefix.replace('_', ' ')}: Fourier Diagnostics vs. Mass Depth")
    fig.savefig(png_path, dpi=220)
    plt.close(fig)

    summary = {
        "prefix": prefix,
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "png_path": str(png_path),
        "csv_path": str(csv_path),
        "period_days": float(period_days),
        "phase_reference_days": float(phase_reference_days),
        "fit_harmonics": int(fit_harmonics),
        "photosphere_log10_one_minus_m_over_M_mean": photosphere_massdepth_mean,
        "photosphere_q_full_mean": photosphere_q_full_mean,
        "massdepth_xlim": [float(np.max(x_plot)), photosphere_massdepth_mean],
        "ionization_zone_source": "phase-averaged RSP profile ionization and thermodynamic fields on the profile mesh",
        "zone_detection_details": zone_details,
        "convective_zone_source": "phase-averaged |Lc|/L on the profile mesh",
        "convective_threshold_abs_Lc_over_L": None if math.isnan(convective_threshold) else float(convective_threshold),
        "zone_spans_log10_one_minus_m_over_M": {
            name: [[float(x0), float(x1)] for x0, x1 in spans] for name, spans in zone_spans.items()
        },
        "fourier_phase_convention": "y = A0 + sum_k A_k sin(2 pi k phase + phi_k)",
        "amplitude_convention": "fractional shell luminosity amplitude relative to the shell mean luminosity",
        "num_visible_shells": int(np.count_nonzero(visible_mask)),
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"Saved {png_path}")
    print(f"Saved {csv_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
