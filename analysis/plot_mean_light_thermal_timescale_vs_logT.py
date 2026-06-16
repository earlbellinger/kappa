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

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
import numpy as np

from plot_fourier_vs_logT import configure_temperature_axis, load_json
from plot_fourier_vs_massdepth_profiles import parse_profile, profile_photosphere_state
from plot_luminosity_logT_phase_cycle_gif import (
    add_zone_overlays_single_axis,
    instantaneous_zone_structure,
    mean_light_zone_label_positions,
)
from plot_mean_light_work_terms_vs_logT import collect_profile_records, select_last_complete_cycle
from plot_work_logT_phase_cycle_gif import evaluate_periodic_pdv_surface_order, prepare_periodic_pdv_model

THERMAL_TIMESCALE_COLOR = "#3B528B"
FUNDAMENTAL_PERIOD_COLOR = "#1F1F1F"
HARMONIC_PERIOD_COLOR = "#8A8A8A"
HOT_SIDE_TEMPERATURE_LIMIT = 6.0e4
DEFAULT_MAX_HARMONIC_ORDER = 12
SECONDS_PER_DAY = 86400.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the nonlinear thermal timescale versus temperature for the last-cycle snapshot closest to zero integrated power."
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory containing the existing scan outputs")
    parser.add_argument("--prefix", required=True, help="Output filename prefix, e.g. mesa_rsp_combined_14507")
    parser.add_argument(
        "--hot-limit",
        type=float,
        default=HOT_SIDE_TEMPERATURE_LIMIT,
        help="Hot-side temperature limit in K for the displayed window",
    )
    parser.add_argument(
        "--max-harmonic-order",
        type=int,
        default=DEFAULT_MAX_HARMONIC_ORDER,
        help="Maximum harmonic order n to draw for P0/n guide lines",
    )
    return parser.parse_args()


def linear_limits(values: np.ndarray, pad_fraction: float = 0.08) -> tuple[float, float]:
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


def positive_log_limits(values: np.ndarray, lower_pad: float = 1.25, upper_pad: float = 1.25) -> tuple[float, float]:
    positive = np.asarray(values, dtype=float)
    positive = positive[np.isfinite(positive) & (positive > 0.0)]
    if positive.size == 0:
        return 1.0e-6, 1.0
    return float(np.nanmin(positive) / lower_pad), float(np.nanmax(positive) * upper_pad)


def phase_of_record(age_days: float, phase_reference_days: float, period_days: float) -> float:
    return float(np.mod((float(age_days) - float(phase_reference_days)) / float(period_days), 1.0))


def select_final_cycle_records(
    profile_records: list[dict[str, object]],
    final_cycle_summary: dict[str, object],
) -> tuple[list[dict[str, object]], str]:
    cycle_start_age_days = final_cycle_summary.get("cycle_start_age_days")
    cycle_end_age_days = final_cycle_summary.get("cycle_end_age_days")
    if cycle_start_age_days is None or cycle_end_age_days is None:
        return [], "final_cycle_summary_window_unavailable"

    cycle_records = [
        record
        for record in profile_records
        if float(cycle_start_age_days) <= float(record["age_days"]) <= float(cycle_end_age_days)
    ]
    return cycle_records, "explicit final_cycle_summary time window"


def compute_thermal_timescale(
    header: dict[str, float],
    columns: dict[str, np.ndarray],
    photosphere_q_env: float,
) -> dict[str, np.ndarray | float]:
    q_surface_order = np.asarray(columns["q"], dtype=float)
    q_order = np.argsort(q_surface_order)

    q_sorted = q_surface_order[q_order]
    dq_sorted = np.asarray(columns["dq"], dtype=float)[q_order]
    temperature_sorted = np.power(10.0, np.asarray(columns["logT"], dtype=float)[q_order])
    cp_sorted = np.asarray(columns["cp"], dtype=float)[q_order]
    luminosity_sorted_cgs = np.asarray(columns["luminosity"], dtype=float)[q_order] * float(header["lsun"])

    visible_mask = (
        np.isfinite(q_sorted)
        & np.isfinite(dq_sorted)
        & np.isfinite(temperature_sorted)
        & np.isfinite(cp_sorted)
        & np.isfinite(luminosity_sorted_cgs)
        & (q_sorted <= float(photosphere_q_env))
    )

    q_visible = q_sorted[visible_mask]
    dq_visible = dq_sorted[visible_mask]
    temperature_visible = temperature_sorted[visible_mask]
    cp_visible = cp_sorted[visible_mask]
    luminosity_visible_cgs = np.abs(luminosity_sorted_cgs[visible_mask])

    envelope_mass_g = float(header["star_mass"]) * float(header["msun"]) - float(header["M_center"])
    if envelope_mass_g <= 0.0:
        raise RuntimeError("Encountered a non-positive envelope mass while computing the thermal timescale.")

    shell_mass_visible_g = dq_visible * envelope_mass_g
    shell_thermal_content_visible_erg = cp_visible * temperature_visible * shell_mass_visible_g
    overlying_thermal_content_visible_erg = np.cumsum(shell_thermal_content_visible_erg[::-1])[::-1]
    thermal_timescale_visible_sec = overlying_thermal_content_visible_erg / np.clip(luminosity_visible_cgs, 1.0e-99, None)
    thermal_timescale_visible_days = thermal_timescale_visible_sec / SECONDS_PER_DAY

    return {
        "q_visible": q_visible,
        "temperature_visible_K": temperature_visible,
        "shell_mass_visible_g": shell_mass_visible_g,
        "shell_thermal_content_visible_erg": shell_thermal_content_visible_erg,
        "overlying_thermal_content_visible_erg": overlying_thermal_content_visible_erg,
        "local_luminosity_visible_cgs": luminosity_visible_cgs,
        "thermal_timescale_visible_sec": thermal_timescale_visible_sec,
        "thermal_timescale_visible_days": thermal_timescale_visible_days,
        "envelope_mass_g": envelope_mass_g,
    }


def integrated_total_power_for_record(
    record: dict[str, object],
    cycle_phase: np.ndarray,
    cycle_records: list[dict[str, object]],
    periodic_pdv_model: dict[str, object],
    period_days: float,
    phase_reference_days: float,
) -> dict[str, object]:
    header, columns = parse_profile(Path(record["path"]))
    photosphere = profile_photosphere_state(header, columns)

    q_surface_order = np.asarray(columns["q"], dtype=float)
    q_order = np.argsort(q_surface_order)
    q_sorted = q_surface_order[q_order]
    dq_sorted = np.asarray(columns["dq"], dtype=float)[q_order]
    luminosity_sorted = np.asarray(columns["luminosity"], dtype=float)[q_order]

    envelope_mass_g = float(header["star_mass"]) * float(header["msun"]) - float(header["M_center"])
    if envelope_mass_g <= 0.0:
        raise RuntimeError("Encountered a non-positive envelope mass while selecting the zero-power snapshot.")
    dldm_scale = float(header["lsun"]) / envelope_mass_g
    heating_total_sorted = -np.gradient(luminosity_sorted, q_sorted) * dldm_scale

    record_phase = phase_of_record(float(record["age_days"]), phase_reference_days, period_days)
    pdv_surface_order = evaluate_periodic_pdv_surface_order(
        periodic_pdv_model,
        np.asarray([record_phase], dtype=float),
        period_days,
    )[0]
    pdv_sorted = np.asarray(pdv_surface_order, dtype=float)[q_order]

    visible_mask = (
        np.isfinite(q_sorted)
        & np.isfinite(dq_sorted)
        & np.isfinite(heating_total_sorted)
        & np.isfinite(pdv_sorted)
        & (q_sorted <= float(photosphere["q_env"]))
    )
    integrated_heating_power = float(np.nansum(heating_total_sorted[visible_mask] * dq_sorted[visible_mask]))
    integrated_pdv_power = float(np.nansum(pdv_sorted[visible_mask] * dq_sorted[visible_mask]))
    integrated_total_power = float(np.nansum((heating_total_sorted + pdv_sorted)[visible_mask] * dq_sorted[visible_mask]))

    return {
        "record": record,
        "header": header,
        "columns": columns,
        "photosphere": photosphere,
        "phase": record_phase,
        "integrated_heating_power_erg_g_s": integrated_heating_power,
        "integrated_pdv_power_erg_g_s": integrated_pdv_power,
        "integrated_total_power_erg_g_s": integrated_total_power,
    }


def select_zero_power_record(
    cycle_records: list[dict[str, object]],
    cycle_phase: np.ndarray,
    period_days: float,
    phase_reference_days: float,
) -> dict[str, object]:
    profile_cache: dict[int, dict[str, object]] = {}
    periodic_pdv_model = prepare_periodic_pdv_model(
        cycle_records,
        cycle_phase,
        profile_cache,
    )
    evaluated_records = [
        integrated_total_power_for_record(
            record,
            cycle_phase=cycle_phase,
            cycle_records=cycle_records,
            periodic_pdv_model=periodic_pdv_model,
            period_days=period_days,
            phase_reference_days=phase_reference_days,
        )
        for record in cycle_records
    ]
    selected_index = int(
        np.argmin(
            np.abs(
                np.asarray(
                    [entry["integrated_total_power_erg_g_s"] for entry in evaluated_records],
                    dtype=float,
                )
            )
        )
    )
    selected = dict(evaluated_records[selected_index])
    selected["selection_source"] = (
        "last-cycle nonlinear profile minimizing the absolute envelope-integrated phase-local total power "
        "( -dL/dm + P d(1/rho)/dt )"
    )
    return selected


def write_csv(
    path: Path,
    q_visible: np.ndarray,
    temperature_visible_K: np.ndarray,
    thermal_timescale_visible_sec: np.ndarray,
    overlying_thermal_content_visible_erg: np.ndarray,
    local_luminosity_visible_cgs: np.ndarray,
    period_days: float,
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "q",
                "temperature_K",
                "log10_T",
                "tau_th_sec",
                "tau_th_days",
                "tau_th_over_P0",
                "overlying_thermal_content_erg",
                "local_luminosity_cgs",
                "local_luminosity_lsun",
            ]
        )
        for row in zip(
            q_visible,
            temperature_visible_K,
            np.log10(np.clip(temperature_visible_K, 1.0e-99, None)),
            thermal_timescale_visible_sec,
            thermal_timescale_visible_sec / SECONDS_PER_DAY,
            thermal_timescale_visible_sec / max(float(period_days) * SECONDS_PER_DAY, 1.0e-99),
            overlying_thermal_content_visible_erg,
            local_luminosity_visible_cgs,
            local_luminosity_visible_cgs / 3.828e33,
        ):
            writer.writerow([f"{float(value):.12g}" for value in row])


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir.resolve()
    prefix = args.prefix

    fourier_summary_path = output_dir / f"{prefix}_fourier_depth_scan_summary.json"
    final_cycle_summary_path = output_dir / f"{prefix}_final_cycle_summary.json"
    png_path = output_dir / f"{prefix}_mean_light_thermal_timescale_vs_logT.png"
    csv_path = output_dir / f"{prefix}_mean_light_thermal_timescale_vs_logT.csv"
    summary_path = output_dir / f"{prefix}_mean_light_thermal_timescale_vs_logT_summary.json"

    if not fourier_summary_path.exists():
        raise RuntimeError(f"Missing Fourier summary JSON: {fourier_summary_path}")
    if not final_cycle_summary_path.exists():
        raise RuntimeError(f"Missing final-cycle summary JSON: {final_cycle_summary_path}")

    fourier_summary = load_json(fourier_summary_path)
    final_cycle_summary = load_json(final_cycle_summary_path)
    run_dir = Path(str(fourier_summary["run_dir"])).resolve()

    profile_records = collect_profile_records(run_dir)
    explicit_cycle_records, explicit_cycle_source = select_final_cycle_records(profile_records, final_cycle_summary)
    period_days = float(final_cycle_summary["period_days"])
    phase_reference_days = float(
        final_cycle_summary.get(
            "max_light_age_days",
            explicit_cycle_records[0]["age_days"] if explicit_cycle_records else profile_records[0]["age_days"],
        )
    )
    if explicit_cycle_records:
        cycle_records = explicit_cycle_records
        cycle_source = explicit_cycle_source
    else:
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

    selected_profile = select_zero_power_record(
        cycle_records,
        cycle_phase=cycle_phase,
        period_days=period_days,
        phase_reference_days=phase_reference_days,
    )
    header = selected_profile["header"]
    columns = selected_profile["columns"]
    photosphere = selected_profile["photosphere"]
    photosphere_temperature = float(10.0 ** float(photosphere["logT"]))

    zone_frame = instantaneous_zone_structure(
        header,
        columns,
        float(args.hot_limit),
    )
    zone_spans = zone_frame["display_zone_spans"]
    zone_details = zone_frame["zone_details"]
    convection_profile = zone_frame["convection_profile"]
    label_positions = mean_light_zone_label_positions(zone_spans)
    outermost_zone_temperature = float(zone_frame["outermost_temperature_K"])

    thermal_profile = compute_thermal_timescale(
        header,
        columns,
        photosphere_q_env=float(photosphere["q_env"]),
    )
    q_visible = np.asarray(thermal_profile["q_visible"], dtype=float)
    temperature_visible_K = np.asarray(thermal_profile["temperature_visible_K"], dtype=float)
    thermal_timescale_visible_sec = np.asarray(thermal_profile["thermal_timescale_visible_sec"], dtype=float)
    thermal_timescale_visible_days = np.asarray(thermal_profile["thermal_timescale_visible_days"], dtype=float)
    overlying_thermal_content_visible_erg = np.asarray(
        thermal_profile["overlying_thermal_content_visible_erg"],
        dtype=float,
    )
    local_luminosity_visible_cgs = np.asarray(thermal_profile["local_luminosity_visible_cgs"], dtype=float)

    plot_mask = temperature_visible_K <= float(args.hot_limit)
    temperature_plot = temperature_visible_K[plot_mask]
    thermal_timescale_plot_days = thermal_timescale_visible_days[plot_mask]

    max_harmonic_order = max(int(args.max_harmonic_order), 1)
    harmonic_orders = list(range(2, max_harmonic_order + 1))
    guide_periods_days = np.asarray([period_days / order for order in range(1, max_harmonic_order + 1)], dtype=float)
    y_limits = positive_log_limits(np.concatenate([thermal_timescale_plot_days, guide_periods_days]))

    photosphere_tau_days = float(
        np.interp(
            float(photosphere["q_env"]),
            q_visible,
            thermal_timescale_visible_days,
        )
    )

    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    add_zone_overlays_single_axis(ax, zone_spans, label_positions, convection_profile)
    ax.axvline(photosphere_temperature, color="0.2", linewidth=1.15, linestyle=":", zorder=-3)
    text_transform = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(
        photosphere_temperature * (10.0 ** -0.010),
        0.98,
        "Photosphere",
        rotation=90,
        ha="left",
        va="top",
        fontsize=7.5,
        color="0.2",
        transform=text_transform,
        path_effects=[pe.withStroke(linewidth=2.4, foreground="white")],
    )

    ax.plot(
        temperature_plot,
        thermal_timescale_plot_days,
        color=THERMAL_TIMESCALE_COLOR,
        linewidth=2.0,
        zorder=2,
    )
    ax.plot(
        [photosphere_temperature],
        [photosphere_tau_days],
        marker="o",
        markersize=4.8,
        color=THERMAL_TIMESCALE_COLOR,
        markeredgecolor="white",
        markeredgewidth=0.55,
        linestyle="None",
        zorder=3,
    )

    ax.axhline(period_days, color=FUNDAMENTAL_PERIOD_COLOR, linewidth=1.15, linestyle="--", zorder=1)
    for harmonic_order in harmonic_orders:
        ax.axhline(
            period_days / float(harmonic_order),
            color=HARMONIC_PERIOD_COLOR,
            linewidth=0.95,
            linestyle=":",
            zorder=1,
        )

    ax.set_yscale("log")
    ax.set_ylim(*y_limits)
    configure_temperature_axis([ax], temperature_plot)
    ax.set_xlim(float(args.hot_limit), outermost_zone_temperature)
    ax.set_xlabel("T [K]")
    ax.set_ylabel("Thermal Timescale [days]")
    ax.grid(False)

    legend = ax.legend(
        handles=[
            Line2D([0], [0], color=THERMAL_TIMESCALE_COLOR, linewidth=2.0, label=r"$\tau_{\rm th}$"),
            Line2D([0], [0], color=FUNDAMENTAL_PERIOD_COLOR, linewidth=1.15, linestyle="--", label=r"$P_0$"),
            Line2D(
                [0],
                [0],
                color=HARMONIC_PERIOD_COLOR,
                linewidth=0.95,
                linestyle=":",
                label=rf"$P_0 / n,\ 2 \leq n \leq {max_harmonic_order}$",
            ),
        ],
        loc="lower left",
        frameon=False,
        fontsize=8,
    )
    legend.set_zorder(5)

    fig.savefig(png_path, dpi=220)
    plt.close(fig)

    write_csv(
        csv_path,
        q_visible=q_visible,
        temperature_visible_K=temperature_visible_K,
        thermal_timescale_visible_sec=thermal_timescale_visible_sec,
        overlying_thermal_content_visible_erg=overlying_thermal_content_visible_erg,
        local_luminosity_visible_cgs=local_luminosity_visible_cgs,
        period_days=period_days,
    )

    summary = {
        "prefix": prefix,
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "png_path": str(png_path),
        "csv_path": str(csv_path),
        "period_days_used": period_days,
        "cycle_source": cycle_source,
        "cycle_profile_count": len(cycle_records),
        "phase_reference_days_used": float(phase_reference_days),
        "selected_profile_path": str(selected_profile["record"]["path"]),
        "selected_profile_age_days": float(selected_profile["record"]["age_days"]),
        "selected_profile_phase": float(selected_profile["phase"]),
        "selected_profile_photosphere_l_lsun": float(selected_profile["record"]["photosphere_l_lsun"]),
        "selected_profile_photosphere_q_env": float(photosphere["q_env"]),
        "selected_profile_photosphere_temperature_K": photosphere_temperature,
        "selected_integrated_heating_power_erg_g_s": float(selected_profile["integrated_heating_power_erg_g_s"]),
        "selected_integrated_pdv_power_erg_g_s": float(selected_profile["integrated_pdv_power_erg_g_s"]),
        "selected_integrated_total_power_erg_g_s": float(selected_profile["integrated_total_power_erg_g_s"]),
        "selected_profile_source": str(selected_profile["selection_source"]),
        "thermal_timescale_definition": {
            "description": "cp-based cumulative thermal reservoir above each depth divided by the local luminosity magnitude",
            "formula": "tau_th(q) = integral_q^q_ph [cp(T) * T * dm] / |L(q)|",
            "shell_mass_definition": "dm = dq * (M_star - M_center)",
            "cp_based_approximation": True,
        },
        "selection_definition": {
            "description": "The plotted snapshot is the last-cycle nonlinear hydro profile with envelope-integrated phase-local total power closest to zero.",
            "formula": "integrated_total_power = integral[(-dL/dm + P d(1/rho)/dt) d(m/M_env)]",
            "heating_sign_convention": "positive values heat the gas",
            "pdv_sign_convention": "positive values correspond to expansion work done by the gas",
        },
        "max_harmonic_order_drawn": max_harmonic_order,
        "plot_temperature_window_K": [float(args.hot_limit), outermost_zone_temperature],
        "thermal_timescale_days_window": [float(y_limits[0]), float(y_limits[1])],
        "photosphere_tau_th_days": photosphere_tau_days,
        "envelope_mass_g": float(thermal_profile["envelope_mass_g"]),
        "zone_annotation_source": "instantaneous_zone_structure display spans from the selected nonlinear hydro profile, matching the phase-cycle animation logic",
        "zone_detection_details": zone_details,
        "zone_spans_temperature_K": {
            name: [[float(x0), float(x1)] for x0, x1 in spans]
            for name, spans in zone_spans.items()
            if name != "Convection"
        },
        "convection_spans_temperature_K": [
            [float(x0), float(x1)] for x0, x1 in zone_spans.get("Convection", [])
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
