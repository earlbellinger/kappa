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

import matplotlib.pyplot as plt
import numpy as np

from plot_fourier_vs_logT import (
    add_zone_overlays,
    configure_temperature_axis,
    load_json,
    mean_light_zone_structure,
    read_profile_header,
)
from plot_fourier_vs_massdepth_profiles import parse_profile, profile_photosphere_state

NET_WORK_COLOR = "#3B528B"
PDV_COLOR = "#1F1F1F"
RADIATIVE_COLOR = "#D55E00"
CONVECTIVE_COLOR = "#009E73"
TURBULENT_COLOR = "#9467BD"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot nonlinear hydro work terms against the mean-light temperature coordinate."
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory containing the existing scan outputs")
    parser.add_argument("--prefix", required=True, help="Output filename prefix, e.g. mesa_rsp_combined_14507")
    return parser.parse_args()


def collect_profile_records(run_dir: Path) -> list[dict[str, object]]:
    logs_dir = run_dir / "LOGS"
    profile_paths = list(logs_dir.glob("profile*.data"))
    if not profile_paths:
        raise RuntimeError(f"No profile files found in {logs_dir}")

    records: list[dict[str, object]] = []
    for path in profile_paths:
        header = read_profile_header(path)
        records.append(
            {
                "path": path,
                "age_days": float(header["star_age"]) * 365.25,
                "photosphere_l_lsun": float(header["photosphere_L"]),
            }
        )
    records.sort(key=lambda record: float(record["age_days"]))
    return records


def select_last_complete_cycle(
    records: list[dict[str, object]],
    final_cycle_summary: dict[str, object],
) -> tuple[list[dict[str, object]], float, float, str]:
    ages = np.asarray([float(record["age_days"]) for record in records], dtype=float)
    period_days = float(final_cycle_summary["period_days"])
    cycle_start_age_days = final_cycle_summary.get("cycle_start_age_days")
    cycle_end_age_days = final_cycle_summary.get("cycle_end_age_days")
    phase_reference_days = final_cycle_summary.get(
        "phase_reference_age_days",
        final_cycle_summary.get("max_light_age_days"),
    )

    if cycle_start_age_days is not None and cycle_end_age_days is not None:
        start_age = float(cycle_start_age_days)
        end_age = float(cycle_end_age_days)
        if end_age > start_age:
            age_tolerance = max(1.0e-9, 1.0e-6 * period_days)
            selected = [
                record
                for record in records
                if start_age - age_tolerance <= float(record["age_days"]) < end_age - age_tolerance
            ]
            if len(selected) >= 2:
                if phase_reference_days is None:
                    phase_reference_days = start_age
                return (
                    selected,
                    period_days,
                    float(phase_reference_days),
                    "final-cycle summary age window",
                )

    if phase_reference_days is not None:
        phase = np.mod((ages - float(phase_reference_days)) / period_days, 1.0)
        wraps = np.where(np.diff(phase) < -0.5)[0]
        if wraps.size >= 2:
            start = int(wraps[-2] + 1)
            end = int(wraps[-1] + 1)
            return records[start:end], period_days, float(phase_reference_days), "last complete phase-wrapped cycle"

    last_age = float(ages[-1])
    cycle_start = last_age - period_days
    cycle_records = [record for record in records if float(record["age_days"]) >= cycle_start]
    if len(cycle_records) < 2:
        raise RuntimeError("Could not identify a complete nonlinear hydro cycle in the profile sequence.")
    return cycle_records, period_days, float(ages[0]), "last-period fallback window"


def select_mean_light_record(cycle_records: list[dict[str, object]]) -> dict[str, object]:
    cycle_luminosity = np.asarray([float(record["photosphere_l_lsun"]) for record in cycle_records], dtype=float)
    mean_light_luminosity = float(np.mean(cycle_luminosity))
    selected_index = int(np.argmin(np.abs(cycle_luminosity - mean_light_luminosity)))
    selected = dict(cycle_records[selected_index])
    selected["mean_cycle_photosphere_l_lsun"] = mean_light_luminosity
    return selected


def read_cycle_matrices(cycle_records: list[dict[str, object]]) -> dict[str, np.ndarray | float]:
    absolute_time_sec: list[float] = []
    pressure: list[np.ndarray] = []
    density: list[np.ndarray] = []
    luminosity_total: list[np.ndarray] = []
    luminosity_radiative: list[np.ndarray] = []
    luminosity_convective: list[np.ndarray] = []
    luminosity_turbulent: list[np.ndarray] = []

    q_grid: np.ndarray | None = None
    header0: dict[str, float] | None = None

    for record in cycle_records:
        header, columns = parse_profile(Path(record["path"]))
        q_current = np.asarray(columns["q"], dtype=float)
        if q_grid is None:
            q_grid = q_current.copy()
            header0 = header
        elif not np.array_equal(q_current, q_grid):
            raise RuntimeError("The q grid changed across the nonlinear hydro cycle.")

        absolute_time_sec.append(float(record["age_days"]) * 86400.0)
        pressure.append(np.asarray(columns["pressure"], dtype=float))
        density.append(np.power(10.0, np.asarray(columns["logRho"], dtype=float)))
        luminosity_total.append(np.asarray(columns["luminosity"], dtype=float))

        lsun_cgs = float(header["lsun"])
        luminosity_radiative.append(np.asarray(columns["rsp_Lr"], dtype=float) / lsun_cgs)
        luminosity_convective.append(np.asarray(columns["rsp_Lc"], dtype=float) / lsun_cgs)
        luminosity_turbulent.append(np.asarray(columns["rsp_Lt"], dtype=float) / lsun_cgs)

    if q_grid is None or header0 is None:
        raise RuntimeError("No nonlinear hydro profiles were loaded for the selected cycle.")

    q_order = np.argsort(q_grid)
    q_sorted = q_grid[q_order]

    star_mass_g = float(header0["star_mass"]) * float(header0["msun"])
    envelope_mass_g = star_mass_g - float(header0["M_center"])

    return {
        "q_sorted": q_sorted,
        "absolute_time_sec": np.asarray(absolute_time_sec, dtype=float),
        "pressure_sorted": np.asarray(pressure, dtype=float)[:, q_order],
        "density_sorted": np.asarray(density, dtype=float)[:, q_order],
        "luminosity_total_sorted": np.asarray(luminosity_total, dtype=float)[:, q_order],
        "luminosity_radiative_sorted": np.asarray(luminosity_radiative, dtype=float)[:, q_order],
        "luminosity_convective_sorted": np.asarray(luminosity_convective, dtype=float)[:, q_order],
        "luminosity_turbulent_sorted": np.asarray(luminosity_turbulent, dtype=float)[:, q_order],
        "envelope_mass_g": envelope_mass_g,
        "lsun_cgs": float(header0["lsun"]),
    }


def close_cycle(values: np.ndarray) -> np.ndarray:
    return np.vstack([values, values[0]])


def integrate_pdv_per_mass(
    pressure_sorted: np.ndarray,
    density_sorted: np.ndarray,
) -> np.ndarray:
    pressure_closed = close_cycle(pressure_sorted)
    density_closed = close_cycle(density_sorted)
    specific_volume = 1.0 / density_closed
    return np.sum(
        -0.5 * (pressure_closed[:-1] + pressure_closed[1:]) * (specific_volume[1:] - specific_volume[:-1]),
        axis=0,
    )


def integrate_luminosity_gradient_work(
    luminosity_sorted_lsun: np.ndarray,
    q_sorted: np.ndarray,
    delta_time_sec: np.ndarray,
    lsun_cgs: float,
    envelope_mass_g: float,
) -> np.ndarray:
    luminosity_closed = close_cycle(luminosity_sorted_lsun)
    dldm = np.gradient(luminosity_closed, q_sorted, axis=1) * (lsun_cgs / envelope_mass_g)
    return np.sum(0.5 * (dldm[:-1] + dldm[1:]) * delta_time_sec[:, None], axis=0)


def write_csv(
    path: Path,
    temperature_visible: np.ndarray,
    q_visible: np.ndarray,
    work_total_visible: np.ndarray,
    work_pdv_visible: np.ndarray,
    work_radiative_visible: np.ndarray,
    work_convective_visible: np.ndarray,
    work_turbulent_visible: np.ndarray,
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "temperature_K",
                "log10_T",
                "q",
                "dW_dm_cycle_erg_per_g",
                "PdV_cycle_erg_per_g",
                "radiative_work_cycle_erg_per_g",
                "convective_work_cycle_erg_per_g",
                "turbulent_work_cycle_erg_per_g",
            ]
        )
        for row in zip(
            temperature_visible,
            np.log10(np.clip(temperature_visible, 1.0e-99, None)),
            q_visible,
            work_total_visible,
            work_pdv_visible,
            work_radiative_visible,
            work_convective_visible,
            work_turbulent_visible,
        ):
            writer.writerow([f"{float(value):.12g}" for value in row])


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    prefix = args.prefix

    fourier_summary_path = output_dir / f"{prefix}_fourier_depth_scan_summary.json"
    final_cycle_summary_path = output_dir / f"{prefix}_final_cycle_summary.json"
    png_path = output_dir / f"{prefix}_mean_light_work_terms_vs_logT.png"
    csv_path = output_dir / f"{prefix}_mean_light_work_terms_vs_logT.csv"
    summary_path = output_dir / f"{prefix}_mean_light_work_terms_vs_logT_summary.json"

    if not fourier_summary_path.exists():
        raise RuntimeError(f"Missing Fourier summary JSON: {fourier_summary_path}")
    if not final_cycle_summary_path.exists():
        raise RuntimeError(f"Missing final-cycle summary JSON: {final_cycle_summary_path}")

    fourier_summary = load_json(fourier_summary_path)
    final_cycle_summary = load_json(final_cycle_summary_path)
    run_dir = Path(str(fourier_summary["run_dir"])).resolve()

    profile_records = collect_profile_records(run_dir)
    cycle_records, period_days, phase_reference_days, cycle_source = select_last_complete_cycle(
        profile_records,
        final_cycle_summary,
    )
    mean_light_record = select_mean_light_record(cycle_records)

    mean_light_header, mean_light_columns = parse_profile(Path(mean_light_record["path"]))
    photosphere = profile_photosphere_state(mean_light_header, mean_light_columns)
    photosphere_temperature = float(10.0 ** float(photosphere["logT"]))

    q_sorted, temperature_sorted, zone_spans, zone_details, convection_profile = mean_light_zone_structure(
        mean_light_header,
        mean_light_columns,
    )
    outermost_zone_temperature = float(np.nanmin(np.asarray(temperature_sorted, dtype=float)))
    hot_side_temperature_limit = 1.0e5
    visible_mask = q_sorted <= float(photosphere["q_env"])
    q_visible = q_sorted[visible_mask]
    temperature_visible = temperature_sorted[visible_mask]

    cycle_data = read_cycle_matrices(cycle_records)
    q_cycle = np.asarray(cycle_data["q_sorted"], dtype=float)
    if not np.allclose(q_cycle, q_sorted, rtol=0.0, atol=1.0e-12):
        raise RuntimeError("The selected cycle q grid does not match the mean-light profile q grid.")

    absolute_time_sec = np.asarray(cycle_data["absolute_time_sec"], dtype=float)
    closed_time_sec = np.concatenate([absolute_time_sec, [absolute_time_sec[0] + period_days * 86400.0]])
    delta_time_sec = np.diff(closed_time_sec)

    work_pdv = integrate_pdv_per_mass(
        pressure_sorted=np.asarray(cycle_data["pressure_sorted"], dtype=float),
        density_sorted=np.asarray(cycle_data["density_sorted"], dtype=float),
    )
    work_total = integrate_luminosity_gradient_work(
        luminosity_sorted_lsun=np.asarray(cycle_data["luminosity_total_sorted"], dtype=float),
        q_sorted=q_sorted,
        delta_time_sec=delta_time_sec,
        lsun_cgs=float(cycle_data["lsun_cgs"]),
        envelope_mass_g=float(cycle_data["envelope_mass_g"]),
    )
    work_radiative = integrate_luminosity_gradient_work(
        luminosity_sorted_lsun=np.asarray(cycle_data["luminosity_radiative_sorted"], dtype=float),
        q_sorted=q_sorted,
        delta_time_sec=delta_time_sec,
        lsun_cgs=float(cycle_data["lsun_cgs"]),
        envelope_mass_g=float(cycle_data["envelope_mass_g"]),
    )
    work_convective = integrate_luminosity_gradient_work(
        luminosity_sorted_lsun=np.asarray(cycle_data["luminosity_convective_sorted"], dtype=float),
        q_sorted=q_sorted,
        delta_time_sec=delta_time_sec,
        lsun_cgs=float(cycle_data["lsun_cgs"]),
        envelope_mass_g=float(cycle_data["envelope_mass_g"]),
    )
    work_turbulent = integrate_luminosity_gradient_work(
        luminosity_sorted_lsun=np.asarray(cycle_data["luminosity_turbulent_sorted"], dtype=float),
        q_sorted=q_sorted,
        delta_time_sec=delta_time_sec,
        lsun_cgs=float(cycle_data["lsun_cgs"]),
        envelope_mass_g=float(cycle_data["envelope_mass_g"]),
    )

    work_total_visible = work_total[visible_mask]
    work_pdv_visible = work_pdv[visible_mask]
    work_radiative_visible = work_radiative[visible_mask]
    work_convective_visible = work_convective[visible_mask]
    work_turbulent_visible = work_turbulent[visible_mask]

    # Display the conventional pulsation-work sign: positive values correspond to driving.
    displayed_work_total = -work_total_visible
    displayed_work_pdv = -work_pdv_visible
    displayed_work_radiative = -work_radiative_visible
    displayed_work_convective = -work_convective_visible
    displayed_work_turbulent = -work_turbulent_visible

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

    fig, axes = plt.subplots(3, 1, figsize=(9.4, 8.6), sharex=True, constrained_layout=True)
    add_zone_overlays(list(axes), zone_spans, photosphere_temperature, convection_profile)

    axes[0].plot(temperature_visible, displayed_work_total, color=NET_WORK_COLOR, linewidth=1.7, label=r"$dW/dm$")
    axes[1].plot(temperature_visible, displayed_work_pdv, color=PDV_COLOR, linewidth=1.55, label=r"$\oint P\,dV$")
    axes[2].plot(
        temperature_visible,
        displayed_work_radiative,
        color=RADIATIVE_COLOR,
        linewidth=1.6,
        label="Radiative",
    )
    axes[2].plot(
        temperature_visible,
        displayed_work_convective,
        color=CONVECTIVE_COLOR,
        linewidth=1.6,
        label="Convective",
    )
    axes[2].plot(
        temperature_visible,
        displayed_work_turbulent,
        color=TURBULENT_COLOR,
        linewidth=1.25,
        linestyle="--",
        alpha=0.9,
        label="Turbulent",
    )

    axes[0].set_ylabel(r"$dW/dm$ [erg g$^{-1}$]")
    axes[1].set_ylabel(r"$\oint P\,dV$ [erg g$^{-1}$]")
    axes[2].set_ylabel("Work Contribution [erg g$^{-1}$]")
    axes[2].set_xlabel(r"$T\ [{\rm K}]$")

    for ax in axes:
        ax.axhline(0.0, color="0.2", linewidth=1.0, linestyle="--", zorder=-2)
        ax.grid(False)

    axes[0].legend(loc="upper left", frameon=True, framealpha=0.9)
    axes[1].legend(loc="upper left", frameon=True, framealpha=0.9)
    axes[2].legend(loc="upper left", frameon=True, framealpha=0.9, ncol=3)

    configure_temperature_axis(list(axes), temperature_visible)
    for ax in axes:
        ax.set_xlim(hot_side_temperature_limit, outermost_zone_temperature)
    fig.savefig(png_path, dpi=220)
    plt.close(fig)

    write_csv(
        path=csv_path,
        temperature_visible=temperature_visible,
        q_visible=q_visible,
        work_total_visible=displayed_work_total,
        work_pdv_visible=displayed_work_pdv,
        work_radiative_visible=displayed_work_radiative,
        work_convective_visible=displayed_work_convective,
        work_turbulent_visible=displayed_work_turbulent,
    )

    summary = {
        "prefix": prefix,
        "output_dir": str(output_dir),
        "run_dir": str(run_dir),
        "png_path": str(png_path),
        "csv_path": str(csv_path),
        "source_fourier_summary_json": str(fourier_summary_path),
        "source_final_cycle_summary_json": str(final_cycle_summary_path),
        "cycle_source": cycle_source,
        "cycle_profile_count": len(cycle_records),
        "cycle_start_age_days": float(cycle_records[0]["age_days"]),
        "cycle_end_age_days": float(cycle_records[-1]["age_days"]),
        "period_days_used": period_days,
        "phase_reference_days_used": phase_reference_days,
        "mean_light_profile_path": str(mean_light_record["path"]),
        "mean_light_profile_age_days": float(mean_light_record["age_days"]),
        "mean_light_profile_photosphere_l_lsun": float(mean_light_record["photosphere_l_lsun"]),
        "mean_cycle_photosphere_l_lsun": float(mean_light_record["mean_cycle_photosphere_l_lsun"]),
        "temperature_coordinate_source": "temperature at the mean-light nonlinear hydro profile on the fixed q mesh",
        "photosphere_temperature_K": photosphere_temperature,
        "outermost_zone_temperature_K": outermost_zone_temperature,
        "plot_temperature_window_K": [hot_side_temperature_limit, outermost_zone_temperature],
        "mean_light_photosphere_q_env": float(photosphere["q_env"]),
        "zone_detection_details": zone_details,
        "zone_spans_temperature_K": {
            name: [[float(x0), float(x1)] for x0, x1 in spans]
            for name, spans in zone_spans.items()
            if name != "Convection"
        },
        "convection_spans_temperature_K": [
            [float(x0), float(x1)] for x0, x1 in zone_spans.get("Convection", [])
        ],
        "definitions": {
            "dW_dm": "cycle-integrated specific work in the conventional positive-driving sign, -integral[dL/dm dt]",
            "PdV": "cycle-integrated specific compressional work in the conventional positive-driving sign, integral[P d(1/rho)]",
            "radiative": "radiative contribution to the conventional work integral, -integral[dLr/dm dt]",
            "convective": "convective contribution to the conventional work integral, -integral[dLc/dm dt]",
            "turbulent": "turbulent-flux contribution to the conventional work integral, -integral[dLt/dm dt]",
        },
        "closure_metrics": {
            "rms_dWdm_minus_PdV_erg_per_g": float(
                np.sqrt(np.nanmean(np.square(displayed_work_total - displayed_work_pdv)))
            ),
            "rms_rad_plus_conv_minus_PdV_erg_per_g": float(
                np.sqrt(
                    np.nanmean(
                        np.square(displayed_work_radiative + displayed_work_convective - displayed_work_pdv)
                    )
                )
            ),
            "rms_rad_plus_conv_plus_turb_minus_PdV_erg_per_g": float(
                np.sqrt(
                    np.nanmean(
                        np.square(
                            displayed_work_radiative
                            + displayed_work_convective
                            + displayed_work_turbulent
                            - displayed_work_pdv
                        )
                    )
                )
            ),
            "mean_abs_turbulent_work_erg_per_g": float(np.nanmean(np.abs(displayed_work_turbulent))),
        },
        "extrema": {
            "dW_dm_peak_logT": float(np.log10(temperature_visible[int(np.nanargmax(np.abs(displayed_work_total)))])),
            "PdV_peak_logT": float(np.log10(temperature_visible[int(np.nanargmax(np.abs(displayed_work_pdv)))])),
            "radiative_peak_logT": float(
                np.log10(temperature_visible[int(np.nanargmax(np.abs(displayed_work_radiative)))])
            ),
            "convective_peak_logT": float(
                np.log10(temperature_visible[int(np.nanargmax(np.abs(displayed_work_convective)))])
            ),
            "turbulent_peak_logT": float(
                np.log10(temperature_visible[int(np.nanargmax(np.abs(displayed_work_turbulent)))])
            ),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"Saved {png_path}")
    print(f"Saved {csv_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
