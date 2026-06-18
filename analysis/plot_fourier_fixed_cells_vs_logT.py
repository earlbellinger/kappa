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
    COMPLEX_TRANSFER_HARMONIC_COLORS,
    add_zone_overlays,
    configure_temperature_axis,
    load_mean_light_profile,
    mean_light_zone_structure,
    relative_phase,
)
from plot_fourier_vs_massdepth_profiles import (
    analyze_signal_stack,
    determine_fit_harmonics,
    load_period_and_phase_reference,
    load_profile_cycle,
    profile_photosphere_state,
)

TWOPI = 2.0 * math.pi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot fixed-Lagrangian-cell Fourier amplitude and phase differences "
            "relative to the photosphere against the mean-light shell temperature."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory containing LOGS/profile*.data")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for outputs")
    parser.add_argument("--prefix", required=True, help="Output filename prefix")
    parser.add_argument("--fit-harmonics", type=int, default=None, help="Fourier order, default 14")
    parser.add_argument("--period-days", type=float, default=None, help="Optional period override")
    return parser.parse_args()


def interpolate_unwrapped_phase(q: np.ndarray, phase: np.ndarray, q_reference: float) -> tuple[np.ndarray, float]:
    unwrapped = np.unwrap(np.asarray(phase, dtype=float))
    reference = float(np.interp(float(q_reference), np.asarray(q, dtype=float), unwrapped))
    return relative_phase(unwrapped, reference), reference


def finite_fractional_padding(values: np.ndarray, fraction: float = 0.08, minimum_half_range: float = 0.02) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return -minimum_half_range, minimum_half_range
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    half_range = max(0.5 * (hi - lo), minimum_half_range)
    center = 0.5 * (hi + lo)
    return center - half_range * (1.0 + fraction), center + half_range * (1.0 + fraction)


def phase_ticks_for_limits(ymin: float, ymax: float) -> tuple[list[float], list[str]]:
    candidates = [
        (-math.pi, r"$-\pi$"),
        (-math.pi / 2.0, r"$-\pi/2$"),
        (-math.pi / 4.0, r"$-\pi/4$"),
        (0.0, r"$0$"),
        (math.pi / 4.0, r"$\pi/4$"),
        (math.pi / 2.0, r"$\pi/2$"),
        (math.pi, r"$\pi$"),
    ]
    ticks = [(value, label) for value, label in candidates if ymin <= value <= ymax]
    if len(ticks) >= 2:
        return [value for value, _label in ticks], [label for _value, label in ticks]
    return [0.0], [r"$0$"]


def write_csv(
    path: Path,
    temperature: np.ndarray,
    q_env: np.ndarray,
    reference: dict[str, float],
    parameters: dict[str, np.ndarray],
    delta_amplitudes: dict[str, np.ndarray],
    delta_phases: dict[str, np.ndarray],
) -> None:
    fields = [
        "temperature_K",
        "log10_T",
        "q_env",
        "A1",
        "A2",
        "A3",
        "A1_minus_photosphere",
        "A2_minus_photosphere",
        "A3_minus_photosphere",
        "phi1_rad",
        "phi2_rad",
        "phi3_rad",
        "delta_phi1_rad",
        "delta_phi2_rad",
        "delta_phi3_rad",
        "A1_photosphere",
        "A2_photosphere",
        "A3_photosphere",
        "phi1_photosphere_rad",
        "phi2_photosphere_rad",
        "phi3_photosphere_rad",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for i in range(temperature.size):
            row = {
                "temperature_K": temperature[i],
                "log10_T": math.log10(float(temperature[i])),
                "q_env": q_env[i],
                "A1": parameters["A1"][i],
                "A2": parameters["A2"][i],
                "A3": parameters["A3"][i],
                "A1_minus_photosphere": delta_amplitudes["A1"][i],
                "A2_minus_photosphere": delta_amplitudes["A2"][i],
                "A3_minus_photosphere": delta_amplitudes["A3"][i],
                "phi1_rad": parameters["phi1"][i],
                "phi2_rad": parameters["phi2"][i],
                "phi3_rad": parameters["phi3"][i],
                "delta_phi1_rad": delta_phases["phi1"][i],
                "delta_phi2_rad": delta_phases["phi2"][i],
                "delta_phi3_rad": delta_phases["phi3"][i],
                "A1_photosphere": reference["A1"],
                "A2_photosphere": reference["A2"],
                "A3_photosphere": reference["A3"],
                "phi1_photosphere_rad": reference["phi1"],
                "phi2_photosphere_rad": reference["phi2"],
                "phi3_photosphere_rad": reference["phi3"],
            }
            writer.writerow({key: f"{float(value):.12g}" for key, value in row.items()})


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix

    png_path = output_dir / f"{prefix}_fourier_fixed_cells_vs_logT.png"
    csv_path = output_dir / f"{prefix}_fourier_fixed_cells_vs_logT.csv"
    summary_path = output_dir / f"{prefix}_fourier_fixed_cells_vs_logT_summary.json"
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

    mean_light_profile = load_mean_light_profile(run_dir, final_cycle_summary_path)
    mean_light_header = mean_light_profile["header"]
    mean_light_columns = mean_light_profile["columns"]
    photosphere = profile_photosphere_state(mean_light_header, mean_light_columns)
    photosphere_q = float(photosphere["q_env"])
    photosphere_temperature = float(10.0 ** float(photosphere["logT"]))

    q_profile_sorted, temperature_profile_sorted, zone_spans, zone_details, convection_profile = mean_light_zone_structure(
        mean_light_header,
        mean_light_columns,
    )
    q_env_all = np.asarray(cycle["q_env_sorted"], dtype=float)
    visible_mask = np.isfinite(q_env_all) & (q_env_all <= photosphere_q)
    if not np.any(visible_mask):
        raise RuntimeError("No fixed Lagrangian cells lie beneath the mean-light photosphere.")

    temperature_all = np.interp(q_env_all, q_profile_sorted, temperature_profile_sorted)
    finite_temperature = np.isfinite(temperature_all) & (temperature_all > 0.0)
    visible_mask &= finite_temperature
    q_env = q_env_all[visible_mask]
    temperature = temperature_all[visible_mask]
    parameters = {name: values[visible_mask] for name, values in parameters_all.items()}

    reference: dict[str, float] = {}
    delta_amplitudes: dict[str, np.ndarray] = {}
    delta_phases: dict[str, np.ndarray] = {}
    for key in ("A1", "A2", "A3"):
        reference[key] = float(np.interp(photosphere_q, q_env, parameters[key]))
        delta_amplitudes[key] = parameters[key] - reference[key]
    for key in ("phi1", "phi2", "phi3"):
        delta, phase_reference = interpolate_unwrapped_phase(q_env, parameters[key], photosphere_q)
        reference[key] = phase_reference
        delta_phases[key] = delta

    order = np.argsort(temperature)
    temperature_plot = temperature[order]
    q_plot = q_env[order]
    parameters_plot = {name: values[order] for name, values in parameters.items()}
    delta_amplitudes_plot = {name: values[order] for name, values in delta_amplitudes.items()}
    delta_phases_plot = {name: values[order] for name, values in delta_phases.items()}

    write_csv(
        csv_path,
        temperature_plot,
        q_plot,
        reference,
        parameters_plot,
        delta_amplitudes_plot,
        delta_phases_plot,
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
    fig, (ax_amp, ax_phase) = plt.subplots(2, 1, figsize=(9.4, 6.5), sharex=True, constrained_layout=True)
    axes = [ax_amp, ax_phase]
    add_zone_overlays(axes, zone_spans, photosphere_temperature, convection_profile)

    for harmonic, key in ((1, "A1"), (2, "A2"), (3, "A3")):
        ax_amp.plot(
            temperature_plot,
            delta_amplitudes_plot[key],
            color=COMPLEX_TRANSFER_HARMONIC_COLORS[harmonic],
            linewidth=1.55,
            label=rf"$\Delta A_{harmonic}$",
        )
    ax_amp.axhline(0.0, color="k", linewidth=1.0, linestyle="--", zorder=-2)
    amp_ylim = finite_fractional_padding(
        np.concatenate([delta_amplitudes_plot["A1"], delta_amplitudes_plot["A2"], delta_amplitudes_plot["A3"]]),
        minimum_half_range=0.01,
    )
    ax_amp.set_ylim(*amp_ylim)
    ax_amp.set_ylabel("Delta Fractional Amplitude")
    ax_amp.grid(False)
    ax_amp.legend(loc="upper left", ncol=3, frameon=True, framealpha=0.9)

    for harmonic, key in ((1, "phi1"), (2, "phi2"), (3, "phi3")):
        ax_phase.plot(
            temperature_plot,
            delta_phases_plot[key],
            color=COMPLEX_TRANSFER_HARMONIC_COLORS[harmonic],
            linewidth=1.5,
            label=rf"$\Delta\phi_{harmonic}$",
        )
    ax_phase.axhline(0.0, color="k", linewidth=1.0, linestyle="--", zorder=-2)
    ax_phase.axhline(math.pi / 4.0, color="0.35", linewidth=1.0, linestyle="--", zorder=-2)
    phase_values = np.concatenate([delta_phases_plot["phi1"], delta_phases_plot["phi2"], delta_phases_plot["phi3"]])
    phase_ylim = finite_fractional_padding(
        np.concatenate([phase_values, np.asarray([-math.pi / 4.0, 0.0, math.pi / 4.0, math.pi / 2.0])]),
        minimum_half_range=math.pi / 4.0,
    )
    ax_phase.set_ylim(*phase_ylim)
    ticks, labels = phase_ticks_for_limits(*phase_ylim)
    ax_phase.set_yticks(ticks, labels)
    ax_phase.set_ylabel("Phase Lag [rad]")
    ax_phase.set_xlabel(r"$T\ [{\rm K}]$")
    ax_phase.grid(False)
    ax_phase.legend(loc="upper left", ncol=3, frameon=True, framealpha=0.9)

    configure_temperature_axis(axes, temperature_plot)
    fig.savefig(png_path, dpi=200)
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
        "num_profiles": int(np.asarray(cycle["absolute_time_days"]).size),
        "num_fixed_cells_plotted": int(temperature_plot.size),
        "temperature_coordinate_source": "mean-light nonlinear RSP profile T(q), interpolated onto fixed Lagrangian q cells",
        "mean_light_profile_path": str(mean_light_profile["path"]),
        "mean_light_selection_source": str(mean_light_profile["selection_source"]),
        "photosphere_q_env": photosphere_q,
        "photosphere_temperature_K": photosphere_temperature,
        "photosphere_reference_method": "amplitudes and unwrapped phases linearly interpolated in fixed q to the mean-light tau=2/3 photosphere",
        "amplitude_convention": "fractional shell luminosity amplitude from L(q,t)/<L(q)> - 1",
        "amplitude_plot": "Delta A_k = A_k(q) - A_k(photosphere)",
        "phase_plot": "Delta phi_k = phi_k(q) - phi_k(photosphere), unwrapped continuously in q",
        "photosphere_fourier_reference": {key: float(value) for key, value in reference.items()},
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
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved {png_path}")
    print(f"Saved {csv_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
