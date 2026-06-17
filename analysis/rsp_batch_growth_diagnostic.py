from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rsp_batch_convergence import absolute_peak_to_peak, fractional_peak_to_peak, read_json
from rsp_batch_convergence_trends import source_rows


ROOT = Path(__file__).resolve().parent
DEFAULT_WORKSPACE = ROOT / "rsp_batch_runs"
CONVERGENCE_TOLERANCE = 1.0e-3


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure active RSP amplitude-growth trends for a batch model."
    )
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--model", default="model_008")
    parser.add_argument("--windows", type=int, nargs="+", default=[50, 100, 250, 500])
    parser.add_argument("--rolling-window", type=int, default=100)
    parser.add_argument("--plot-tail", type=int, default=400)
    parser.add_argument("--output-prefix", type=Path, default=None)
    return parser.parse_args()


def find_record(workspace: Path, model: str) -> dict[str, object]:
    manifest = read_json(workspace / "inputs" / "manifest.json")
    if not isinstance(manifest, list):
        raise RuntimeError(f"Could not read manifest list from {workspace / 'inputs' / 'manifest.json'}")
    for record in manifest:
        if isinstance(record, dict) and model in {str(record.get("model_id")), str(record.get("run_name"))}:
            return record
    raise KeyError(f"No model matching {model!r}")


def finite_metric(values: list[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def fit_window(rows: list[dict[str, float]], key: str, window: int) -> dict[str, object]:
    if len(rows) < window:
        return {"available": False, "reason": f"only {len(rows)} rows for {window}-cycle window"}
    tail = rows[-window:]
    x = np.asarray([row["period_number"] for row in tail], dtype=float)
    y = np.asarray([row.get(key, np.nan) for row in tail], dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 2:
        return {"available": False, "reason": f"{key} has fewer than two finite values"}
    x = x[mask]
    y = y[mask]
    x_centered = x - float(np.mean(x))
    y_centered = y - float(np.mean(y))
    denominator = float(np.sum(x_centered * x_centered))
    slope = float(np.sum(x_centered * y_centered) / denominator) if denominator > 0.0 else math.nan
    median = float(np.nanmedian(np.abs(y)))
    frac_slope = float(slope / median) if median > 0.0 else None
    peak_to_peak = float(np.nanmax(y) - np.nanmin(y))
    frac_ptp = float(peak_to_peak / median) if median > 0.0 else None
    log_slope = None
    efolding_cycles = None
    if np.all(y > 0.0) and denominator > 0.0:
        log_y = np.log(y)
        log_y_centered = log_y - float(np.mean(log_y))
        log_slope = float(np.sum(x_centered * log_y_centered) / denominator)
        if log_slope > 0.0:
            efolding_cycles = float(1.0 / log_slope)
    return {
        "available": True,
        "window_cycles": int(window),
        "window_start_period": float(x[0]),
        "window_end_period": float(x[-1]),
        "first": float(y[0]),
        "last": float(y[-1]),
        "median": float(np.nanmedian(y)),
        "peak_to_peak": peak_to_peak,
        "fractional_peak_to_peak": frac_ptp,
        "slope_per_cycle": slope,
        "fractional_slope_per_cycle": frac_slope,
        "log_slope_per_cycle": log_slope,
        "efolding_cycles": efolding_cycles,
    }


def rolling_rows(rows: list[dict[str, float]], window: int) -> list[dict[str, object]]:
    if len(rows) < window:
        return []
    output: list[dict[str, object]] = []
    for end in range(window, len(rows) + 1):
        tail = rows[end - window : end]
        gamma = np.asarray([row.get("gamma", np.nan) for row in tail], dtype=float)
        period = np.asarray([row.get("period_days", np.nan) for row in tail], dtype=float)
        delta_r = np.asarray([row.get("delta_r", np.nan) for row in tail], dtype=float)
        output.append(
            {
                "window_start_period": float(tail[0]["period_number"]),
                "window_end_period": float(tail[-1]["period_number"]),
                "gamma_peak_to_peak": absolute_peak_to_peak(gamma),
                "period_fractional_peak_to_peak": fractional_peak_to_peak(period),
                "delta_r_fractional_peak_to_peak": fractional_peak_to_peak(delta_r),
            }
        )
    return output


def safe_array(rows: list[dict[str, float]], key: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([row["period_number"] for row in rows], dtype=float)
    y = np.asarray([row.get(key, np.nan) for row in rows], dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def growth_outlook(
    fits: dict[str, dict[str, dict[str, object]]],
    rolling: list[dict[str, object]],
    rows: list[dict[str, float]],
    *,
    tolerance: float = CONVERGENCE_TOLERANCE,
) -> dict[str, object]:
    preferred_fit = fits.get("100", {}).get("delta_r")
    if not (isinstance(preferred_fit, dict) and preferred_fit.get("available")):
        available = [
            fit.get("delta_r", {})
            for _window, fit in sorted(fits.items(), key=lambda item: int(item[0]))
            if isinstance(fit.get("delta_r"), dict) and fit["delta_r"].get("available")
        ]
        preferred_fit = available[-1] if available else {}
    latest = rolling[-1] if rolling else {}
    delta_metric = latest.get("delta_r_fractional_peak_to_peak") if isinstance(latest, dict) else None
    criterion_factor = None
    if delta_metric is not None and tolerance > 0.0:
        criterion_factor = float(delta_metric) / float(tolerance)
    log_slope = preferred_fit.get("log_slope_per_cycle") if isinstance(preferred_fit, dict) else None
    efolding_cycles = preferred_fit.get("efolding_cycles") if isinstance(preferred_fit, dict) else None
    doubling_cycles = None
    status = "insufficient amplitude-growth fit"
    if log_slope is not None:
        log_slope = float(log_slope)
        if log_slope > 0.0:
            doubling_cycles = float(math.log(2.0) / log_slope)
            status = "amplitude still growing"
        elif log_slope < 0.0:
            status = "amplitude declining"
        else:
            status = "amplitude nearly flat"
    if criterion_factor is not None and criterion_factor <= 1.0:
        status = "DeltaR convergence term is within tolerance"

    velocity_fit = fits.get("100", {}).get("max_vsurf_div_cs")
    if not (isinstance(velocity_fit, dict) and velocity_fit.get("available")):
        available_velocity = [
            fit.get("max_vsurf_div_cs", {})
            for _window, fit in sorted(fits.items(), key=lambda item: int(item[0]))
            if isinstance(fit.get("max_vsurf_div_cs"), dict) and fit["max_vsurf_div_cs"].get("available")
        ]
        velocity_fit = available_velocity[-1] if available_velocity else {}
    latest_velocity = None
    for row in reversed(rows):
        value = row.get("max_vsurf_div_cs")
        if value is not None and np.isfinite(float(value)):
            latest_velocity = float(value)
            break
    velocity_log_slope = velocity_fit.get("log_slope_per_cycle") if isinstance(velocity_fit, dict) else None
    velocity_efolding_cycles = velocity_fit.get("efolding_cycles") if isinstance(velocity_fit, dict) else None
    cycles_to_vsurf_0p8 = None
    cycles_to_vsurf_1p0 = None
    if (
        latest_velocity is not None
        and latest_velocity > 0.0
        and velocity_log_slope is not None
        and float(velocity_log_slope) > 0.0
    ):
        for target, key in ((0.8, "cycles_to_vsurf_0p8"), (1.0, "cycles_to_vsurf_1p0")):
            if latest_velocity < target:
                cycles = float(math.log(target / latest_velocity) / float(velocity_log_slope))
            else:
                cycles = 0.0
            if key == "cycles_to_vsurf_0p8":
                cycles_to_vsurf_0p8 = cycles
            else:
                cycles_to_vsurf_1p0 = cycles

    return {
        "status": status,
        "tolerance": float(tolerance),
        "delta_r_fractional_peak_to_peak": None if delta_metric is None else float(delta_metric),
        "delta_r_criterion_factor": criterion_factor,
        "log_slope_per_cycle": log_slope,
        "efolding_cycles": None if efolding_cycles is None else float(efolding_cycles),
        "doubling_cycles": doubling_cycles,
        "fit_window_cycles": preferred_fit.get("window_cycles") if isinstance(preferred_fit, dict) else None,
        "max_vsurf_div_cs_latest": latest_velocity,
        "max_vsurf_div_cs_log_slope_per_cycle": (
            None if velocity_log_slope is None else float(velocity_log_slope)
        ),
        "max_vsurf_div_cs_efolding_cycles": (
            None if velocity_efolding_cycles is None else float(velocity_efolding_cycles)
        ),
        "max_vsurf_div_cs_fit_window_cycles": (
            velocity_fit.get("window_cycles") if isinstance(velocity_fit, dict) else None
        ),
        "cycles_to_vsurf_div_cs_0p8": cycles_to_vsurf_0p8,
        "cycles_to_vsurf_div_cs_1p0": cycles_to_vsurf_1p0,
    }


def plot_diagnostic(
    path: Path,
    rows: list[dict[str, float]],
    fits: dict[str, dict[str, dict[str, object]]],
    rolling: list[dict[str, object]],
    outlook: dict[str, object],
    *,
    model_id: str,
    plot_tail: int,
) -> None:
    shown_rows = rows[-plot_tail:] if len(rows) > plot_tail else rows
    fig, axes = plt.subplots(4, 1, figsize=(11.5, 11.5), sharex=False, constrained_layout=True)
    fig.suptitle(f"{model_id} active limit-cycle growth diagnostic", fontsize=15)

    x_delta, y_delta = safe_array(shown_rows, "delta_r")
    axes[0].plot(x_delta, y_delta, color="#C1121F", lw=1.8)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Delta R")
    axes[0].set_title("Radius-amplitude growth", loc="left", fontsize=11)
    delta_fit = fits.get("500", {}).get("delta_r") or fits.get("100", {}).get("delta_r") or {}
    if delta_fit.get("available") and delta_fit.get("log_slope_per_cycle") is not None:
        start = float(delta_fit["window_start_period"])
        end = float(delta_fit["window_end_period"])
        slope = float(delta_fit["log_slope_per_cycle"])
        x_fit = np.linspace(start, end, 120)
        y0 = float(delta_fit["first"])
        y_fit = y0 * np.exp(slope * (x_fit - start))
        axes[0].plot(x_fit, y_fit, color="#FFB703", lw=1.4, ls="--")
        efold = delta_fit.get("efolding_cycles")
        fit_text = (
            f"last {int(delta_fit['window_cycles'])} cycles: e-fold {float(efold):.0f} cycles"
            if efold is not None
            else f"last {int(delta_fit['window_cycles'])} cycles: fitted log-slope {slope:.3g}/cycle"
        )
        axes[0].text(
            0.02,
            0.94,
            fit_text,
            transform=axes[0].transAxes,
            va="top",
            ha="left",
            fontsize=10,
            color="0.15",
        )
        doubling_cycles = outlook.get("doubling_cycles")
        if doubling_cycles is not None:
            axes[0].text(
                0.02,
                0.84,
                f"amplitude doubles in {float(doubling_cycles):.0f} cycles if this rate persists",
                transform=axes[0].transAxes,
                va="top",
                ha="left",
                fontsize=10,
                color="0.15",
            )

    x_vsurf, y_vsurf = safe_array(shown_rows, "max_vsurf_div_cs")
    if y_vsurf.size:
        axes[1].plot(x_vsurf, y_vsurf, color="#FB8500", lw=1.7)
    axes[1].axhline(0.8, color="0.25", lw=1.0, ls=":", label=r"$v_\mathrm{surf}/c_s=0.8$")
    axes[1].axhline(1.0, color="0.15", lw=1.0, ls="--", label=r"$v_\mathrm{surf}/c_s=1$")
    axes[1].set_ylabel(r"max $v_\mathrm{surf}/c_s$")
    axes[1].set_title("Surface velocity growth", loc="left", fontsize=11)
    velocity_fit = fits.get("500", {}).get("max_vsurf_div_cs") or fits.get("100", {}).get("max_vsurf_div_cs") or {}
    if velocity_fit.get("available") and velocity_fit.get("log_slope_per_cycle") is not None:
        start = float(velocity_fit["window_start_period"])
        end = float(velocity_fit["window_end_period"])
        slope = float(velocity_fit["log_slope_per_cycle"])
        x_fit = np.linspace(start, end, 120)
        y0 = float(velocity_fit["first"])
        y_fit = y0 * np.exp(slope * (x_fit - start))
        axes[1].plot(x_fit, y_fit, color="#FDF0D5", lw=1.4, ls="--")
    cycles_to_08 = outlook.get("cycles_to_vsurf_div_cs_0p8")
    if cycles_to_08 is not None:
        axes[1].text(
            0.02,
            0.94,
            f"0.8 in {float(cycles_to_08):.0f} cycles if this rate persists",
            transform=axes[1].transAxes,
            va="top",
            ha="left",
            fontsize=10,
            color="0.15",
        )
    axes[1].legend(frameon=False, loc="upper right", fontsize=9)

    x_gamma, y_gamma = safe_array(shown_rows, "gamma")
    gamma_mask = y_gamma > 0.0
    x_gamma = x_gamma[gamma_mask]
    y_gamma = y_gamma[gamma_mask]
    axes[2].plot(x_gamma, y_gamma, color="#669BBC", lw=1.4, label="Gamma")
    axes[2].axhline(0.0, color="0.2", lw=1.0, ls="--")
    axes[2].set_ylabel("Gamma")
    axes[2].set_title("Growth-rate history", loc="left", fontsize=11)

    if rolling:
        x_roll = np.asarray([row["window_end_period"] for row in rolling], dtype=float)
        for key, label, color in (
            ("delta_r_fractional_peak_to_peak", "Delta R frac", "#C1121F"),
            ("gamma_peak_to_peak", "Gamma ptp", "#669BBC"),
            ("period_fractional_peak_to_peak", "P frac", "#FFB703"),
        ):
            y = np.asarray(
                [np.nan if row.get(key) is None else float(row[key]) for row in rolling],
                dtype=float,
            )
            mask = np.isfinite(x_roll) & np.isfinite(y) & (y > 0.0)
            if np.any(mask):
                axes[3].plot(x_roll[mask], y[mask], lw=1.5, color=color, label=label)
    axes[3].axhline(1.0e-3, color="0.1", lw=1.1, ls="--", label="1e-3 criterion")
    axes[3].set_yscale("log")
    axes[3].set_ylabel("final-window metric")
    axes[3].set_xlabel("cumulative cycle")
    axes[3].set_title("Strict convergence terms", loc="left", fontsize=11)
    axes[3].legend(ncol=4, frameon=False, loc="upper right")
    criterion_factor = outlook.get("delta_r_criterion_factor")
    if criterion_factor is not None:
        axes[3].text(
            0.02,
            0.94,
            f"Delta R term is {float(criterion_factor):.0f}x the 1e-3 criterion",
            transform=axes[3].transAxes,
            va="top",
            ha="left",
            fontsize=10,
            color="0.15",
        )
    if shown_rows:
        axes[3].set_xlim(float(shown_rows[0]["period_number"]), float(shown_rows[-1]["period_number"]))

    for ax in axes:
        ax.grid(axis="y", color="0.88", lw=0.7)
        ax.set_axisbelow(True)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    record = find_record(workspace, args.model)
    source, source_kind, rows = source_rows(record, int(args.rolling_window))
    model_id = str(record["model_id"])
    output_prefix = args.output_prefix
    if output_prefix is None:
        output_prefix = workspace / "output" / f"{model_id}_growth_diagnostic"
    output_prefix = output_prefix.resolve()

    if not rows:
        payload = {
            "generated_at": now_iso(),
            "workspace": str(workspace),
            "model_id": model_id,
            "run_name": record.get("run_name"),
            "source": source,
            "source_kind": source_kind,
            "cycle_count": 0,
            "first_period": None,
            "last_period": None,
            "rolling_window": int(args.rolling_window),
            "fits": {},
            "latest_rolling": None,
            "growth_outlook": {
                "status": "no convergence rows available yet",
                "tolerance": CONVERGENCE_TOLERANCE,
            },
            "status": "no convergence rows available yet",
        }
        json_path = output_prefix.with_suffix(".json")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json_path)
        return 0

    fits: dict[str, dict[str, dict[str, object]]] = {}
    for window in args.windows:
        fits[str(window)] = {
            key: fit_window(rows, key, int(window))
            for key in ("delta_r", "period_days", "gamma", "max_vsurf_div_cs")
        }
    rolling = rolling_rows(rows, int(args.rolling_window))
    outlook = growth_outlook(fits, rolling, rows)
    payload = {
        "generated_at": now_iso(),
        "workspace": str(workspace),
        "model_id": model_id,
        "run_name": record.get("run_name"),
        "source": source,
        "source_kind": source_kind,
        "cycle_count": len(rows),
        "first_period": rows[0]["period_number"] if rows else None,
        "last_period": rows[-1]["period_number"] if rows else None,
        "rolling_window": int(args.rolling_window),
        "fits": fits,
        "latest_rolling": rolling[-1] if rolling else None,
        "growth_outlook": outlook,
    }
    json_path = output_prefix.with_suffix(".json")
    png_path = output_prefix.with_suffix(".png")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    plot_diagnostic(
        png_path,
        rows,
        fits,
        rolling,
        outlook,
        model_id=model_id,
        plot_tail=int(args.plot_tail),
    )
    print(json_path)
    print(png_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
