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


def plot_diagnostic(
    path: Path,
    rows: list[dict[str, float]],
    fits: dict[str, dict[str, dict[str, object]]],
    rolling: list[dict[str, object]],
    *,
    model_id: str,
    plot_tail: int,
) -> None:
    shown_rows = rows[-plot_tail:] if len(rows) > plot_tail else rows
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 9.5), sharex=False, constrained_layout=True)
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
        axes[0].text(
            0.02,
            0.94,
            f"last {int(delta_fit['window_cycles'])} cycles: e-fold {efold:.0f} cycles",
            transform=axes[0].transAxes,
            va="top",
            ha="left",
            fontsize=10,
            color="0.15",
        )

    x_gamma, y_gamma = safe_array(shown_rows, "gamma")
    gamma_mask = y_gamma > 0.0
    x_gamma = x_gamma[gamma_mask]
    y_gamma = y_gamma[gamma_mask]
    axes[1].plot(x_gamma, y_gamma, color="#669BBC", lw=1.4, label="Gamma")
    axes[1].axhline(0.0, color="0.2", lw=1.0, ls="--")
    axes[1].set_ylabel("Gamma")
    axes[1].set_title("Growth-rate history", loc="left", fontsize=11)

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
                axes[2].plot(x_roll[mask], y[mask], lw=1.5, color=color, label=label)
    axes[2].axhline(1.0e-3, color="0.1", lw=1.1, ls="--", label="1e-3 criterion")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("final-window metric")
    axes[2].set_xlabel("cumulative cycle")
    axes[2].set_title("Strict convergence terms", loc="left", fontsize=11)
    axes[2].legend(ncol=4, frameon=False, loc="upper right")
    if shown_rows:
        axes[2].set_xlim(float(shown_rows[0]["period_number"]), float(shown_rows[-1]["period_number"]))

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

    fits: dict[str, dict[str, dict[str, object]]] = {}
    for window in args.windows:
        fits[str(window)] = {
            key: fit_window(rows, key, int(window))
            for key in ("delta_r", "period_days", "gamma")
        }
    rolling = rolling_rows(rows, int(args.rolling_window))
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
        model_id=model_id,
        plot_tail=int(args.plot_tail),
    )
    print(json_path)
    print(png_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
