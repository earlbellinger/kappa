from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rsp_batch_convergence import (
    absolute_peak_to_peak,
    attach_steps_from_period_log,
    fractional_peak_to_peak,
    median_value,
    read_json,
    select_exact_source,
    select_log_source,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_WORKSPACE = ROOT / "rsp_batch_runs"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build rolling RSP limit-cycle convergence diagnostics."
    )
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--last-cycles", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=1.0e-3)
    return parser.parse_args()


def safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def source_rows(record: dict[str, object], last_cycles: int) -> tuple[str | None, str, list[dict[str, float]]]:
    run_dir = Path(str(record["run_dir"]))
    output_dir = Path(str(record["output_dir"]))
    exact_label, _exact_path, exact_rows = select_exact_source(run_dir, last_cycles)
    if exact_rows:
        exact_rows, _steps_label, _steps_path = attach_steps_from_period_log(
            exact_rows,
            output_dir,
            last_cycles,
        )
        return exact_label, "history_exact_rsp_columns", exact_rows
    log_label, _log_path, log_rows, _stops = select_log_source(output_dir, last_cycles)
    return log_label, "period_log_fallback", log_rows


def rolling_rows(
    record: dict[str, object],
    *,
    last_cycles: int,
    tolerance: float,
) -> list[dict[str, object]]:
    source, source_kind, rows = source_rows(record, last_cycles)
    model_id = str(record["model_id"])
    run_name = str(record["run_name"])
    if len(rows) < last_cycles:
        return []
    output: list[dict[str, object]] = []
    for end in range(last_cycles, len(rows) + 1):
        tail = rows[end - last_cycles : end]
        gamma = np.asarray([row.get("gamma", np.nan) for row in tail], dtype=float)
        period = np.asarray([row["period_days"] for row in tail], dtype=float)
        delta_r = np.asarray([row["delta_r"] for row in tail], dtype=float)
        steps = np.asarray([row.get("steps", np.nan) for row in tail], dtype=float)
        gamma_ptp = absolute_peak_to_peak(gamma)
        period_frac = fractional_peak_to_peak(period)
        delta_r_frac = fractional_peak_to_peak(delta_r)
        has_gamma = gamma_ptp is not None and bool(np.any(np.isfinite(gamma)))
        converged_gamma = bool(has_gamma and gamma_ptp <= tolerance)
        converged_period = bool(period_frac is not None and period_frac <= tolerance)
        converged_delta_r = bool(delta_r_frac is not None and delta_r_frac <= tolerance)
        output.append(
            {
                "model_id": model_id,
                "run_name": run_name,
                "source": source,
                "source_kind": source_kind,
                "window_cycles": int(last_cycles),
                "window_start_period": float(tail[0]["period_number"]),
                "window_end_period": float(tail[-1]["period_number"]),
                "gamma_peak_to_peak": gamma_ptp,
                "period_fractional_peak_to_peak": period_frac,
                "delta_r_fractional_peak_to_peak": delta_r_frac,
                "steps_median": median_value(steps),
                "steps_min": float(np.nanmin(steps)) if np.any(np.isfinite(steps)) else None,
                "steps_max": float(np.nanmax(steps)) if np.any(np.isfinite(steps)) else None,
                "converged_gamma": converged_gamma,
                "converged_period": converged_period,
                "converged_delta_r": converged_delta_r,
                "converged_exact": bool(converged_gamma and converged_period and converged_delta_r),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "model_id",
        "run_name",
        "source",
        "source_kind",
        "window_cycles",
        "window_start_period",
        "window_end_period",
        "gamma_peak_to_peak",
        "period_fractional_peak_to_peak",
        "delta_r_fractional_peak_to_peak",
        "steps_median",
        "steps_min",
        "steps_max",
        "converged_gamma",
        "converged_period",
        "converged_delta_r",
        "converged_exact",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def latest_by_model(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    latest: dict[str, dict[str, object]] = {}
    for row in rows:
        model_id = str(row["model_id"])
        previous = latest.get(model_id)
        if previous is None or float(row["window_end_period"]) > float(previous["window_end_period"]):
            latest[model_id] = row
    return latest


def plot_trends(
    path: Path,
    rows: list[dict[str, object]],
    tolerance: float,
    *,
    title: str = "Rolling final-100-cycle convergence diagnostics",
) -> None:
    if not rows:
        return
    models = sorted({str(row["model_id"]) for row in rows})
    color_map = plt.get_cmap("tab10")
    colors = {model_id: color_map(index % 10) for index, model_id in enumerate(models)}
    fig, axes = plt.subplots(4, 1, figsize=(12.5, 11), sharex=True, constrained_layout=True)
    fig.suptitle(title, fontsize=16)
    panels = [
        ("gamma_peak_to_peak", "Gamma peak-to-peak", True, tolerance),
        ("period_fractional_peak_to_peak", "P fractional peak-to-peak", True, tolerance),
        ("delta_r_fractional_peak_to_peak", "Delta R fractional peak-to-peak", True, tolerance),
        ("steps_median", "median steps per cycle", False, 1000.0),
    ]
    for model_id in models:
        model_rows = [row for row in rows if str(row["model_id"]) == model_id]
        x = np.asarray([float(row["window_end_period"]) for row in model_rows], dtype=float)
        label = model_id.replace("model_", "m")
        style = "-" if model_rows[-1].get("source_kind") == "history_exact_rsp_columns" else "--"
        for ax, (key, _ylabel, use_log, _line_value) in zip(axes, panels):
            y = np.asarray(
                [
                    np.nan if safe_float(row.get(key)) is None else float(row[key])
                    for row in model_rows
                ],
                dtype=float,
            )
            if not np.any(np.isfinite(y)):
                continue
            ax.plot(x, y, style, lw=1.35, color=colors[model_id], alpha=0.9, label=label)

    for ax, (_key, ylabel, use_log, line_value) in zip(axes, panels):
        ax.axhline(line_value, color="0.1", lw=1.1, ls="--")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="0.88", lw=0.7)
        ax.set_axisbelow(True)
        if use_log:
            ax.set_yscale("log")
            finite_values = []
            for line in ax.lines:
                finite_values.extend([value for value in line.get_ydata() if np.isfinite(value) and value > 0])
            if finite_values:
                ymin = max(min(finite_values + [line_value]) / 4.0, 1e-5)
                ymax = max(finite_values + [line_value]) * 2.5
                ax.set_ylim(ymin, ymax)
        else:
            finite_values = []
            for line in ax.lines:
                finite_values.extend([value for value in line.get_ydata() if np.isfinite(value)])
            ymax = max(finite_values + [line_value]) * 1.15 if finite_values else line_value * 1.15
            ax.set_ylim(0, ymax)
    axes[-1].set_xlabel("window end period")
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, ncol=min(5, len(handles)), frameon=False, loc="upper right")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    manifest = read_json(workspace / "inputs" / "manifest.json")
    if not isinstance(manifest, list):
        raise RuntimeError(f"Could not read manifest list from {workspace / 'inputs' / 'manifest.json'}")
    rows: list[dict[str, object]] = []
    for record in manifest:
        if isinstance(record, dict):
            rows.extend(
                rolling_rows(
                    record,
                    last_cycles=int(args.last_cycles),
                    tolerance=float(args.tolerance),
                )
            )

    output_dir = workspace / "output"
    json_path = output_dir / "convergence_trends_last100.json"
    csv_path = output_dir / "convergence_trends_last100.csv"
    png_path = output_dir / "convergence_trends_last100.png"
    exact_png_path = output_dir / "convergence_trends_exact_last100.png"
    latest = latest_by_model(rows)
    payload = {
        "generated_at": now_iso(),
        "workspace": str(workspace),
        "criterion": {
            "window_cycles": int(args.last_cycles),
            "tolerance": float(args.tolerance),
        },
        "latest_by_model": latest,
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_csv(csv_path, rows)
    plot_trends(png_path, rows, float(args.tolerance))
    exact_rows = [row for row in rows if row.get("source_kind") == "history_exact_rsp_columns"]
    if exact_rows:
        plot_trends(
            exact_png_path,
            exact_rows,
            float(args.tolerance),
            title="Rolling final-100-cycle convergence diagnostics: exact-history runs",
        )
    print(json_path)
    print(csv_path)
    print(png_path)
    if exact_rows:
        print(exact_png_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
