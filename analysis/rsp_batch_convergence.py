from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from rsp_batch_run import read_history


ROOT = Path(__file__).resolve().parent
DEFAULT_WORKSPACE = ROOT / "rsp_batch_runs"
PERIOD_LINE_RE = re.compile(
    r"^\s*period\s+(?P<n>\d+)\s+(?P<period>[+\-0-9.EeDd]+)\s+"
    r"delta R\s+(?P<delta_r>[+\-0-9.EeDd]+)\s+"
    r"(?:max\s+vsurf/cs\s+(?P<max_vsurf_div_cs>[+\-0-9.EeDd]+)\s+)?"
    r".*?"
    r"steps\s+(?P<steps>\d+)"
)
STOP_RE = re.compile(r"stop because\s+(?P<reason>.+)")
TERMINATION_RE = re.compile(r"termination code:\s+(?P<reason>.+)")
EXACT_COLUMNS = ("rsp_GREKM", "rsp_DeltaR", "rsp_num_periods", "rsp_period_in_days")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate RSP limit-cycle convergence over the final N recorded cycles."
    )
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--last-cycles", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=1.0e-3)
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional model_id or run_name list to refresh instead of parsing the full manifest.",
    )
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="When --models is used, merge refreshed rows into the existing summary file.",
    )
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict | list:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def record_matches_models(record: dict[str, object], selected: set[str]) -> bool:
    if not selected:
        return True
    return str(record.get("model_id")) in selected or str(record.get("run_name")) in selected


def ordered_records(manifest: list[object], selected: set[str]) -> list[dict[str, object]]:
    return [
        record
        for record in manifest
        if isinstance(record, dict) and record_matches_models(record, selected)
    ]


def merge_summary_rows(
    existing_payload: dict | list,
    refreshed_rows: list[dict[str, object]],
    manifest: list[object],
) -> list[dict[str, object]]:
    if not isinstance(existing_payload, dict):
        return refreshed_rows
    existing_rows = existing_payload.get("models")
    if not isinstance(existing_rows, list):
        return refreshed_rows
    by_model: dict[str, dict[str, object]] = {
        str(row.get("model_id")): row
        for row in existing_rows
        if isinstance(row, dict) and row.get("model_id")
    }
    for row in refreshed_rows:
        by_model[str(row.get("model_id"))] = row
    ordered_ids = [
        str(record.get("model_id"))
        for record in manifest
        if isinstance(record, dict) and record.get("model_id") in by_model
    ]
    ordered_ids.extend(model_id for model_id in by_model if model_id not in ordered_ids)
    return [by_model[model_id] for model_id in ordered_ids]


def fortran_float(text: str) -> float:
    return float(str(text).replace("D", "E").replace("d", "e"))


def finite_array(values: list[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def fractional_peak_to_peak(values: list[float] | np.ndarray) -> float | None:
    array = finite_array(values)
    if array.size == 0:
        return None
    scale = float(np.nanmedian(np.abs(array)))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return float((np.nanmax(array) - np.nanmin(array)) / scale)


def absolute_peak_to_peak(values: list[float] | np.ndarray) -> float | None:
    array = finite_array(values)
    if array.size == 0:
        return None
    return float(np.nanmax(array) - np.nanmin(array))


def median_value(values: list[float] | np.ndarray) -> float | None:
    array = finite_array(values)
    if array.size == 0:
        return None
    return float(np.nanmedian(array))


def linear_slope_per_cycle(values: list[float] | np.ndarray) -> float | None:
    array = finite_array(values)
    if array.size < 2:
        return None
    x = np.arange(array.size, dtype=float)
    x -= float(np.nanmean(x))
    y = array - float(np.nanmean(array))
    denominator = float(np.nansum(x * x))
    if denominator <= 0.0:
        return None
    return float(np.nansum(x * y) / denominator)


def history_candidates(run_dir: Path) -> list[tuple[str, Path]]:
    resume_histories = sorted(
        run_dir.glob("LOGS_continue_saturation_resume_*/history.data"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    candidates: list[tuple[str, Path]] = [
        (f"continue_saturation_resume:{path.parent.name}", path) for path in resume_histories
    ]
    candidates.extend(
        [
            ("continue_saturation", run_dir / "LOGS_continue_saturation" / "history.data"),
            ("saturation", run_dir / "LOGS_saturation" / "history.data"),
            ("deep", run_dir / "LOGS" / "history.data"),
        ]
    )
    seen: set[Path] = set()
    unique: list[tuple[str, Path]] = []
    for label, path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append((label, path))
    return unique


def cumulative_history_candidates(run_dir: Path) -> list[tuple[str, Path]]:
    resume_histories = sorted(
        run_dir.glob("LOGS_continue_saturation_resume_*/history.data"),
        key=lambda path: path.stat().st_mtime,
    )
    candidates: list[tuple[str, Path]] = [
        ("saturation", run_dir / "LOGS_saturation" / "history.data"),
        ("continue_saturation", run_dir / "LOGS_continue_saturation" / "history.data"),
    ]
    candidates.extend((f"continue_saturation_resume:{path.parent.name}", path) for path in resume_histories)
    return [(label, path) for label, path in candidates if path.exists()]


def cumulative_log_candidates(output_dir: Path) -> list[tuple[str, Path]]:
    resume_logs = sorted(
        (output_dir / "logs").glob("continue_saturation_resume_*.log"),
        key=lambda path: path.stat().st_mtime,
    )
    candidates: list[tuple[str, Path]] = [
        ("create", output_dir / "logs" / "create.log"),
        ("continue_saturation", output_dir / "logs" / "continue_saturation.log"),
    ]
    candidates.extend((path.stem, path) for path in resume_logs)
    return [(label, path) for label, path in candidates if path.exists()]


def exact_cycle_rows(history_path: Path) -> list[dict[str, float]]:
    history = read_history(history_path)
    if any(name not in history for name in EXACT_COLUMNS):
        return []
    period_numbers = np.asarray(history["rsp_num_periods"], dtype=float)
    grekm = np.asarray(history["rsp_GREKM"], dtype=float)
    delta_r = np.asarray(history["rsp_DeltaR"], dtype=float)
    period_days = np.asarray(history["rsp_period_in_days"], dtype=float)
    rows: list[dict[str, float]] = []
    finite = (
        np.isfinite(period_numbers)
        & np.isfinite(grekm)
        & np.isfinite(delta_r)
        & np.isfinite(period_days)
        & (period_numbers > 0)
    )
    if not np.any(finite):
        return []
    indices = np.flatnonzero(finite)
    last_index_by_period: dict[int, int] = {}
    for index in indices:
        last_index_by_period[int(round(float(period_numbers[index])))] = int(index)
    for period_number in sorted(last_index_by_period):
        index = last_index_by_period[period_number]
        rows.append(
            {
                "period_number": float(period_number),
                "gamma": float(grekm[index]),
                "period_days": float(period_days[index]),
                "delta_r": float(delta_r[index]),
            }
        )
    return rows


def append_cumulative_rows(
    combined: list[dict[str, float]],
    rows: list[dict[str, float]],
    source_label: str,
) -> list[dict[str, float]]:
    if not rows:
        return combined
    if not combined:
        offset = 0.0
        keep_periods_after: float | None = None
    else:
        last_cumulative_period = float(combined[-1]["period_number"])
        first_segment_period = float(rows[0]["period_number"])
        last_segment_period = float(rows[-1]["period_number"])
        if first_segment_period <= last_cumulative_period < last_segment_period:
            # Resume histories can overlap the previous history while retaining
            # absolute period numbers. Keep only the new cycles in that case.
            offset = 0.0
            keep_periods_after = last_cumulative_period
        elif first_segment_period <= last_cumulative_period:
            offset = last_cumulative_period
            keep_periods_after = None
        else:
            offset = 0.0
            keep_periods_after = None
    for row in rows:
        original_period = float(row["period_number"])
        if keep_periods_after is not None and original_period <= keep_periods_after:
            continue
        copied = dict(row)
        copied["segment_period_number"] = original_period
        copied["segment_label"] = source_label
        copied["period_number"] = original_period + offset
        if combined and copied["period_number"] <= float(combined[-1]["period_number"]):
            continue
        combined.append(copied)
    return combined


def cumulative_exact_source(run_dir: Path) -> tuple[str | None, list[Path], list[dict[str, float]]]:
    combined: list[dict[str, float]] = []
    labels: list[str] = []
    paths: list[Path] = []
    for label, path in cumulative_history_candidates(run_dir):
        rows = exact_cycle_rows(path)
        if not rows:
            continue
        previous_count = len(combined)
        append_cumulative_rows(combined, rows, label)
        if len(combined) > previous_count:
            labels.append(label)
            paths.append(path)
    if not combined:
        return None, [], []
    return "cumulative:" + "+".join(labels), paths, combined


def parse_period_log(path: Path) -> tuple[list[dict[str, float]], list[str]]:
    segments: list[tuple[list[dict[str, float]], list[str]]] = []
    rows: list[dict[str, float]] = []
    stops: list[str] = []
    last_period_number: int | None = None
    if not path.exists():
        return rows, stops
    for line in path.read_text(errors="replace").splitlines():
        match = PERIOD_LINE_RE.match(line)
        if match:
            period_number = int(match.group("n"))
            if last_period_number is not None and period_number <= last_period_number and rows:
                segments.append((rows, stops))
                rows = []
                stops = []
            row = {
                "period_number": float(period_number),
                "period_days": fortran_float(match.group("period")),
                "delta_r": fortran_float(match.group("delta_r")),
                "steps": float(int(match.group("steps"))),
            }
            max_vsurf = match.group("max_vsurf_div_cs")
            if max_vsurf is not None:
                row["max_vsurf_div_cs"] = fortran_float(max_vsurf)
            rows.append(row)
            last_period_number = period_number
        stop = STOP_RE.search(line)
        if stop:
            stops.append(stop.group("reason").strip())
        termination = TERMINATION_RE.search(line)
        if termination:
            stops.append(f"termination code: {termination.group('reason').strip()}")
    if rows or stops:
        segments.append((rows, stops))
    if segments:
        return segments[-1]
    return rows, stops


def cumulative_log_source(output_dir: Path) -> tuple[str | None, list[Path], list[dict[str, float]], list[str]]:
    combined: list[dict[str, float]] = []
    labels: list[str] = []
    paths: list[Path] = []
    latest_stops: list[str] = []
    for label, path in cumulative_log_candidates(output_dir):
        rows, stops = parse_period_log(path)
        if not rows:
            if stops:
                latest_stops = stops
            continue
        previous_count = len(combined)
        append_cumulative_rows(combined, rows, label)
        if len(combined) > previous_count:
            labels.append(label)
            paths.append(path)
            latest_stops = stops
    if not combined:
        return None, [], [], latest_stops
    return "cumulative:" + "+".join(labels), paths, combined, latest_stops


def log_candidates(output_dir: Path) -> list[tuple[str, Path]]:
    resume_logs = sorted(
        (output_dir / "logs").glob("continue_saturation_resume_*.log"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    candidates: list[tuple[str, Path]] = [(path.stem, path) for path in resume_logs]
    candidates.extend(
        [
            ("continue_saturation", output_dir / "logs" / "continue_saturation.log"),
            ("create", output_dir / "logs" / "create.log"),
        ]
    )
    return [(label, path) for label, path in candidates if path.exists()]


def select_exact_source(run_dir: Path, required_cycles: int) -> tuple[str | None, Path | None, list[dict[str, float]]]:
    cumulative_label, cumulative_paths, cumulative_rows = cumulative_exact_source(run_dir)
    if cumulative_rows:
        latest_path = cumulative_paths[-1] if cumulative_paths else None
        return cumulative_label, latest_path, cumulative_rows

    parsed: list[tuple[str, Path, list[dict[str, float]]]] = []
    for label, path in history_candidates(run_dir):
        if not path.exists():
            continue
        rows = exact_cycle_rows(path)
        if rows:
            parsed.append((label, path, rows))
    for label, path, rows in parsed:
        if len(rows) >= required_cycles:
            return label, path, rows
    if parsed:
        return parsed[0]
    return None, None, []


def select_log_source(output_dir: Path, required_cycles: int) -> tuple[str | None, Path | None, list[dict[str, float]], list[str]]:
    cumulative_label, cumulative_paths, cumulative_rows, cumulative_stops = cumulative_log_source(output_dir)
    if cumulative_rows:
        latest_path = cumulative_paths[-1] if cumulative_paths else None
        return cumulative_label, latest_path, cumulative_rows, cumulative_stops

    parsed: list[tuple[str, Path, list[dict[str, float]], list[str]]] = []
    for label, path in log_candidates(output_dir):
        rows, stops = parse_period_log(path)
        if rows:
            parsed.append((label, path, rows, stops))
    for label, path, rows, stops in parsed:
        if len(rows) >= required_cycles:
            return label, path, rows, stops
    if parsed:
        return parsed[0]
    return None, None, [], []


def attach_steps_from_period_log(
    rows: list[dict[str, float]],
    output_dir: Path,
    required_cycles: int,
) -> tuple[list[dict[str, float]], str | None, Path | None]:
    log_label, log_path, log_rows, _stops = select_log_source(output_dir, required_cycles)
    if not rows or not log_rows:
        return rows, log_label, log_path
    metrics_by_period: dict[int, dict[str, float]] = {}
    for row in log_rows:
        period = int(round(float(row["period_number"])))
        metrics: dict[str, float] = {}
        for key in ("steps", "max_vsurf_div_cs"):
            if key in row and np.isfinite(float(row[key])):
                metrics[key] = float(row[key])
        if metrics:
            metrics_by_period[period] = metrics
    merged: list[dict[str, float]] = []
    for row in rows:
        copied = dict(row)
        metrics = metrics_by_period.get(int(round(float(row["period_number"]))))
        if metrics is not None:
            copied.update(metrics)
        merged.append(copied)
    return merged, log_label, log_path


def summarize_rows(
    rows: list[dict[str, float]],
    *,
    last_cycles: int,
    tolerance: float,
    source_kind: str,
) -> dict[str, object]:
    if not rows:
        return {
            "cycle_count": 0,
            "last_cycle_count_used": 0,
            "source_kind": source_kind,
            "converged_exact": False,
        }
    tail = rows[-int(last_cycles) :]
    period = np.asarray([row["period_days"] for row in tail], dtype=float)
    delta_r = np.asarray([row["delta_r"] for row in tail], dtype=float)
    gamma = np.asarray([row.get("gamma", np.nan) for row in tail], dtype=float)
    steps = np.asarray([row.get("steps", np.nan) for row in tail], dtype=float)
    max_vsurf = np.asarray([row.get("max_vsurf_div_cs", np.nan) for row in tail], dtype=float)
    gamma_ptp = absolute_peak_to_peak(gamma)
    period_fraction = fractional_peak_to_peak(period)
    delta_r_fraction = fractional_peak_to_peak(delta_r)
    has_full_window = bool(len(tail) >= int(last_cycles))
    has_gamma = bool(gamma_ptp is not None and np.any(np.isfinite(gamma)))
    converged_gamma = bool(has_full_window and has_gamma and gamma_ptp <= tolerance)
    converged_period = bool(has_full_window and period_fraction is not None and period_fraction <= tolerance)
    converged_delta_r = bool(has_full_window and delta_r_fraction is not None and delta_r_fraction <= tolerance)
    result: dict[str, object] = {
        "source_kind": source_kind,
        "cycle_count": len(rows),
        "last_cycle_count_used": len(tail),
        "window_start_period_number": float(tail[0]["period_number"]),
        "window_end_period_number": float(tail[-1]["period_number"]),
        "last_period_number": float(rows[-1]["period_number"]),
        "gamma_peak_to_peak_last_window": gamma_ptp,
        "gamma_median_last_window": median_value(gamma),
        "gamma_first_last_window": [
            float(gamma[0]) if np.isfinite(gamma[0]) else None,
            float(gamma[-1]) if np.isfinite(gamma[-1]) else None,
        ],
        "gamma_slope_per_cycle_last_window": linear_slope_per_cycle(gamma),
        "period_fractional_peak_to_peak_last_window": period_fraction,
        "period_median_days_last_window": median_value(period),
        "period_first_last_days_window": [float(period[0]), float(period[-1])],
        "period_slope_days_per_cycle_last_window": linear_slope_per_cycle(period),
        "delta_r_fractional_peak_to_peak_last_window": delta_r_fraction,
        "delta_r_median_last_window": median_value(delta_r),
        "delta_r_first_last_window": [float(delta_r[0]), float(delta_r[-1])],
        "delta_r_slope_per_cycle_last_window": linear_slope_per_cycle(delta_r),
        "steps_median_last_window": median_value(steps),
        "steps_min_last_window": float(np.nanmin(steps)) if np.any(np.isfinite(steps)) else None,
        "steps_max_last_window": float(np.nanmax(steps)) if np.any(np.isfinite(steps)) else None,
        "max_vsurf_div_cs_median_last_window": median_value(max_vsurf),
        "max_vsurf_div_cs_first_last_window": [
            float(max_vsurf[0]) if np.isfinite(max_vsurf[0]) else None,
            float(max_vsurf[-1]) if np.isfinite(max_vsurf[-1]) else None,
        ],
        "max_vsurf_div_cs_slope_per_cycle_last_window": linear_slope_per_cycle(max_vsurf),
        "max_vsurf_div_cs_min_last_window": (
            float(np.nanmin(max_vsurf)) if np.any(np.isfinite(max_vsurf)) else None
        ),
        "max_vsurf_div_cs_max_last_window": (
            float(np.nanmax(max_vsurf)) if np.any(np.isfinite(max_vsurf)) else None
        ),
        "has_full_window": has_full_window,
        "has_gamma": has_gamma,
        "converged_gamma": converged_gamma,
        "converged_period": converged_period,
        "converged_delta_r": converged_delta_r,
        "converged_exact": bool(converged_gamma and converged_period and converged_delta_r),
    }
    result["limit_cycle_converged"] = result["converged_exact"]
    return result


def summarize_model(record: dict[str, object], last_cycles: int, tolerance: float) -> dict[str, object]:
    run_dir = Path(str(record["run_dir"]))
    output_dir = Path(str(record["output_dir"]))
    exact_label, exact_path, exact_rows = select_exact_source(run_dir, last_cycles)
    if exact_rows:
        exact_rows, steps_label, steps_path = attach_steps_from_period_log(
            exact_rows,
            output_dir,
            last_cycles,
        )
        summary = summarize_rows(
            exact_rows,
            last_cycles=last_cycles,
            tolerance=tolerance,
            source_kind="history_exact_rsp_columns",
        )
        summary.update(
            {
                "source": exact_label,
                "source_path": str(exact_path),
                "gamma_status": "exact rsp_GREKM history column",
                "stop_reasons": [],
                "grekm_stopped": None,
                "period_cap_stopped": None,
                "steps_source": steps_label,
                "steps_source_path": str(steps_path) if steps_path is not None else None,
            }
        )
    else:
        log_label, log_path, log_rows, stops = select_log_source(output_dir, last_cycles)
        summary = summarize_rows(
            log_rows,
            last_cycles=last_cycles,
            tolerance=tolerance,
            source_kind="period_log_fallback",
        )
        grekm_stopped = any("GREKM_avg_abs < RSP_GREKM_avg_abs_limit" in reason for reason in stops)
        summary.update(
            {
                "source": log_label,
                "source_path": str(log_path) if log_path is not None else None,
                "gamma_status": (
                    "not recorded in history; GREKM stop is a weak proxy only"
                    if grekm_stopped
                    else "not recorded in history"
                ),
                "stop_reasons": stops[-3:],
                "grekm_stopped": grekm_stopped,
                "period_cap_stopped": any(
                    "period_number >= max_period_number" in reason
                    or "reached max number of periods" in reason
                    for reason in stops
                ),
                "converged_gamma": False,
                "converged_exact": False,
                "limit_cycle_converged": False,
            }
        )
    summary.update(
        {
            "model_id": record["model_id"],
            "run_name": record["run_name"],
            "tolerance": float(tolerance),
            "required_last_cycles": int(last_cycles),
        }
    )
    return summary


def write_csv(path: Path, models: list[dict[str, object]]) -> None:
    fields = [
        "model_id",
        "run_name",
        "source_kind",
        "source",
        "cycle_count",
        "last_cycle_count_used",
        "last_period_number",
        "gamma_peak_to_peak_last_window",
        "period_fractional_peak_to_peak_last_window",
        "delta_r_fractional_peak_to_peak_last_window",
        "steps_median_last_window",
        "steps_min_last_window",
        "steps_max_last_window",
        "max_vsurf_div_cs_median_last_window",
        "max_vsurf_div_cs_first_last_window",
        "max_vsurf_div_cs_slope_per_cycle_last_window",
        "max_vsurf_div_cs_min_last_window",
        "max_vsurf_div_cs_max_last_window",
        "grekm_stopped",
        "period_cap_stopped",
        "converged_gamma",
        "converged_period",
        "converged_delta_r",
        "converged_exact",
        "gamma_status",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in models:
            writer.writerow({field: row.get(field) for field in fields})


def as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def plot_summary(path: Path, models: list[dict[str, object]], tolerance: float) -> None:
    shown = [model for model in models if model.get("model_id") != "model_000"]
    labels = [str(model.get("model_id", "?")).replace("model_", "m") for model in shown]
    x = np.arange(len(shown), dtype=float)

    colors = []
    hatches = []
    for model in shown:
        if bool(model.get("converged_exact")):
            colors.append("#2A9D8F")
            hatches.append("")
        elif model.get("source_kind") == "history_exact_rsp_columns":
            colors.append("#F4A261")
            hatches.append("")
        else:
            colors.append("#C1121F")
            hatches.append("//")

    fig, axes = plt.subplots(4, 1, figsize=(12, 10.5), sharex=True, constrained_layout=True)
    fig.suptitle("Strict limit-cycle convergence over final 100 recorded cycles", fontsize=16)

    panels = [
        (
            axes[0],
            "period_fractional_peak_to_peak_last_window",
            "P variation",
            "fractional peak-to-peak",
            True,
        ),
        (
            axes[1],
            "delta_r_fractional_peak_to_peak_last_window",
            "Delta R variation",
            "fractional peak-to-peak",
            True,
        ),
        (
            axes[2],
            "gamma_peak_to_peak_last_window",
            "Gamma variation",
            "absolute peak-to-peak",
            True,
        ),
        (
            axes[3],
            "steps_median_last_window",
            "time resolution",
            "median steps per cycle",
            False,
        ),
    ]

    for ax, key, title, ylabel, use_tolerance in panels:
        values = [as_float(model.get(key)) for model in shown]
        finite_values = [value for value in values if value is not None and value > 0.0]
        heights = [value if value is not None and value > 0.0 else np.nan for value in values]
        bars = ax.bar(x, heights, color=colors, edgecolor="0.15", linewidth=0.6)
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=12)
        ax.grid(axis="y", color="0.88", lw=0.7)
        ax.set_axisbelow(True)

        if use_tolerance:
            ax.axhline(tolerance, color="0.1", ls="--", lw=1.1, label="1e-3 criterion")
            ax.set_yscale("log")
            ymax = max(finite_values + [tolerance]) * 2.5
            ymin = min(finite_values + [tolerance]) / 4.0
            ax.set_ylim(max(ymin, 1e-5), ymax)
        else:
            ax.axhline(1000.0, color="0.1", ls=":", lw=1.1, label="1000 steps/cycle")
            ymax = max(finite_values + [1000.0]) * 1.2
            ax.set_ylim(0.0, ymax)

        for xi, value, model in zip(x, values, shown):
            if value is None:
                ax.text(
                    xi,
                    0.45,
                    "missing",
                    ha="center",
                    va="center",
                    rotation=90,
                    fontsize=8,
                    color="0.35",
                    transform=ax.get_xaxis_transform(),
                )
            elif use_tolerance and value <= tolerance and bool(model.get("converged_exact")):
                ax.text(xi, value * 1.35, "ok", ha="center", va="bottom", fontsize=8, color="#1B4332")

        ax.legend(frameon=False, fontsize=8, loc="upper left")

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels)
    axes[-1].set_xlabel("batch model")
    axes[-1].legend(
        handles=[
            Patch(facecolor="#2A9D8F", edgecolor="0.15", label="strictly converged"),
            Patch(facecolor="#F4A261", edgecolor="0.15", label="exact columns, not converged"),
            Patch(facecolor="#C1121F", edgecolor="0.15", hatch="//", label="period-log fallback"),
        ],
        frameon=False,
        fontsize=8,
        loc="upper right",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    manifest = read_json(workspace / "inputs" / "manifest.json")
    if not isinstance(manifest, list):
        raise RuntimeError(f"Could not read manifest list from {workspace / 'inputs' / 'manifest.json'}")
    selected = set(args.models or [])
    records = ordered_records(manifest, selected)
    models = [
        summarize_model(record, int(args.last_cycles), float(args.tolerance))
        for record in records
    ]
    output_dir = workspace / "output"
    json_path = output_dir / "convergence_summary_last100.json"
    if selected and args.merge_existing:
        models = merge_summary_rows(read_json(json_path), models, manifest)
    payload = {
        "generated_at": now_iso(),
        "workspace": str(workspace),
        "criterion": {
            "required_last_cycles": int(args.last_cycles),
            "tolerance": float(args.tolerance),
            "period_numbering": (
                "create/saturation, continuation, and continuation-resume histories are merged "
                "chronologically; when a later stage resets its period counter, period numbers are "
                "offset so the convergence window follows the cumulative integration"
            ),
            "gamma": "absolute peak-to-peak rsp_GREKM over final window <= tolerance",
            "period": "fractional peak-to-peak rsp_period_in_days or log period over final window <= tolerance",
            "delta_r": "fractional peak-to-peak rsp_DeltaR or log delta R over final window <= tolerance",
        },
        "models": models,
    }
    csv_path = output_dir / "convergence_summary_last100.csv"
    png_path = output_dir / "convergence_summary_last100.png"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_csv(csv_path, models)
    plot_summary(png_path, models, float(args.tolerance))
    print(json_path)
    print(csv_path)
    print(png_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
