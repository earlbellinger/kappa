from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_WORKSPACE = ROOT / "rsp_batch_runs"
DEFAULT_MODELS = ("model_007", "model_008", "model_009")
TOLERANCE = 1.0e-3
EXPECTED_CONTINUE_MAX_PERIODS = 5000
EXPECTED_TARGET_STEPS = 1000

INLIST_KEYS = (
    "RSP_max_num_periods",
    "RSP_GREKM_avg_abs_limit",
    "RSP_target_steps_per_cycle",
    "max_num_profile_models",
    "profile_interval",
    "history_interval",
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the strict convergence gate for active RSP batch models.")
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    return parser.parse_args()


def read_json(path: Path) -> object:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def rows_by_model(payload: object, key: str = "models") -> dict[str, dict[str, object]]:
    if not isinstance(payload, dict):
        return {}
    rows = payload.get(key, [])
    if not isinstance(rows, list):
        return {}
    result: dict[str, dict[str, object]] = {}
    for row in rows:
        if isinstance(row, dict) and row.get("model_id"):
            result[str(row["model_id"])] = row
    return result


def manifest_by_model(workspace: Path) -> dict[str, dict[str, object]]:
    manifest = read_json(workspace / "inputs" / "manifest.json")
    if not isinstance(manifest, list):
        return {}
    return {
        str(row["model_id"]): row
        for row in manifest
        if isinstance(row, dict) and row.get("model_id")
    }


def parse_number(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip().rstrip(",").replace("d", "e").replace("D", "e")
    try:
        return float(text)
    except ValueError:
        return None


def parse_inlist(path: Path) -> dict[str, object]:
    values: dict[str, object] = {}
    if not path.exists():
        return values
    text = path.read_text(errors="replace")
    for key in INLIST_KEYS:
        match = re.search(rf"(?m)^\s*{re.escape(key)}\s*=\s*([^!\n]+)", text)
        if not match:
            continue
        raw = match.group(1).strip().rstrip(",")
        number = parse_number(raw)
        values[key] = number if number is not None else raw
    return values


def active_inlist_name(active_stage: object) -> str:
    stage = str(active_stage or "")
    if stage == "create":
        return "inlist_create"
    if stage == "continue_saturation":
        return "inlist_continue_saturation"
    if stage == "deep2cycles":
        return "inlist_deep2cycles"
    return ""


def ratio(value: object, tolerance: float) -> float | None:
    number = parse_number(value)
    if number is None:
        return None
    return number / tolerance


def bool_pass(value: object, tolerance: float) -> bool | None:
    number = parse_number(value)
    if number is None:
        return None
    return number <= tolerance


def post_actions_by_model(payload: object) -> dict[str, dict[str, object]]:
    if not isinstance(payload, dict):
        return {}
    actions = payload.get("actions", [])
    if not isinstance(actions, list):
        return {}
    result: dict[str, dict[str, object]] = {}
    for action in actions:
        if isinstance(action, dict) and action.get("model_id"):
            result[str(action["model_id"])] = action
    return result


def summarize_model(
    model_id: str,
    record: dict[str, object],
    live: dict[str, object],
    convergence: dict[str, object],
    forecast: dict[str, object],
    post_action: dict[str, object],
    tolerance: float,
) -> dict[str, object]:
    run_dir = Path(str(record.get("run_dir", "")))
    active_stage = live.get("active_stage")
    active_inlist = active_inlist_name(active_stage)
    active_config = parse_inlist(run_dir / active_inlist) if active_inlist else {}
    continue_config = parse_inlist(run_dir / "inlist_continue_saturation")
    deep_config = parse_inlist(run_dir / "inlist_deep2cycles")

    gamma = convergence.get("gamma_peak_to_peak_last_window")
    period = convergence.get("period_fractional_peak_to_peak_last_window")
    delta_r = convergence.get("delta_r_fractional_peak_to_peak_last_window")
    converged_exact = convergence.get("converged_exact") is True
    blocking = []
    if bool_pass(gamma, tolerance) is False:
        blocking.append("gamma")
    if bool_pass(period, tolerance) is False:
        blocking.append("period")
    if bool_pass(delta_r, tolerance) is False:
        blocking.append("delta_r")

    continue_max = parse_number(continue_config.get("RSP_max_num_periods"))
    continue_steps = parse_number(continue_config.get("RSP_target_steps_per_cycle"))
    deep_periods = parse_number(deep_config.get("RSP_max_num_periods"))
    deep_profile_interval = parse_number(deep_config.get("profile_interval"))
    deep_history_interval = parse_number(deep_config.get("history_interval"))

    continue_config_ok = (
        continue_max is not None
        and continue_max >= EXPECTED_CONTINUE_MAX_PERIODS
        and continue_steps is not None
        and continue_steps >= EXPECTED_TARGET_STEPS
    )
    deep_config_ok = (
        deep_periods == 2
        and deep_profile_interval == 1
        and deep_history_interval == 1
    )
    product_gate_ok = bool(post_action) and (
        (converged_exact and post_action.get("action") != "waiting_for_convergence")
        or ((not converged_exact) and post_action.get("action") == "waiting_for_convergence")
    )

    return {
        "model_id": model_id,
        "run_name": record.get("run_name"),
        "active_stage": active_stage,
        "latest_period": live.get("latest_period"),
        "latest_history_model": live.get("latest_history_model"),
        "latest_surface_velocity_status": live.get("latest_surface_velocity_status"),
        "latest_max_vsurf_div_cs": live.get("latest_max_vsurf_div_cs"),
        "strict_tolerance": tolerance,
        "converged_exact": converged_exact,
        "gamma_peak_to_peak_last_window": gamma,
        "period_fractional_peak_to_peak_last_window": period,
        "delta_r_fractional_peak_to_peak_last_window": delta_r,
        "gamma_ratio_to_tolerance": ratio(gamma, tolerance),
        "period_ratio_to_tolerance": ratio(period, tolerance),
        "delta_r_ratio_to_tolerance": ratio(delta_r, tolerance),
        "blocking_metrics": ",".join(blocking),
        "forecast_status": forecast.get("forecast_status"),
        "forecast_blocking_metrics": forecast.get("blocking_metrics"),
        "delta_r_periods_to_tolerance_linear": forecast.get("delta_r_periods_to_tolerance_linear"),
        "post_convergence_action": post_action.get("action"),
        "post_convergence_product_gate_ok": product_gate_ok,
        "active_inlist": active_inlist,
        "active_inlist_config": active_config,
        "continue_inlist_config": continue_config,
        "deep_inlist_config": deep_config,
        "continue_config_ok": continue_config_ok,
        "deep_config_ok": deep_config_ok,
        "configuration_ok": continue_config_ok and deep_config_ok,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    columns = [
        "model_id",
        "active_stage",
        "latest_period",
        "latest_history_model",
        "latest_surface_velocity_status",
        "converged_exact",
        "gamma_ratio_to_tolerance",
        "period_ratio_to_tolerance",
        "delta_r_ratio_to_tolerance",
        "blocking_metrics",
        "forecast_status",
        "delta_r_periods_to_tolerance_linear",
        "post_convergence_action",
        "post_convergence_product_gate_ok",
        "configuration_ok",
        "continue_config_ok",
        "deep_config_ok",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def main() -> int:
    args = parse_args()
    workspace = args.workspace
    output_dir = workspace / "output"

    manifest = manifest_by_model(workspace)
    live_by_id = rows_by_model(read_json(output_dir / "live_status.json"))
    convergence_by_id = rows_by_model(read_json(output_dir / "convergence_summary_last100.json"))
    forecast_by_id = rows_by_model(read_json(output_dir / "convergence_forecast_last100.json"))
    post_by_id = post_actions_by_model(read_json(output_dir / "post_convergence_products_status.json"))

    rows = []
    for model_id in args.models:
        record = manifest.get(model_id, {})
        rows.append(
            summarize_model(
                model_id,
                record,
                live_by_id.get(model_id, {}),
                convergence_by_id.get(model_id, {}),
                forecast_by_id.get(model_id, {}),
                post_by_id.get(model_id, {}),
                float(args.tolerance),
            )
        )

    summary = {
        "generated_at": now_iso(),
        "workspace": str(workspace),
        "tolerance": float(args.tolerance),
        "criterion": (
            "converged_exact requires Gamma, period fractional peak-to-peak, and DeltaR fractional "
            "peak-to-peak to each be <= tolerance over the final 100 cycles"
        ),
        "expected_continue_max_num_periods": EXPECTED_CONTINUE_MAX_PERIODS,
        "expected_target_steps_per_cycle": EXPECTED_TARGET_STEPS,
        "models": rows,
    }

    json_path = output_dir / "convergence_gate_audit.json"
    csv_path = output_dir / "convergence_gate_audit.csv"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(csv_path, rows)
    print(json_path)
    print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
