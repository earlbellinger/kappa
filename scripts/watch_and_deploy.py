from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_PYTHON = Path(r"C:\Program Files\Python311\python.exe")
DEFAULT_RRE_ROOT = Path(r"C:\Users\earlb\Downloads\rre")
DEFAULT_KAPPA_ROOT = Path(r"C:\Users\earlb\Downloads\kappa")
DEFAULT_PAGES_ROOT = Path(r"C:\Users\earlb\Downloads\earlbellinger.github.io")
REFRESH_MARKER_NAME = "gallery_refresh_in_progress.json"
REFRESH_MARKER_STALE_SECONDS = 1800
ANIMATION_SUFFIX = "_work_r_over_R_phase_cycle_dark_main_terms_gas_heating_pav_work"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy Kappa when meaningful RSP batch outputs change.")
    parser.add_argument("--rre-root", type=Path, default=DEFAULT_RRE_ROOT)
    parser.add_argument("--kappa-root", type=Path, default=DEFAULT_KAPPA_ROOT)
    parser.add_argument("--pages-root", type=Path, default=DEFAULT_PAGES_ROOT)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable))
    parser.add_argument("--interval-seconds", type=int, default=900)
    parser.add_argument("--refresh-wait-seconds", type=int, default=240)
    parser.add_argument("--state-file", type=Path, default=None)
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--deploy-initial", action="store_true")
    return parser.parse_args()


def append_log(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip() + "\n")


def load_json(path: Path) -> object | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def stable_animation_summary(summary: dict[str, object]) -> dict[str, object]:
    pdv_generation = summary.get("pdv_generation", {})
    if not isinstance(pdv_generation, dict):
        pdv_generation = {}
    opacity_scaling = summary.get("opacity_scaling", {})
    if not isinstance(opacity_scaling, dict):
        opacity_scaling = {}
    panel_y_ranges = summary.get("panel_y_ranges", {})
    if not isinstance(panel_y_ranges, dict):
        panel_y_ranges = {}
    left_power_panel = panel_y_ranges.get("left_power", {})
    if not isinstance(left_power_panel, dict):
        left_power_panel = {}
    scaled_diagnostic_bounds = summary.get("scaled_diagnostic_bounds", {})
    if not isinstance(scaled_diagnostic_bounds, dict):
        scaled_diagnostic_bounds = {}

    return {
        "scaling_method_version": summary.get("scaling_method_version"),
        "coordinate": summary.get("coordinate"),
        "pressure_work_mode": summary.get("pressure_work_mode") or pdv_generation.get("pressure_work_mode"),
        "heating_mode": summary.get("heating_mode"),
        "main_terms_only": summary.get("main_terms_only"),
        "phase_panel_repeat_version": summary.get("phase_panel_repeat_version"),
        "phase_panel_repeat_mode": summary.get("phase_panel_repeat_mode"),
        "x_limits": summary.get("x_limits"),
        "left_power_ylim": summary.get("left_power_ylim"),
        "left_power_visible_data_bounds": summary.get("left_power_visible_data_bounds"),
        "left_power_panel_limits": left_power_panel.get("limits"),
        "left_power_panel_visible_data_bounds": left_power_panel.get("visible_data_bounds"),
        "scaled_opacity_visible_data_bounds": scaled_diagnostic_bounds.get("opacity"),
        "opacity_scaling": {
            "method": opacity_scaling.get("method"),
            "panel_bottom_fraction": opacity_scaling.get("panel_bottom_fraction"),
            "panel_top_fraction": opacity_scaling.get("panel_top_fraction"),
            "visible_data_bounds": opacity_scaling.get("visible_data_bounds"),
            "reference_visible_data_bounds": opacity_scaling.get("reference_visible_data_bounds"),
            "effective_data_bounds": opacity_scaling.get("effective_data_bounds"),
            "display_units_per_opacity_unit": opacity_scaling.get("display_units_per_opacity_unit"),
            "opacity_min_baseline": opacity_scaling.get("opacity_min_baseline"),
            "opacity_max_display_value": opacity_scaling.get("opacity_max_display_value"),
            "scaled_visible_display_bounds": opacity_scaling.get("scaled_visible_display_bounds"),
            "scaled_effective_display_bounds": opacity_scaling.get("scaled_effective_display_bounds"),
        },
    }


def animation_artifacts_signature(rre_root: Path) -> list[dict[str, object]]:
    manifest = load_json(rre_root / "rsp_batch_runs" / "inputs" / "manifest.json")
    if not isinstance(manifest, list):
        return []

    artifacts: list[dict[str, object]] = []
    for model in manifest:
        if not isinstance(model, dict) or not model.get("model_id"):
            continue
        output_dir_value = model.get("output_dir")
        output_dir = Path(str(output_dir_value)) if output_dir_value else None
        summary_path: Path | None = None
        gif_path: Path | None = None
        if output_dir is not None and output_dir.exists():
            summaries = sorted(output_dir.glob(f"*{ANIMATION_SUFFIX}_summary.json"))
            gifs = sorted(output_dir.glob(f"*{ANIMATION_SUFFIX}.gif"))
            summary_path = summaries[-1] if summaries else None
            gif_path = gifs[-1] if gifs else None

        summary = load_json(summary_path) if summary_path is not None else None
        artifact: dict[str, object] = {
            "model_id": model.get("model_id"),
            "summary_exists": isinstance(summary, dict),
            "gif_exists": gif_path is not None and gif_path.exists(),
            "gif_size_bytes": gif_path.stat().st_size if gif_path is not None and gif_path.exists() else None,
            "gif_mtime_bucket_60s": (
                int(gif_path.stat().st_mtime // 60) if gif_path is not None and gif_path.exists() else None
            ),
        }
        if isinstance(summary, dict):
            artifact.update(stable_animation_summary(summary))
        artifacts.append(artifact)
    return artifacts


def cycle_modulation_signature(rre_root: Path) -> dict[str, object]:
    output_dir = rre_root / "rsp_batch_runs" / "output"
    cycle_path = output_dir / "cycle_modulation_summary.json"
    data = load_json(cycle_path)
    rows = data.get("models", []) if isinstance(data, dict) else []
    models = []
    for row in rows:
        if not isinstance(row, dict) or not row.get("model_id"):
            continue
        diagnostic_png = Path(str(row["diagnostic_png"])) if row.get("diagnostic_png") else None
        models.append(
            {
                "model_id": row.get("model_id"),
                "history_source": row.get("history_source"),
                "cycle_count": row.get("cycle_count"),
                "last_cycle_count_used": row.get("last_cycle_count_used"),
                "max_l_modulation_fraction": row.get("max_l_modulation_fraction"),
                "min_v_modulation_mag": row.get("min_v_modulation_mag"),
                "period_modulation_fraction": row.get("period_modulation_fraction"),
                "radius_amplitude_modulation_fraction": row.get("radius_amplitude_modulation_fraction"),
                "history_candidate_cycle_counts": row.get("history_candidate_cycle_counts"),
                "diagnostic_png_exists": diagnostic_png.exists() if diagnostic_png is not None else False,
                "diagnostic_png_size_bytes": (
                    diagnostic_png.stat().st_size if diagnostic_png is not None and diagnostic_png.exists() else None
                ),
            }
        )
    overview = output_dir / "cycle_diagnostics" / "cycle_modulation_overview.png"
    return {
        "summary_exists": isinstance(data, dict),
        "generated_at": data.get("generated_at") if isinstance(data, dict) else None,
        "overview_png_exists": overview.exists(),
        "overview_png_size_bytes": overview.stat().st_size if overview.exists() else None,
        "models": models,
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def refresh_in_progress(rre_root: Path, log_path: Path, wait_seconds: int = 0) -> bool:
    marker = rre_root / "rsp_batch_runs" / "output" / REFRESH_MARKER_NAME
    if not marker.exists():
        return False
    deadline = time.time() + max(0, wait_seconds)
    while marker.exists():
        age = time.time() - marker.stat().st_mtime
        if age > REFRESH_MARKER_STALE_SECONDS:
            append_log(log_path, f"[{now_iso()}] ignoring stale refresh marker age={age:.0f}s")
            try:
                marker.unlink()
            except OSError as exc:
                append_log(log_path, f"[{now_iso()}] could not remove stale refresh marker: {exc!r}")
            return False
        if time.time() >= deadline:
            break
        remaining = int(max(0, deadline - time.time()))
        append_log(log_path, f"[{now_iso()}] local refresh in progress; waiting up to {remaining}s")
        time.sleep(min(15, max(1, remaining)))
    if not marker.exists():
        append_log(log_path, f"[{now_iso()}] local refresh completed; continuing deploy check")
        return False
    age = time.time() - marker.stat().st_mtime
    if age <= REFRESH_MARKER_STALE_SECONDS:
        append_log(log_path, f"[{now_iso()}] local refresh in progress; deferring deploy")
        return True
    return False


def run_command(command: list[str], cwd: Path, log_path: Path) -> subprocess.CompletedProcess[str]:
    append_log(log_path, f"[{now_iso()}] $ {' '.join(command)}")
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    append_log(log_path, f"[{now_iso()}] returncode={completed.returncode}")
    if completed.stdout.strip():
        append_log(log_path, completed.stdout.strip())
    if completed.stderr.strip():
        append_log(log_path, completed.stderr.strip())
    return completed


def status_signature(rre_root: Path) -> dict[str, object]:
    output_dir = rre_root / "rsp_batch_runs" / "output"
    live = load_json(output_dir / "live_status.json")
    audit = load_json(output_dir / "batch_audit_summary.json")
    quality = load_json(output_dir / "quality_extension_status.json")
    convergence = load_json(output_dir / "convergence_summary_last100.json")
    convergence_trends = load_json(output_dir / "convergence_trends_last100.json")
    phase_seam = load_json(output_dir / "phase_seam_audit.json")
    cycle_boundary = load_json(output_dir / "cycle_boundary_audit.json")
    growth_diagnostics = []
    for path in sorted(output_dir.glob("*_growth_diagnostic.json")):
        data = load_json(path)
        png_path = output_dir / path.name.replace(".json", ".png")
        if not isinstance(data, dict):
            continue
        outlook = data.get("growth_outlook", {})
        growth_diagnostics.append(
            {
                "name": path.name,
                "png_exists": png_path.exists(),
                "cycle_count": data.get("cycle_count"),
                "latest_period": data.get("latest_period"),
                "max_vsurf_div_cs_latest": (
                    outlook.get("max_vsurf_div_cs_latest") if isinstance(outlook, dict) else None
                ),
                "delta_r_criterion_factor": (
                    outlook.get("delta_r_criterion_factor") if isinstance(outlook, dict) else None
                ),
                "cycles_to_vsurf_div_cs_0p8": (
                    outlook.get("cycles_to_vsurf_div_cs_0p8") if isinstance(outlook, dict) else None
                ),
            }
        )
    live_models = []
    batch_results = []
    if isinstance(live, dict):
        batch_status = live.get("batch_status")
        if isinstance(batch_status, dict):
            results = batch_status.get("results")
            if isinstance(results, dict):
                for model_id, result in sorted(results.items()):
                    if not isinstance(result, dict):
                        continue
                    batch_results.append(
                        {
                            "model_id": model_id,
                            "status": result.get("status"),
                            "driver_pid_process_running": result.get("driver_pid_process_running"),
                            "driver_pid_source": result.get("driver_pid_source"),
                            "raw_driver_pid_process_running": result.get("raw_driver_pid_process_running"),
                        }
                    )
        for model in live.get("models", []):
            if not isinstance(model, dict):
                continue
            try:
                period = int(float(str(model.get("latest_period"))))
            except (TypeError, ValueError):
                period = None
            try:
                history_model = int(float(str(model.get("latest_history_model"))))
            except (TypeError, ValueError):
                history_model = None
            try:
                max_vsurf = float(str(model.get("latest_max_vsurf_div_cs")))
            except (TypeError, ValueError):
                max_vsurf = None
            active_stage = model.get("active_stage")
            live_models.append(
                {
                    "model_id": model.get("model_id"),
                    "active_stage": active_stage,
                    "period_progress_bucket_50": period // 50 if period is not None else None,
                    "history_model_bucket_1000": (
                        history_model // 1000
                        if history_model is not None and active_stage == "create"
                        else None
                    ),
                    "max_vsurf_div_cs_bucket_0p02": (
                        int(max_vsurf / 0.02) if max_vsurf is not None else None
                    ),
                    "latest_surface_velocity_status": model.get("latest_surface_velocity_status"),
                    "gif_exists": bool(model.get("gif_exists")),
                    "trusted_animation": bool(model.get("trusted_animation")),
                    "trusted_animation_reason": model.get("trusted_animation_reason"),
                    "diagnostic_animation_reason": model.get("diagnostic_animation_reason"),
                    "verification_passed": bool(model.get("verification_passed")),
                    "stages": model.get("stages", {}),
                }
            )
    convergence_models = []
    if isinstance(convergence, dict):
        for model in convergence.get("models", []):
            if not isinstance(model, dict):
                continue
            convergence_models.append(
                {
                    "model_id": model.get("model_id"),
                    "source_kind": model.get("source_kind"),
                    "cycle_count": model.get("cycle_count"),
                    "last_period_number": model.get("last_period_number"),
                    "has_full_window": model.get("has_full_window"),
                    "converged_exact": model.get("converged_exact"),
                    "max_vsurf_div_cs_max_last_window": model.get("max_vsurf_div_cs_max_last_window"),
                }
            )
    convergence_trend_models = []
    if isinstance(convergence_trends, dict):
        latest_by_model = convergence_trends.get("latest_by_model", {})
        if isinstance(latest_by_model, dict):
            for model_id, model in latest_by_model.items():
                if not isinstance(model, dict):
                    continue
                convergence_trend_models.append(
                    {
                        "model_id": model_id,
                        "source_kind": model.get("source_kind"),
                        "window_end_period": model.get("window_end_period"),
                        "converged_exact": model.get("converged_exact"),
                        "max_vsurf_div_cs_max": model.get("max_vsurf_div_cs_max"),
                    }
                )
    phase_seam_models = []
    if isinstance(phase_seam, dict):
        for row in phase_seam.get("rows", []):
            if not isinstance(row, dict) or not row.get("model_id"):
                continue
            phase_seam_models.append(
                {
                    "model_id": row.get("model_id"),
                    "phase_seam_ok": row.get("phase_seam_ok"),
                    "worst_seam_metric": row.get("worst_seam_metric"),
                    "worst_seam_fraction": row.get("worst_seam_fraction"),
                    "verification_passed": row.get("verification_passed"),
                    "trusted_animation": row.get("trusted_animation"),
                    "trusted_animation_reason": row.get("trusted_animation_reason"),
                }
            )
    cycle_boundary_models = []
    if isinstance(cycle_boundary, dict):
        for row in cycle_boundary.get("rows", []):
            if not isinstance(row, dict) or not row.get("model_id"):
                continue
            cycle_boundary_models.append(
                {
                    "model_id": row.get("model_id"),
                    "period_days": row.get("period_days"),
                    "boundary_luminosity_lsun_seam_fraction": row.get(
                        "boundary_luminosity_lsun_seam_fraction"
                    ),
                    "boundary_radius_rsun_seam_fraction": row.get("boundary_radius_rsun_seam_fraction"),
                    "boundary_teff_k_seam_fraction": row.get("boundary_teff_k_seam_fraction"),
                    "latest_radius_maximum_radius_change_rsun": row.get(
                        "latest_radius_maximum_radius_change_rsun"
                    ),
                }
            )
    return {
        "batch_status": live.get("batch_status", {}).get("status") if isinstance(live, dict) else None,
        "completed_gif_count": live.get("completed_gif_count") if isinstance(live, dict) else None,
        "verified_model_count": live.get("verified_model_count") if isinstance(live, dict) else None,
        "audit_complete": audit.get("complete") if isinstance(audit, dict) else None,
        "audit_status": audit.get("status") if isinstance(audit, dict) else None,
        "quality_status": quality.get("status") if isinstance(quality, dict) else None,
        "quality_complete": quality.get("complete") if isinstance(quality, dict) else None,
        "batch_results": batch_results,
        "convergence_models": convergence_models,
        "convergence_trend_models": convergence_trend_models,
        "phase_seam": {
            "row_count": phase_seam.get("row_count") if isinstance(phase_seam, dict) else None,
            "failed_phase_seam_count": (
                phase_seam.get("failed_phase_seam_count") if isinstance(phase_seam, dict) else None
            ),
            "untrusted_animation_count": (
                phase_seam.get("untrusted_animation_count") if isinstance(phase_seam, dict) else None
            ),
            "models": phase_seam_models,
        },
        "cycle_boundary": {
            "row_count": cycle_boundary.get("row_count") if isinstance(cycle_boundary, dict) else None,
            "models": cycle_boundary_models,
        },
        "growth_diagnostics": growth_diagnostics,
        "animation_artifacts": animation_artifacts_signature(rre_root),
        "cycle_modulation": cycle_modulation_signature(rre_root),
        "models": live_models,
    }


def git_dirty(repo: Path) -> bool:
    completed = subprocess.run(["git", "status", "--short"], cwd=repo, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"git status failed in {repo}")
    return bool(completed.stdout.strip())


def commit_and_push(repo: Path, message: str, log_path: Path, branch: str) -> None:
    if not git_dirty(repo):
        append_log(log_path, f"[{now_iso()}] {repo}: no changes to commit")
        return
    for command in (
        ["git", "add", "."],
        ["git", "commit", "-m", message],
        ["git", "pull", "--rebase", "-X", "theirs", "origin", branch],
        ["git", "push", "origin", branch],
    ):
        completed = run_command(command, repo, log_path)
        if completed.returncode != 0:
            raise RuntimeError(f"Command failed in {repo}: {' '.join(command)}")


def update_branch(repo: Path, log_path: Path, branch: str) -> None:
    completed = run_command(["git", "pull", "--rebase", "-X", "theirs", "origin", branch], repo, log_path)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed in {repo}: git pull --rebase -X theirs origin {branch}")


def copy_site_to_pages(kappa_root: Path, pages_root: Path) -> None:
    source = (kappa_root / "site").resolve()
    pages = pages_root.resolve()
    target = (pages / "apps" / "kappa").resolve()
    if not source.exists():
        raise RuntimeError(f"Missing Kappa site source: {source}")
    if not pages.exists():
        raise RuntimeError(f"Missing Pages repository: {pages}")
    if pages not in target.parents:
        raise RuntimeError(f"Refusing to delete target outside Pages root: {target}")
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        destination = target / child.name
        if child.is_dir():
            shutil.copytree(child, destination)
        else:
            shutil.copy2(child, destination)


def deploy(args: argparse.Namespace, log_path: Path) -> None:
    kappa_root = args.kappa_root.resolve()
    pages_root = args.pages_root.resolve()
    build_script = kappa_root / "scripts" / "build_static_site.py"
    completed = run_command(
        [
            str(args.python),
            str(build_script),
            "--rre-root",
            str(args.rre_root.resolve()),
            "--output",
            str(kappa_root / "site"),
        ],
        kappa_root,
        log_path,
    )
    if completed.returncode != 0:
        raise RuntimeError("Kappa site build failed")

    leak_check = run_command(
        ["rg", "-n", r"file:///|C:\\|Users/earlb|Users\\earlb", str(kappa_root / "site")],
        kappa_root,
        log_path,
    )
    if leak_check.returncode not in (0, 1):
        raise RuntimeError("Path leak check failed")
    if leak_check.returncode == 0:
        raise RuntimeError("Path leak check found local paths in Kappa site")

    commit_and_push(kappa_root, "Refresh batch status snapshot", log_path, "main")
    update_branch(pages_root, log_path, "master")
    copy_site_to_pages(kappa_root, pages_root)
    commit_and_push(pages_root, "Refresh Kappa batch status", log_path, "master")


def main() -> int:
    args = parse_args()
    output_dir = args.rre_root.resolve() / "rsp_batch_runs" / "output"
    state_path = (args.state_file or output_dir / "kappa_deploy_watch_state.json").resolve()
    log_path = (args.log or output_dir / "kappa_deploy_watch.log").resolve()
    append_log(log_path, f"[{now_iso()}] Kappa deploy watcher started")

    while True:
        if refresh_in_progress(args.rre_root.resolve(), log_path, int(args.refresh_wait_seconds)):
            time.sleep(max(60, int(args.interval_seconds)))
            continue
        state = load_json(state_path)
        previous_signature = state.get("signature") if isinstance(state, dict) else None
        signature = status_signature(args.rre_root.resolve())
        should_deploy = bool(args.deploy_initial) if previous_signature is None else signature != previous_signature
        if should_deploy:
            append_log(log_path, f"[{now_iso()}] meaningful batch change detected")
            try:
                deploy(args, log_path)
                write_json(
                    state_path,
                    {
                        "signature": signature,
                        "last_deployed_at": now_iso(),
                    },
                )
            except Exception as exc:
                append_log(log_path, f"[{now_iso()}] deploy failed: {exc!r}")
        elif previous_signature is None:
            write_json(
                state_path,
                {
                    "signature": signature,
                    "last_deployed_at": None,
                    "initialized_at": now_iso(),
                },
            )
            append_log(log_path, f"[{now_iso()}] initialized state without deploying")
        else:
            append_log(log_path, f"[{now_iso()}] no meaningful batch change")

        time.sleep(max(60, int(args.interval_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
