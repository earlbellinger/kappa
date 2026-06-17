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


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy Kappa when meaningful RSP batch outputs change.")
    parser.add_argument("--rre-root", type=Path, default=DEFAULT_RRE_ROOT)
    parser.add_argument("--kappa-root", type=Path, default=DEFAULT_KAPPA_ROOT)
    parser.add_argument("--pages-root", type=Path, default=DEFAULT_PAGES_ROOT)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable))
    parser.add_argument("--interval-seconds", type=int, default=900)
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


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def refresh_in_progress(rre_root: Path, log_path: Path) -> bool:
    marker = rre_root / "rsp_batch_runs" / "output" / REFRESH_MARKER_NAME
    if not marker.exists():
        return False
    age = time.time() - marker.stat().st_mtime
    if age <= REFRESH_MARKER_STALE_SECONDS:
        append_log(log_path, f"[{now_iso()}] local refresh in progress; deferring deploy")
        return True
    append_log(log_path, f"[{now_iso()}] ignoring stale refresh marker age={age:.0f}s")
    try:
        marker.unlink()
    except OSError as exc:
        append_log(log_path, f"[{now_iso()}] could not remove stale refresh marker: {exc!r}")
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
    live_models = []
    if isinstance(live, dict):
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
    return {
        "batch_status": live.get("batch_status", {}).get("status") if isinstance(live, dict) else None,
        "completed_gif_count": live.get("completed_gif_count") if isinstance(live, dict) else None,
        "verified_model_count": live.get("verified_model_count") if isinstance(live, dict) else None,
        "audit_complete": audit.get("complete") if isinstance(audit, dict) else None,
        "audit_status": audit.get("status") if isinstance(audit, dict) else None,
        "quality_status": quality.get("status") if isinstance(quality, dict) else None,
        "quality_complete": quality.get("complete") if isinstance(quality, dict) else None,
        "convergence_models": convergence_models,
        "convergence_trend_models": convergence_trend_models,
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
        ["git", "pull", "--rebase", "origin", branch],
        ["git", "push", "origin", branch],
    ):
        completed = run_command(command, repo, log_path)
        if completed.returncode != 0:
            raise RuntimeError(f"Command failed in {repo}: {' '.join(command)}")


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
    copy_site_to_pages(kappa_root, pages_root)
    commit_and_push(pages_root, "Refresh Kappa batch status", log_path, "master")


def main() -> int:
    args = parse_args()
    output_dir = args.rre_root.resolve() / "rsp_batch_runs" / "output"
    state_path = (args.state_file or output_dir / "kappa_deploy_watch_state.json").resolve()
    log_path = (args.log or output_dir / "kappa_deploy_watch.log").resolve()
    append_log(log_path, f"[{now_iso()}] Kappa deploy watcher started")

    while True:
        if refresh_in_progress(args.rre_root.resolve(), log_path):
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
