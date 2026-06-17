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
    live_models = []
    if isinstance(live, dict):
        for model in live.get("models", []):
            if not isinstance(model, dict):
                continue
            try:
                period = int(float(str(model.get("latest_period"))))
            except (TypeError, ValueError):
                period = None
            live_models.append(
                {
                    "model_id": model.get("model_id"),
                    "active_stage": model.get("active_stage"),
                    "period_progress_bucket_50": period // 50 if period is not None else None,
                    "gif_exists": bool(model.get("gif_exists")),
                    "verification_passed": bool(model.get("verification_passed")),
                    "stages": model.get("stages", {}),
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
