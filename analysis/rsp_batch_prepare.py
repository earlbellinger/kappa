from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import zipfile
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_ZIP = ROOT / "inlists.zip"
DEFAULT_WORKSPACE = ROOT / "rsp_batch_runs"
DEFAULT_TEMPLATE_RUN = ROOT / "mesa_rsp_combined_14507_full_pressure_minus_eq"
DEFAULT_TEMPLATE_OUTPUT = ROOT / "output" / "combined_14507_full_pressure_minus_eq"

CREATE_TEMPLATE = "inlist_combined_14507_create"
CONTINUE_TEMPLATE = "inlist_combined_14507_continue_saturation"
RESTART_TEMPLATE = "inlist_combined_14507_restart"
DEEP_TEMPLATE = "inlist_combined_14507_deep2cycles"

STAGE_TEMPLATES = {
    "create": CREATE_TEMPLATE,
    "continue_saturation": CONTINUE_TEMPLATE,
    "restart": RESTART_TEMPLATE,
    "deep2cycles": DEEP_TEMPLATE,
}

GENERATED_INLISTS = {
    "create": "inlist_create",
    "continue_saturation": "inlist_continue_saturation",
    "restart": "inlist_restart",
    "deep2cycles": "inlist_deep2cycles",
}

STELLAR_KEYS = ("RSP_mass", "RSP_Teff", "RSP_L", "RSP_X", "RSP_Z")
CONVECTION_KEYS = (
    "RSP_alfa",
    "RSP_alfac",
    "RSP_alfas",
    "RSP_alfad",
    "RSP_alfam",
    "RSP_gammar",
    "RSP_alfap",
    "RSP_alfat",
)
PARAMETER_KEYS = ("Zbase", *STELLAR_KEYS, *CONVECTION_KEYS)
ALLOWED_DIFF_KEYS = {
    *PARAMETER_KEYS,
    "save_model_filename",
    "load_model_filename",
    "history_columns_file",
    "profile_columns_file",
    "log_directory",
    "photo_directory",
}

REQUIRED_DEEP_COLUMNS = (
    "zone",
    "q",
    "dq",
    "radius",
    "vel_km_per_s",
    "luminosity",
    "rsp_Lc",
    "rsp_Lc_div_L",
    "rsp_Lr",
    "rsp_Lt",
    "rsp_Lt_div_L",
    "rsp_Pt",
    "rsp_Pvsc",
    "rsp_Eq",
    "rsp_damp",
    "rsp_dampR",
    "rsp_src",
    "rsp_src_snk",
    "logT",
    "logRho",
    "logP",
    "pressure",
    "x_mass_fraction_H",
    "y_mass_fraction_He",
    "z_mass_fraction_metals",
    "opacity",
    "tau",
    "cp",
    "mu",
    "grada",
    "gradr",
    "gamma1",
    "typical_charge h1",
    "ionization h1",
    "typical_charge he4",
    "ionization he4",
    "rsp_Et",
    "rsp_Chi",
)
CONVERGENCE_HISTORY_COLUMNS = (
    "rsp_GREKM",
    "rsp_DeltaR",
    "rsp_num_periods",
    "rsp_period_in_days",
)

ASSIGNMENT_RE = re.compile(
    r"^(?P<indent>\s*)(?P<key>[A-Za-z][A-Za-z0-9_]*(?:\(\d+\))?)\s*=\s*(?P<value>.*?)(?P<comment>\s*!.*)?$"
)


@dataclass
class ModelRecord:
    model_id: str
    model_index: int
    source_inlist: str
    registered_existing: bool
    run_name: str
    run_dir: str
    output_dir: str
    prefix: str
    product_stem: str
    RSP_mass: str
    RSP_Teff: str
    RSP_L: str
    RSP_X: str
    RSP_Z: str
    Zbase: str
    RSP_alfa: str
    RSP_alfac: str
    RSP_alfas: str
    RSP_alfad: str
    RSP_alfam: str
    RSP_gammar: str
    RSP_alfap: str
    RSP_alfat: str
    create_model: str
    saturated_model: str
    restart_model: str
    deep_model: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare congruent batch RSP workdirs from parameter-only inlists."
    )
    parser.add_argument("--inlists-zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--template-run", type=Path, default=DEFAULT_TEMPLATE_RUN)
    parser.add_argument("--template-output", type=Path, default=DEFAULT_TEMPLATE_OUTPUT)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recreate generated run directories. Existing LOGS/photos/model files inside them are removed.",
    )
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_assignments(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        match = ASSIGNMENT_RE.match(line)
        if match:
            values[match.group("key")] = clean_value(match.group("value"))
    return values


def clean_value(value: str) -> str:
    return value.strip().rstrip(",").strip()


def extract_parameter_seed(text: str) -> dict[str, str]:
    assignments = parse_assignments(text)
    missing = [key for key in (*STELLAR_KEYS, *CONVECTION_KEYS) if key not in assignments]
    if missing:
        raise ValueError(f"Missing required parameter(s): {', '.join(missing)}")
    params = {key: assignments[key] for key in (*STELLAR_KEYS, *CONVECTION_KEYS)}
    params["Zbase"] = params["RSP_Z"]
    return params


def sanitize_number_for_name(value: str, decimals: int | None = None) -> str:
    raw = value.strip().strip("'\"").replace("d", "e").replace("D", "e")
    if decimals is None and "e" not in raw.lower():
        return raw.replace("-", "m").replace("+", "").replace(".", "p")
    try:
        number = float(raw)
        if decimals is None:
            text = f"{number:g}"
        else:
            text = f"{number:.{decimals}f}".rstrip("0").rstrip(".")
    except ValueError:
        text = raw
    return text.replace("-", "m").replace("+", "").replace(".", "p").replace("e", "e")


def model_run_name(model_index: int, params: dict[str, str]) -> str:
    mass = sanitize_number_for_name(params["RSP_mass"], decimals=4)
    z = sanitize_number_for_name(params["RSP_Z"])
    teff = sanitize_number_for_name(params["RSP_Teff"], decimals=0)
    return f"model_{model_index:03d}_M{mass}_Z{z}_T{teff}"


def replace_assignments(text: str, replacements: dict[str, str]) -> str:
    seen: set[str] = set()
    lines: list[str] = []
    for line in text.splitlines():
        match = ASSIGNMENT_RE.match(line)
        if match and match.group("key") in replacements:
            key = match.group("key")
            lines.append(f"{match.group('indent')}{key} = {replacements[key]}")
            seen.add(key)
        else:
            lines.append(line)
    missing = sorted(set(replacements) - seen)
    if missing:
        raise ValueError(f"Could not replace missing assignment(s): {', '.join(missing)}")
    return "\n".join(lines) + "\n"


def template_ignore(_dir: str, names: list[str]) -> set[str]:
    stale_run_scripts = {
        "rn",
        "rn_continue_saturation",
        "rn_continue_to_saturation",
        "rn_create",
        "rn_deep_two_cycles",
        "rn_from_final_model",
        "rn_restart",
    }
    ignored: set[str] = set()
    for name in names:
        lower = name.lower()
        if name in {".mesa_temp_cache", "__pycache__", "inlist"} | stale_run_scripts:
            ignored.add(name)
        elif name.startswith("LOGS") or name.startswith("photos"):
            ignored.add(name)
        elif name.startswith("final_") and lower.endswith(".mod"):
            ignored.add(name)
        elif name.startswith("equilibrium_") and lower.endswith(".mod"):
            ignored.add(name)
        elif name.startswith("inlist_combined_14507"):
            ignored.add(name)
    return ignored


def copy_template_workdir(template_run: Path, run_dir: Path, force: bool) -> None:
    if force and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(template_run, run_dir, dirs_exist_ok=True, ignore=template_ignore)


def write_executable(path: Path, text: str) -> None:
    path.write_text(text, newline="\n")
    try:
        path.chmod(path.stat().st_mode | 0o111)
    except OSError:
        pass


def generate_run_scripts(run_dir: Path) -> None:
    scripts = {
        "rn_create": ("inlist_create", "create run"),
        "rn_continue_saturation": ("inlist_continue_saturation", "continuation-to-saturation run"),
        "rn_restart": ("inlist_restart", "restart-from-saturated-model run"),
        "rn_deep_two_cycles": ("inlist_deep2cycles", "deep two-cycle run"),
    }
    for script_name, (inlist_name, label) in scripts.items():
        write_executable(
            run_dir / script_name,
            "#!/bin/bash\n\n"
            f"cp -f {inlist_name} inlist || exit 1\n"
            "rm -f restart_photo\n\n"
            "date \"+DATE: %Y-%m-%d%nTIME: %H:%M:%S\"\n"
            "./star\n"
            "status=$?\n"
            "date \"+DATE: %Y-%m-%d%nTIME: %H:%M:%S\"\n"
            f"echo 'finished {label} with status '${{status}}\n"
            "exit ${status}\n",
        )


def stage_replacements(stage: str, params: dict[str, str], record: ModelRecord) -> dict[str, str]:
    replacements = {key: params[key] for key in PARAMETER_KEYS}
    replacements["history_columns_file"] = "'history_columns_batch.list'"
    replacements["log_directory"] = {
        "create": "'LOGS_saturation'",
        "continue_saturation": "'LOGS_continue_saturation'",
        "restart": "'LOGS_restart'",
        "deep2cycles": "'LOGS'",
    }[stage]
    replacements["photo_directory"] = {
        "create": "'photos_saturation'",
        "continue_saturation": "'photos_continue_saturation'",
        "restart": "'photos_restart'",
        "deep2cycles": "'photos'",
    }[stage]
    replacements["save_model_filename"] = {
        "create": f"'{record.create_model}'",
        "continue_saturation": f"'{record.saturated_model}'",
        "restart": f"'{record.restart_model}'",
        "deep2cycles": f"'{record.deep_model}'",
    }[stage]
    if stage == "continue_saturation":
        replacements["load_model_filename"] = f"'{record.create_model}'"
    elif stage == "restart":
        replacements["load_model_filename"] = f"'{record.saturated_model}'"
    elif stage == "deep2cycles":
        replacements["load_model_filename"] = f"'{record.restart_model}'"
        replacements["profile_columns_file"] = "'profile_columns_batch_deep.list'"
    return replacements


def write_generated_inlists(template_run: Path, run_dir: Path, params: dict[str, str], record: ModelRecord) -> None:
    for stage, template_name in STAGE_TEMPLATES.items():
        template_text = (template_run / template_name).read_text()
        text = replace_assignments(template_text, stage_replacements(stage, params, record))
        (run_dir / GENERATED_INLISTS[stage]).write_text(text, newline="\n")


def write_column_lists(template_run: Path, run_dir: Path) -> None:
    write_history_columns(template_run / "history_columns_combined_14507.list", run_dir / "history_columns_batch.list")
    source_profile = template_run / "profile_columns_combined_14507_deep.list"
    shutil.copy2(source_profile, run_dir / "profile_columns_batch_deep.list")


def write_history_columns(source: Path, destination: Path) -> None:
    text = source.read_text()
    active_columns = {
        line.strip().split()[0]
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith(("!", "#"))
    }
    additions = [name for name in CONVERGENCE_HISTORY_COLUMNS if name not in active_columns]
    if additions:
        text = text.rstrip() + "\n\n# RSP limit-cycle convergence diagnostics\n" + "\n".join(additions) + "\n"
    destination.write_text(text, newline="\n")


def normalize_assignment_value(value: str) -> str:
    return re.sub(r"\s+", "", value.strip()).lower()


def validate_generated_inlists(template_run: Path, run_dir: Path) -> dict[str, object]:
    stage_reports: dict[str, object] = {}
    unexpected_count = 0
    for stage, template_name in STAGE_TEMPLATES.items():
        template_values = parse_assignments((template_run / template_name).read_text())
        generated_values = parse_assignments((run_dir / GENERATED_INLISTS[stage]).read_text())
        allowed: dict[str, dict[str, str]] = {}
        unexpected: dict[str, dict[str, str]] = {}
        all_keys = sorted(set(template_values) | set(generated_values))
        for key in all_keys:
            before = template_values.get(key)
            after = generated_values.get(key)
            if normalize_assignment_value(str(before)) == normalize_assignment_value(str(after)):
                continue
            target = allowed if key in ALLOWED_DIFF_KEYS else unexpected
            target[key] = {"template": str(before), "generated": str(after)}
        unexpected_count += len(unexpected)
        stage_reports[stage] = {
            "allowed_diffs": allowed,
            "unexpected_diffs": unexpected,
            "passed": not unexpected,
        }
    return {"passed": unexpected_count == 0, "stages": stage_reports}


def build_record(
    model_index: int,
    params: dict[str, str],
    source_inlist: str,
    workspace: Path,
    registered_existing: bool = False,
    existing_run_dir: Path | None = None,
    existing_output_dir: Path | None = None,
) -> ModelRecord:
    if registered_existing:
        run_name = "model_000_combined_14507_full_pressure_minus_eq"
        run_dir = existing_run_dir or DEFAULT_TEMPLATE_RUN
        output_dir = existing_output_dir or DEFAULT_TEMPLATE_OUTPUT
        prefix = "mesa_rsp_combined_14507_full_pressure_minus_eq"
        product_stem = f"{prefix}_work_r_over_R_phase_cycle_dark_main_terms_gas_heating_pav_work"
    else:
        run_name = model_run_name(model_index, params)
        run_dir = workspace / "runs" / run_name
        output_dir = workspace / "output" / run_name
        prefix = f"mesa_rsp_{run_name}"
        product_stem = f"{prefix}_work_r_over_R_phase_cycle_dark_main_terms_gas_heating_pav_work"

    stem = f"final_{run_name}"
    return ModelRecord(
        model_id=f"model_{model_index:03d}",
        model_index=model_index,
        source_inlist=source_inlist,
        registered_existing=registered_existing,
        run_name=run_name,
        run_dir=str(run_dir),
        output_dir=str(output_dir),
        prefix=prefix,
        product_stem=product_stem,
        RSP_mass=params["RSP_mass"],
        RSP_Teff=params["RSP_Teff"],
        RSP_L=params["RSP_L"],
        RSP_X=params["RSP_X"],
        RSP_Z=params["RSP_Z"],
        Zbase=params["Zbase"],
        RSP_alfa=params["RSP_alfa"],
        RSP_alfac=params["RSP_alfac"],
        RSP_alfas=params["RSP_alfas"],
        RSP_alfad=params["RSP_alfad"],
        RSP_alfam=params["RSP_alfam"],
        RSP_gammar=params["RSP_gammar"],
        RSP_alfap=params["RSP_alfap"],
        RSP_alfat=params["RSP_alfat"],
        create_model=f"{stem}_create.mod",
        saturated_model=f"{stem}_saturated.mod",
        restart_model=f"{stem}_restart.mod",
        deep_model=f"{stem}_deep2cycles.mod",
    )


def read_zip_inlists(zip_path: Path) -> list[tuple[str, str]]:
    with zipfile.ZipFile(zip_path) as archive:
        names = sorted(name for name in archive.namelist() if not name.endswith("/"))
        return [(name, archive.read(name).decode("utf-8", errors="replace")) for name in names]


def write_manifest(workspace: Path, records: list[ModelRecord]) -> None:
    inputs_dir = workspace / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    json_path = inputs_dir / "manifest.json"
    csv_path = inputs_dir / "manifest.csv"
    json_path.write_text(json.dumps([asdict(record) for record in records], indent=2) + "\n")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def write_status(output_dir: Path, record: ModelRecord, validation: dict[str, object] | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    status = {
        "model_id": record.model_id,
        "run_name": record.run_name,
        "registered_existing": record.registered_existing,
        "run_dir": record.run_dir,
        "output_dir": record.output_dir,
        "prefix": record.prefix,
        "product_stem": record.product_stem,
        "updated_at": now_iso(),
        "stages": {
            "prepared": {
                "status": "complete",
                "timestamp": now_iso(),
            }
        },
    }
    if record.registered_existing:
        gif = Path(record.output_dir) / f"{record.product_stem}.gif"
        status["stages"]["registered_existing"] = {
            "status": "complete",
            "gif_exists": gif.exists(),
            "timestamp": now_iso(),
        }
    if validation is not None:
        status["validation"] = validation
    (output_dir / "run_status.json").write_text(json.dumps(status, indent=2) + "\n")


def write_template_snapshot(workspace: Path, template_run: Path) -> None:
    template_dir = workspace / "template"
    template_dir.mkdir(parents=True, exist_ok=True)
    for name in STAGE_TEMPLATES.values():
        shutil.copy2(template_run / name, template_dir / name)
    write_history_columns(template_run / "history_columns_combined_14507.list", template_dir / "history_columns_batch.list")
    shutil.copy2(template_run / "profile_columns_combined_14507_deep.list", template_dir / "profile_columns_batch_deep.list")
    info = {
        "source_template_run": str(template_run),
        "created_at": now_iso(),
        "allowed_parameter_keys": list(PARAMETER_KEYS),
        "allowed_nonphysical_filename_keys": sorted(ALLOWED_DIFF_KEYS - set(PARAMETER_KEYS)),
        "required_deep_columns": list(REQUIRED_DEEP_COLUMNS),
    }
    (template_dir / "template_info.json").write_text(json.dumps(info, indent=2) + "\n")


def write_readme(workspace: Path) -> None:
    readme = f"""# Congruent Batch RSP Runs

Prepared by `rsp_batch_prepare.py`.

## Canonical recipe

The generated workdirs use the 14507 full-pressure-minus-Eq inlists as the physics and numerical template. Only the stellar parameters, convection parameters, model filenames, and log/photo directories are changed.

The canonical final-cycle extraction and animation command is handled by `rsp_batch_run.py`:

```powershell
& 'C:\\Program Files\\Python311\\python.exe' C:\\Users\\earlb\\Downloads\\rre\\rsp_batch_run.py --workspace {workspace} --model model_001 --stage all
```

The plotting stage uses:

```text
--coordinate r_over_R --dark-mode --main-terms-only --pressure-work-mode gas_plus_pav --heating-mode gas_minus_c --fps 35
```

`model_000` is registered from the existing completed 14507 run and is not rerun by default.
"""
    (workspace / "README.md").write_text(readme, newline="\n")


def write_validation_reports(workspace: Path, reports: dict[str, object]) -> None:
    report_dir = workspace / "inputs"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "validation_report.json").write_text(json.dumps(reports, indent=2) + "\n")
    lines = ["Congruent inlist validation report", ""]
    for model_id, report in reports.items():
        passed = bool(report.get("passed"))
        lines.append(f"{model_id}: {'PASS' if passed else 'FAIL'}")
        for stage, stage_report in dict(report.get("stages", {})).items():
            unexpected = dict(stage_report.get("unexpected_diffs", {}))
            allowed = dict(stage_report.get("allowed_diffs", {}))
            lines.append(f"  {stage}: {len(allowed)} allowed diff(s), {len(unexpected)} unexpected diff(s)")
            for key, diff in unexpected.items():
                lines.append(f"    UNEXPECTED {key}: {diff['template']} -> {diff['generated']}")
    (report_dir / "validation_report.txt").write_text("\n".join(lines) + "\n", newline="\n")


def cleanup_stale_generated_dirs(workspace: Path, records: list[ModelRecord]) -> None:
    keep_names = {record.run_name for record in records if not record.registered_existing}
    for parent_name in ("runs", "output"):
        parent = workspace / parent_name
        if not parent.exists():
            continue
        for child in parent.iterdir():
            if child.is_dir() and child.name not in keep_names:
                resolved_parent = parent.resolve()
                resolved_child = child.resolve()
                if resolved_parent not in resolved_child.parents:
                    raise RuntimeError(f"Refusing to remove path outside workspace: {resolved_child}")
                shutil.rmtree(child)


def main() -> None:
    args = parse_args()
    workspace = args.workspace
    template_run = args.template_run
    inlists_zip = args.inlists_zip
    if not template_run.exists():
        raise FileNotFoundError(template_run)
    if not inlists_zip.exists():
        raise FileNotFoundError(inlists_zip)

    workspace.mkdir(parents=True, exist_ok=True)
    inputs_dir = workspace / "inputs"
    source_inlists_dir = inputs_dir / "inlists"
    source_inlists_dir.mkdir(parents=True, exist_ok=True)
    (inputs_dir / "inlists.zip").write_bytes(inlists_zip.read_bytes())
    write_template_snapshot(workspace, template_run)

    model0_params = extract_parameter_seed((template_run / CREATE_TEMPLATE).read_text())
    model0 = build_record(
        0,
        model0_params,
        str(template_run / CREATE_TEMPLATE),
        workspace,
        registered_existing=True,
        existing_run_dir=template_run,
        existing_output_dir=args.template_output,
    )
    (inputs_dir / "inlist_0").write_text((template_run / CREATE_TEMPLATE).read_text(), newline="\n")

    records = [model0]
    validation_reports: dict[str, object] = {}

    for model_index, (zip_name, text) in enumerate(read_zip_inlists(inlists_zip), start=1):
        params = extract_parameter_seed(text)
        source_path = source_inlists_dir / f"inlist_{model_index}"
        source_path.write_text(text, newline="\n")
        record = build_record(model_index, params, str(source_path), workspace)
        records.append(record)

        run_dir = Path(record.run_dir)
        output_dir = Path(record.output_dir)
        copy_template_workdir(template_run, run_dir, args.force)
        write_column_lists(template_run, run_dir)
        write_generated_inlists(template_run, run_dir, params, record)
        generate_run_scripts(run_dir)
        validation = validate_generated_inlists(template_run, run_dir)
        validation_reports[record.model_id] = validation
        write_status(output_dir, record, validation)

    if args.force:
        cleanup_stale_generated_dirs(workspace, records)

    write_manifest(workspace, records)
    write_validation_reports(workspace, validation_reports)
    write_readme(workspace)

    failures = [model_id for model_id, report in validation_reports.items() if not report.get("passed")]
    print(f"Prepared {len(records) - 1} new workdir(s) plus registered model_000.")
    print(f"Workspace: {workspace}")
    print(f"Manifest: {inputs_dir / 'manifest.csv'}")
    print(f"Validation: {inputs_dir / 'validation_report.txt'}")
    if failures:
        raise SystemExit(f"Validation failed for: {', '.join(failures)}")


if __name__ == "__main__":
    main()
