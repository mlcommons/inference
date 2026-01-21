#!/usr/bin/env python3
from pathlib import Path
import shutil
import argparse


# -------------------------------------------------------
# Command-line arguments
# -------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Restructure directory layout from ROOT into NEW"
)

parser.add_argument(
    "--root",
    type=Path,
    default=Path("/Users/dfeddema/NEW_v5.1_git_clone/closed/RedHat"),
    help="Root directory of the original structure (default: built-in path)"
)

parser.add_argument(
    "--new",
    type=Path,
    default=Path("/Users/dfeddema/new_format_v6.0/closed/RedHat"),
    help="Destination directory for the new structure (default: built-in path)"
)

args = parser.parse_args()

ROOT = args.root
NEW = args.new
ORIG = ROOT

print(f"Creating new directory structure under: {NEW} using {ROOT}")


# -------------------------------------------------------
# Helpers
# -------------------------------------------------------
def copy_tree(src: Path, dst: Path):
    """Copy directory tree if it exists."""
    if src.exists():
        print(f"Copying directory: {src} → {dst}")
        shutil.copytree(src, dst, dirs_exist_ok=True)


def copy_file(src: Path, dst: Path):
    """Copy file if it exists."""
    if src.exists():
        print(f"Copying file: {src} → {dst}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


# -------------------------------------------------------
# 1. Rename code → src
# -------------------------------------------------------
code_dir = ORIG / "code"
if code_dir.is_dir():
    print("Copying and renaming code → src")
    copy_tree(code_dir, NEW / "src")


# -------------------------------------------------------
# 2. Create new documents directory
# -------------------------------------------------------
documents_dir = NEW / "documents"
documents_dir.mkdir(parents=True, exist_ok=True)
copy_tree(ORIG / "documentation", documents_dir)


# -------------------------------------------------------
# 3. Copy systems directory
# -------------------------------------------------------
copy_tree(ORIG / "systems", NEW / "systems")


# -------------------------------------------------------
# 4. Create top-level results directory (IMPORTANT)
# -------------------------------------------------------
results_dir = NEW / "results"
results_dir.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------
# 5. Move compliance / measurements / results content
# -------------------------------------------------------
def move_workload_dir(system: str, workload: str, mode: str, src_dir: Path):
    dest = results_dir / system / workload / mode
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Processing: SYSTEM={system} WORKLOAD={workload} MODE={mode}")

    # Ensure non-empty README.md exists
    readme = dest / "README.md"
    if not readme.exists():
        print("  Creating non-empty README.md")
        readme.write_text("This is a non-empty file.\n")

    # Copy accuracy & performance directories
    for metric in ("accuracy", "performance"):
        metric_dir = src_dir / metric
        if metric_dir.is_dir():
            print(f"  Copying metric: {metric}")
            copy_tree(metric_dir, dest / metric)

    # Copy test result folders (TEST*)
    for test_dir in src_dir.glob("TEST*"):
        if test_dir.is_dir():
            print(f"  Copying test folder: {test_dir.name}")
            copy_tree(test_dir, dest / test_dir.name)

    # Copy config files
    for cfg in ("user.conf", "mlperf.conf"):
        copy_file(src_dir / cfg, dest / cfg)

    # Special rename rule:
    # measurements/{SYSTEM}.json → measurements.json
    system_json = src_dir / f"{system}.json"
    if system_json.exists():
        print(f"  Renaming {system}.json → measurements.json")
        copy_file(system_json, dest / "measurements.json")

    # Copy extra files into documents/
    for extra in ("README.md", "run_calibration.sh"):
        extra_file = src_dir / extra
        if extra_file.exists():
            print(f"  Copying {extra} → documents/")
            copy_file(extra_file, documents_dir / extra_file.name)


# -------------------------------------------------------
# 6. Iterate through original sections
# -------------------------------------------------------
for section in ("compliance", "measurements", "results"):
    sec_dir = ORIG / section
    if not sec_dir.is_dir():
        continue

    print(f"Scanning section: {section}")

    for system_dir in sec_dir.iterdir():
        if not system_dir.is_dir():
            continue

        system = system_dir.name
        print(f"  Processing system: {system}")

        for workload_dir in system_dir.iterdir():
            if not workload_dir.is_dir():
                continue

            workload = workload_dir.name

            for mode in ("Offline", "Server"):
                src_path = workload_dir / mode
                if src_path.is_dir():
                    move_workload_dir(system, workload, mode, src_path)


print("Restructure complete!")

