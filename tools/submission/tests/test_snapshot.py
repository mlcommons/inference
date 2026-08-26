"""Snapshot (golden-file) test for the MLPerf submission checker.

This runs the submission checker end-to-end against a *pinned* commit of a
real submission repository and compares the produced ``summary.csv`` against a
committed snapshot using `syrupy <https://github.com/syrupy-project/syrupy>`_.
It catches unintended changes to the checker's output (columns, computed
metrics, inferred scenarios, units, ...) without anyone hand-writing expected
values.

How it works
------------
* The submission repo is cloned in CI (see
  ``.github/workflows/test-submission-checker-snapshot.yml``) and its path is
  passed in via the ``MLPERF_SUBMISSION_DIR`` environment variable.
* We run against *all* submitters in the pinned repo. Because the commit is
  fixed, the output is fully deterministic, so the snapshot only changes when
  the checker's behavior changes.
* Volatile fields are normalized before comparison: the ``Location`` column
  holds an absolute path, so we rewrite it to be relative to the submission
  root, and rows are sorted so directory-iteration order never matters.
* The snapshot is stored as a plain, reviewable ``.csv`` (via syrupy's
  single-file extension) under ``tests/__snapshots__/test_snapshot/``.

Updating the snapshot
---------------------
When a checker change *intentionally* alters the output, regenerate with
syrupy's standard flag::

    export MLPERF_SUBMISSION_DIR=/path/to/cloned/submissions/repo
    pytest tools/submission/tests/test_snapshot.py --snapshot-update

then review and commit the updated file under ``tests/__snapshots__/``.

If ``MLPERF_SUBMISSION_DIR`` is not set (e.g. a plain local ``pytest`` run
without the cloned repo), the test is skipped rather than failed.
"""

import csv
import io
import os
import subprocess
import sys
from pathlib import Path

import pytest
from syrupy.extensions.single_file import SingleFileSnapshotExtension, WriteMode

# Directory layout: this file lives at tools/submission/tests/test_snapshot.py
TESTS_DIR = Path(__file__).resolve().parent
CHECKER_ROOT = TESTS_DIR.parent  # tools/submission (parent of the package)

# --- Pinned inputs. Keep these in sync with the CI workflow. -----------------
SUBMISSION_REPO = "https://github.com/mlcommons/inference_results_v6.0"
PINNED_SHA = "4d3916ac9cf474b679cdfcf492d43a0559418ad1"
VERSION = "v6.0"
SKIP_FLAGS = [
    "--skip-power-check",
    "--skip-extra-files-in-root-check",
    "--skip-extra-accuracy-files-check",
    "--skip-all-systems-have-results-check",
    "--skip-calibration-check",
]

LOCATION_COLUMN = "Location"


class CSVSnapshotExtension(SingleFileSnapshotExtension):
    """Store each snapshot as a plain, diff-friendly ``.csv`` text file."""

    _write_mode = WriteMode.TEXT
    file_extension = "csv"


@pytest.fixture
def csv_snapshot(snapshot):
    return snapshot.use_extension(CSVSnapshotExtension)


def _submission_dir() -> Path:
    raw = os.environ.get("MLPERF_SUBMISSION_DIR")
    if not raw:
        pytest.skip(
            "MLPERF_SUBMISSION_DIR is not set; clone "
            f"{SUBMISSION_REPO} at {PINNED_SHA} and point this variable at it "
            "to run the snapshot test."
        )
    path = Path(raw).resolve()
    if not path.is_dir():
        pytest.skip(f"MLPERF_SUBMISSION_DIR={path} does not exist.")
    return path


def _run_checker(submission_dir: Path, out_csv: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "submission_checker.main",
        "--input",
        str(submission_dir),
        "--version",
        VERSION,
        "--csv",
        str(out_csv),
        *SKIP_FLAGS,
    ]
    result = subprocess.run(
        cmd,
        cwd=CHECKER_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "submission checker exited non-zero\n"
        f"STDOUT:\n{result.stdout[-4000:]}\n"
        f"STDERR:\n{result.stderr[-4000:]}"
    )


def _normalize(csv_text: str, submission_dir: Path) -> str:
    """Make the CSV portable and order-independent.

    * Rewrite the absolute ``Location`` path to be relative to the submission
      root (so the snapshot does not embed a machine-specific prefix).
    * Sort the data rows so directory-iteration order does not affect the
      comparison.
    """
    prefix = str(submission_dir).rstrip("/") + "/"
    reader = csv.DictReader(io.StringIO(csv_text))
    fieldnames = reader.fieldnames
    assert fieldnames, "checker produced a CSV with no header"

    rows = []
    for row in reader:
        loc = row.get(LOCATION_COLUMN, "")
        if loc.startswith(prefix):
            row[LOCATION_COLUMN] = loc[len(prefix):]
        rows.append(row)

    rows.sort(key=lambda r: [r.get(c, "") for c in fieldnames])

    buf = io.StringIO()
    # match the checker's own quoting: every field wrapped in double quotes
    writer = csv.DictWriter(
        buf, fieldnames=fieldnames, quoting=csv.QUOTE_ALL, lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def test_summary_csv_matches_snapshot(tmp_path, csv_snapshot):
    submission_dir = _submission_dir()
    out_csv = tmp_path / "summary.csv"

    _run_checker(submission_dir, out_csv)
    actual = _normalize(out_csv.read_text(), submission_dir)

    assert actual == csv_snapshot
