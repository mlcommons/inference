#!/usr/bin/env python3
"""
Parse MLPerf performance logs (mlperf_log_summary.txt + mlperf_log_detail.txt)
and store results in a Neon (PostgreSQL) database.

Usage:
    python parse_perf_logs.py --root /path/to/submission --dsn "postgresql://..."

The script walks the submission tree looking for:
    {division}/{submitter}/results/{system}/{model}/{scenario}/performance/run_{N}/

Tables created (if not exist):
    performance_summary  -- one row per run, key metrics as columns
    performance_detail   -- one row per MLLOG entry per run
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import psycopg

# ---------------------------------------------------------------------------
# Path parsing
# ---------------------------------------------------------------------------

# Match: .../{division}/{submitter}/results/{system}/{model}/{scenario}/performance/run_{N}
_PERF_RUN_RE = re.compile(
    r"(?P<division>[^/]+)"
    r"/(?P<submitter>[^/]+)"
    r"/results"
    r"/(?P<system>[^/]+)"
    r"/(?P<model>[^/]+)"
    r"/(?P<scenario>[^/]+)"               # is there data in here we want ?  ./offline/TEST01/performance/run_1
    r"/performance"
    r"/run_(?P<run_number>\d+)$"
)


def parse_path_context(run_dir: Path, root: Path) -> dict | None:
    rel = run_dir.relative_to(root).as_posix()

    # Debug
    print (f"run_dir relative to root : {rel}")
    m = _PERF_RUN_RE.search(rel)
    if not m:
        return None
    return {k: v for k, v in m.groupdict().items()}


# ---------------------------------------------------------------------------
# mlperf_log_summary.txt parser
# ---------------------------------------------------------------------------

def parse_summary(path: Path) -> dict:
    """Return a flat dict of all fields in mlperf_log_summary.txt."""
    data = {}
    if not path.exists():
        return data

    text = path.read_text(errors="replace")

    # --- Results section ---
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("="):
            continue

        # Key : value pairs
        if ":" in line:
            key, _, val = line.partition(":")
            key = key.strip().lower().replace(" ", "_").replace("(", "").replace(")", "")
            val = val.strip()
            data[key] = _coerce(val)

    # Warnings / errors
    data["has_warnings"] = "No warnings encountered" not in text
    data["has_errors"] = "No errors encountered" not in text

    # Normalise primary metric  ------------------------------------------------
    # Offline/Server → samples_per_second
    # SingleStream/MultiStream → 90th or 99th percentile latency
    # The summary already contains these as parsed keys; just alias the main one.
    if "samples_per_second" in data:
        data["primary_metric"] = "samples_per_second"
        data["primary_value"] = data["samples_per_second"]
    elif "scheduled_samples_per_second" in data:
        data["primary_metric"] = "scheduled_samples_per_second"
        data["primary_value"] = data["scheduled_samples_per_second"]
    elif "90th_percentile_latency_ns" in data:
        data["primary_metric"] = "90th_percentile_latency_ns"
        data["primary_value"] = data["90th_percentile_latency_ns"]
    elif "99th_percentile_latency_ns" in data:
        data["primary_metric"] = "99th_percentile_latency_ns"
        data["primary_value"] = data["99th_percentile_latency_ns"]
    else:
        data["primary_metric"] = None
        data["primary_value"] = None

    return data


def _coerce(val: str):
    """Try to parse value as int, float, bool, or leave as str."""
    if val in ("Yes", "True", "true"):
        return True
    if val in ("No", "False", "false"):
        return False
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val


# ---------------------------------------------------------------------------
# mlperf_log_detail.txt parser
# ---------------------------------------------------------------------------

_MLLOG_PREFIX = ":::MLLOG "


def parse_detail(path: Path) -> list[dict]:
    """Return list of dicts, one per MLLOG entry."""
    entries = []
    if not path.exists():
        return entries

    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith(_MLLOG_PREFIX):
            continue
        raw = line[len(_MLLOG_PREFIX):]
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue

        meta = obj.get("metadata", {})
        value = obj.get("value")
        # Store complex values as JSON string
        if isinstance(value, (dict, list)):
            value_str = json.dumps(value)
            value_num = None
        else:
            value_str = str(value) if value is not None else None
            value_num = float(value) if isinstance(value, (int, float)) else None

        entries.append({
            "key": obj.get("key"),
            "value_str": value_str,
            "value_num": value_num,
            "time_ms": obj.get("time_ms"),
            "event_type": obj.get("event_type"),
            "is_error": meta.get("is_error", False),
            "is_warning": meta.get("is_warning", False),
            "source_file": meta.get("file"),
            "line_no": meta.get("line_no"),
        })

    return entries


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

DDL = """
CREATE TABLE IF NOT EXISTS performance_summary (
    id                          SERIAL PRIMARY KEY,
    submitter                   TEXT NOT NULL,
    division                    TEXT,
    system                      TEXT,
    model                       TEXT,
    scenario                    TEXT,
    run_number                  INTEGER,

    -- Core result
    sut_name                    TEXT,
    mode                        TEXT,
    result_validity             TEXT,
    primary_metric              TEXT,
    primary_value               DOUBLE PRECISION,

    -- Latency stats (ns)
    min_latency_ns              BIGINT,
    max_latency_ns              BIGINT,
    mean_latency_ns             BIGINT,
    p50_latency_ns              BIGINT,
    p90_latency_ns              BIGINT,
    p95_latency_ns              BIGINT,
    p97_latency_ns              BIGINT,
    p99_latency_ns              BIGINT,
    p999_latency_ns             BIGINT,

    -- Test parameters
    samples_per_query           INTEGER,
    target_qps                  DOUBLE PRECISION,
    target_latency_ns           BIGINT,
    max_async_queries           INTEGER,
    min_duration_ms             BIGINT,
    max_duration_ms             BIGINT,
    performance_sample_count    INTEGER,
    accuracy_sample_count       INTEGER,

    -- Quality flags
    has_warnings                BOOLEAN,
    has_errors                  BOOLEAN,

    -- Provenance
    file_path                   TEXT,
    ingested_at                 TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (submitter, division, system, model, scenario, run_number)
);

CREATE TABLE IF NOT EXISTS performance_detail (
    id          SERIAL PRIMARY KEY,
    submitter   TEXT NOT NULL,
    division    TEXT,
    system      TEXT,
    model       TEXT,
    scenario    TEXT,
    run_number  INTEGER,

    key         TEXT,
    value_str   TEXT,
    value_num   DOUBLE PRECISION,
    time_ms     DOUBLE PRECISION,
    event_type  TEXT,
    is_error    BOOLEAN,
    is_warning  BOOLEAN,
    source_file TEXT,
    line_no     INTEGER,

    file_path   TEXT,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_perf_detail_run
    ON performance_detail (submitter, system, model, scenario, run_number);
CREATE INDEX IF NOT EXISTS idx_perf_detail_key
    ON performance_detail (key);
"""


def ensure_schema(conn):
    with conn.cursor() as cur:
        cur.execute(DDL)
    conn.commit()


# ---------------------------------------------------------------------------
# Ingest one run directory
# ---------------------------------------------------------------------------

def ingest_run(conn, run_dir: Path, root: Path, dry_run: bool = False):
    ctx = parse_path_context(run_dir, root)
    if ctx is None:
        print(f"  SKIP (path doesn't match expected structure): {run_dir}", file=sys.stderr)
        return

    # These are the files we expect to find.
    # They will be parsed and added to a PostGres DB    
    summary_path = run_dir / "mlperf_log_summary.txt"
    detail_path = run_dir / "mlperf_log_detail.txt"

    summary = parse_summary(summary_path)
    detail_entries = parse_detail(detail_path)

    if not summary and not detail_entries:
        print(f"  SKIP (no data): {run_dir}", file=sys.stderr)
        return

    if dry_run:
        print(f"  DRY RUN {ctx['submitter']} / {ctx['model']} / {ctx['scenario']} run {ctx['run_number']}")
        print(f"    summary keys: {list(summary.keys())}")
        print(f"    detail entries: {len(detail_entries)}")
        return

    def _bigint(d, key):
        v = d.get(key)
        return int(v) if v is not None else None

    def _float(d, key):
        v = d.get(key)
        return float(v) if v is not None else None

    with conn.cursor() as cur:
        # --- performance_summary upsert ---
        cur.execute("""
            INSERT INTO performance_summary (
                submitter, division, system, model, scenario, run_number,
                sut_name, mode, result_validity,
                primary_metric, primary_value,
                min_latency_ns, max_latency_ns, mean_latency_ns,
                p50_latency_ns, p90_latency_ns, p95_latency_ns,
                p97_latency_ns, p99_latency_ns, p999_latency_ns,
                samples_per_query, target_qps, target_latency_ns,
                max_async_queries, min_duration_ms, max_duration_ms,
                performance_sample_count, accuracy_sample_count,
                has_warnings, has_errors,
                file_path
            ) VALUES (
                %(submitter)s, %(division)s, %(system)s, %(model)s,
                %(scenario)s, %(run_number)s,
                %(sut_name)s, %(mode)s, %(result_validity)s,
                %(primary_metric)s, %(primary_value)s,
                %(min_latency_ns)s, %(max_latency_ns)s, %(mean_latency_ns)s,
                %(p50_latency_ns)s, %(p90_latency_ns)s, %(p95_latency_ns)s,
                %(p97_latency_ns)s, %(p99_latency_ns)s, %(p999_latency_ns)s,
                %(samples_per_query)s, %(target_qps)s, %(target_latency_ns)s,
                %(max_async_queries)s, %(min_duration_ms)s, %(max_duration_ms)s,
                %(performance_sample_count)s, %(accuracy_sample_count)s,
                %(has_warnings)s, %(has_errors)s,
                %(file_path)s
            )
            ON CONFLICT (submitter, division, system, model, scenario, run_number)
            DO UPDATE SET
                sut_name                 = EXCLUDED.sut_name,
                mode                     = EXCLUDED.mode,
                result_validity          = EXCLUDED.result_validity,
                primary_metric           = EXCLUDED.primary_metric,
                primary_value            = EXCLUDED.primary_value,
                min_latency_ns           = EXCLUDED.min_latency_ns,
                max_latency_ns           = EXCLUDED.max_latency_ns,
                mean_latency_ns          = EXCLUDED.mean_latency_ns,
                p50_latency_ns           = EXCLUDED.p50_latency_ns,
                p90_latency_ns           = EXCLUDED.p90_latency_ns,
                p95_latency_ns           = EXCLUDED.p95_latency_ns,
                p97_latency_ns           = EXCLUDED.p97_latency_ns,
                p99_latency_ns           = EXCLUDED.p99_latency_ns,
                p999_latency_ns          = EXCLUDED.p999_latency_ns,
                samples_per_query        = EXCLUDED.samples_per_query,
                target_qps               = EXCLUDED.target_qps,
                target_latency_ns        = EXCLUDED.target_latency_ns,
                max_async_queries        = EXCLUDED.max_async_queries,
                min_duration_ms          = EXCLUDED.min_duration_ms,
                max_duration_ms          = EXCLUDED.max_duration_ms,
                performance_sample_count = EXCLUDED.performance_sample_count,
                accuracy_sample_count    = EXCLUDED.accuracy_sample_count,
                has_warnings             = EXCLUDED.has_warnings,
                has_errors               = EXCLUDED.has_errors,
                file_path                = EXCLUDED.file_path,
                ingested_at              = NOW()
        """, {
            **ctx,
            "run_number": int(ctx["run_number"]),
            "sut_name": summary.get("sut_name"),
            "mode": summary.get("mode"),
            "result_validity": str(summary.get("result_is", summary.get("result", ""))),
            "primary_metric": summary.get("primary_metric"),
            "primary_value": _float(summary, "primary_value"),
            "min_latency_ns": _bigint(summary, "min_latency_ns"),
            "max_latency_ns": _bigint(summary, "max_latency_ns"),
            "mean_latency_ns": _bigint(summary, "mean_latency_ns"),
            "p50_latency_ns": _bigint(summary, "50.00_percentile_latency_ns"),
            "p90_latency_ns": _bigint(summary, "90.00_percentile_latency_ns"),
            "p95_latency_ns": _bigint(summary, "95.00_percentile_latency_ns"),
            "p97_latency_ns": _bigint(summary, "97.00_percentile_latency_ns"),
            "p99_latency_ns": _bigint(summary, "99.00_percentile_latency_ns"),
            "p999_latency_ns": _bigint(summary, "99.90_percentile_latency_ns"),
            "samples_per_query": _bigint(summary, "samples_per_query"),
            "target_qps": _float(summary, "target_qps"),
            "target_latency_ns": _bigint(summary, "target_latency_ns"),
            "max_async_queries": _bigint(summary, "max_async_queries"),
            "min_duration_ms": _bigint(summary, "min_duration_ms"),
            "max_duration_ms": _bigint(summary, "max_duration_ms"),
            "performance_sample_count": _bigint(summary, "performance_sample_count"),
            "accuracy_sample_count": _bigint(summary, "accuracy_sample_count"),
            "has_warnings": summary.get("has_warnings", False),
            "has_errors": summary.get("has_errors", False),
            "file_path": str(summary_path),
        })

        # --- performance_detail: delete old entries then bulk insert ---
        cur.execute("""
            DELETE FROM performance_detail
            WHERE submitter = %s AND system = %s AND model = %s
              AND scenario = %s AND run_number = %s
        """, (ctx["submitter"], ctx["system"], ctx["model"],
              ctx["scenario"], int(ctx["run_number"])))

        if detail_entries:
            cur.executemany("""
                INSERT INTO performance_detail (
                    submitter, division, system, model, scenario, run_number,
                    key, value_str, value_num, time_ms,
                    event_type, is_error, is_warning, source_file, line_no,
                    file_path
                ) VALUES (
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s
                )
            """, [
                (
                    ctx["submitter"], ctx["division"], ctx["system"],
                    ctx["model"], ctx["scenario"], int(ctx["run_number"]),
                    e["key"], e["value_str"], e["value_num"], e["time_ms"],
                    e["event_type"], e["is_error"], e["is_warning"],
                    e["source_file"], e["line_no"],
                    str(detail_path),
                )
                for e in detail_entries
            ])

    conn.commit()
    print(f"  OK  {ctx['submitter']:30s} {ctx['model']:30s} {ctx['scenario']:15s} run {ctx['run_number']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_run_dirs(root: Path):
    """Yield all .../performance/run_N directories under root."""
    for p in root.rglob("run_*"):
        if p.is_dir() and p.parent.name == "performance":
            yield p


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True,
                        help="Root of the submission directory tree")
    parser.add_argument("--dsn", required=True,
                        help="Neon/PostgreSQL connection string, e.g. "
                             "postgresql://user:pass@host/dbname?sslmode=require")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse files and print what would be inserted, "
                             "without touching the database")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        sys.exit(f"Root path does not exist: {root}")

    run_dirs = sorted(find_run_dirs(root))
    if not run_dirs:
        sys.exit(f"No performance/run_* directories found under {root}")

    # Debug
    #print (f"run_dirs = {run_dirs}")
    for i in range (0,len(run_dirs)):
        print (run_dirs[i])

    # Should have a format line ../.../run_y
    print(f"Found {len(run_dirs)} run director{'y' if len(run_dirs) == 1 else 'ies'} under {root}")

    if args.dry_run:
        for rd in run_dirs:
            ingest_run(None, rd, root, dry_run=True)
        return

    # connect to PostGres DB
    conn = psycopg.connect(args.dsn)
    try:
        ensure_schema(conn)
        print("Schema ready.")
        for rd in run_dirs:
            ingest_run(conn, rd, root)
    finally:
        conn.close()

    print("Done.")


if __name__ == "__main__":
    main()
