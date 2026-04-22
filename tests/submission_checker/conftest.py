import json
import pytest


MLLOG_MARKER = ":::MLLOG"


def make_mllog_line(key, value, is_error=False, is_warning=False):
    entry = {
        "key": key,
        "value": value,
        "time_ms": 0,
        "namespace": "",
        "event_type": "POINT_IN_TIME",
        "metadata": {
            "file": "test.py",
            "line_no": 1,
            "is_error": is_error,
            "is_warning": is_warning,
        },
    }
    return f"{MLLOG_MARKER} {json.dumps(entry)}\n"


@pytest.fixture()
def simple_mllog(tmp_path):
    """A minimal valid MLPerf log with two entries."""
    p = tmp_path / "mlperf_log_detail.txt"
    lines = [
        make_mllog_line("result_validity", "VALID"),
        make_mllog_line("effective_scenario", "Offline"),
        make_mllog_line("result_samples_per_second", 123.4),
    ]
    p.write_text("".join(lines))
    return p


@pytest.fixture()
def mllog_with_error(tmp_path):
    """An MLPerf log containing one error entry."""
    p = tmp_path / "mlperf_log_detail.txt"
    lines = [
        make_mllog_line("result_validity", "INVALID"),
        make_mllog_line("loadgen_error", "something went wrong", is_error=True),
    ]
    p.write_text("".join(lines))
    return p


@pytest.fixture()
def mllog_duplicate_key(tmp_path):
    """An MLPerf log with the same key appearing twice."""
    p = tmp_path / "mlperf_log_detail.txt"
    lines = [
        make_mllog_line("seeds", 1234),
        make_mllog_line("seeds", 5678),
    ]
    p.write_text("".join(lines))
    return p
