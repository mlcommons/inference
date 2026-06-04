import json
import pytest
from submission_checker.parsers.loadgen_parser import LoadgenParser


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


class TestLoadgenParserBasic:
    def test_parses_valid_log(self, simple_mllog):
        p = LoadgenParser(str(simple_mllog))
        assert "result_validity" in p.get_keys()
        assert "effective_scenario" in p.get_keys()

    def test_getitem_returns_first_value(self, simple_mllog):
        p = LoadgenParser(str(simple_mllog))
        assert p["result_validity"] == "VALID"
        assert p["effective_scenario"] == "Offline"

    def test_getitem_missing_key_returns_none(self, simple_mllog):
        p = LoadgenParser(str(simple_mllog))
        assert p["nonexistent_key"] is None

    def test_num_messages(self, simple_mllog):
        p = LoadgenParser(str(simple_mllog))
        assert p.num_messages() == 3

    def test_get_messages_is_dict(self, simple_mllog):
        p = LoadgenParser(str(simple_mllog))
        assert isinstance(p.get_messages(), dict)


class TestLoadgenParserErrors:
    def test_no_error_in_clean_log(self, simple_mllog):
        p = LoadgenParser(str(simple_mllog))
        assert p.num_errors() == 0
        assert not p.has_error()

    def test_detects_error_entry(self, mllog_with_error):
        p = LoadgenParser(str(mllog_with_error))
        assert p.num_errors() == 1
        assert p.has_error()

    def test_get_errors_returns_list(self, mllog_with_error):
        p = LoadgenParser(str(mllog_with_error))
        errors = p.get_errors()
        assert len(errors) == 1
        assert errors[0]["key"] == "loadgen_error"


class TestLoadgenParserDuplicateKeys:
    def test_duplicate_key_stored_twice(self, mllog_duplicate_key):
        p = LoadgenParser(str(mllog_duplicate_key))
        entries = p.get("seeds")
        assert len(entries) == 2

    def test_getitem_returns_first_on_duplicate(self, mllog_duplicate_key):
        p = LoadgenParser(str(mllog_duplicate_key))
        assert p["seeds"] == 1234


class TestLoadgenParserStrict:
    def test_invalid_first_line_raises(self, tmp_path):
        bad = tmp_path / "bad.txt"
        bad.write_text("not a valid mllog line\n")
        with pytest.raises(RuntimeError, match="Marker not found"):
            LoadgenParser(str(bad))

    def test_invalid_json_strict_raises(self, tmp_path):
        p = tmp_path / "log.txt"
        p.write_text(":::MLLOG not-valid-json\n")
        with pytest.raises(RuntimeError):
            LoadgenParser(str(p), strict=True)

    def test_invalid_json_non_strict_skips(self, tmp_path):
        p = tmp_path / "log.txt"
        valid_line = make_mllog_line("result_validity", "VALID")
        p.write_text(valid_line + ":::MLLOG not-valid-json\n")
        parser = LoadgenParser(str(p), strict=False)
        assert parser["result_validity"] == "VALID"


class TestLoadgenParserEndpoints:
    def test_endpoints_marker_accepted(self, tmp_path):
        p = tmp_path / "log.txt"
        entry = json.dumps({
            "key": "endpoint_key",
            "value": "endpoint_value",
            "metadata": {"is_error": False, "is_warning": False},
        })
        p.write_text(f":::ENDPTS {entry}\n")
        parser = LoadgenParser(str(p))
        assert parser.log_is_endpoints
        assert parser["endpoint_key"] == "endpoint_value"


class TestLoadgenParserDump:
    def test_dump_writes_json(self, simple_mllog, tmp_path):
        parser = LoadgenParser(str(simple_mllog))
        out = tmp_path / "out.json"
        parser.dump(str(out))
        data = json.loads(out.read_text())
        assert "result_validity" in data
