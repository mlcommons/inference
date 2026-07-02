import os
import pytest
from submission_checker.utils import (
    files_diff,
    get_boolean,
    is_number,
    lower_list,
    contains_list,
    merge_two_dict,
    sum_dict_values,
    split_path,
    list_dir,
    list_files,
    list_empty_dirs_recursively,
    list_files_recursively,
)


# ---------------------------------------------------------------------------
# files_diff
# ---------------------------------------------------------------------------

class TestFilesDiff:
    def test_identical_lists_no_diff(self):
        assert files_diff(["a.txt", "b.txt"], ["a.txt", "b.txt"]) == set()

    def test_missing_file_reported(self):
        diff = files_diff(["a.txt"], ["a.txt", "b.txt"])
        assert "b.txt" in diff

    def test_extra_file_reported(self):
        diff = files_diff(["a.txt", "extra.txt"], ["a.txt"])
        assert "extra.txt" in diff

    def test_optional_files_ignored(self):
        # mlperf_log_trace.json is always optional
        diff = files_diff(["a.txt", "mlperf_log_trace.json"], ["a.txt"])
        assert diff == set()

    def test_custom_optional_ignored(self):
        diff = files_diff(["a.txt", "custom.json"], [
                          "a.txt"], optional=["custom.json"])
        assert diff == set()


# ---------------------------------------------------------------------------
# get_boolean
# ---------------------------------------------------------------------------

class TestGetBoolean:
    @pytest.mark.parametrize("val", [True, "true", "True", "TRUE", 1])
    def test_truthy_values(self, val):
        assert get_boolean(val) is True

    @pytest.mark.parametrize("val", [False, "false", "False", "FALSE", 0])
    def test_falsy_values(self, val):
        assert get_boolean(val) is False

    def test_none_returns_false(self):
        assert get_boolean(None) is False

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            get_boolean([])


# ---------------------------------------------------------------------------
# is_number
# ---------------------------------------------------------------------------

class TestIsNumber:
    @pytest.mark.parametrize("val", ["3.14", "0", "-1", "1e5", "123"])
    def test_numeric_strings(self, val):
        assert is_number(val) is True

    @pytest.mark.parametrize("val", ["abc", "", "1.2.3"])
    def test_non_numeric_strings(self, val):
        assert is_number(val) is False

    def test_nan_is_numeric(self):
        # float("NaN") succeeds in Python, so is_number returns True
        assert is_number("NaN") is True


# ---------------------------------------------------------------------------
# lower_list
# ---------------------------------------------------------------------------

def test_lower_list_converts_to_lowercase():
    assert lower_list(["Hello", "WORLD", "123"]) == ["hello", "world", "123"]


def test_lower_list_empty():
    assert lower_list([]) == []


# ---------------------------------------------------------------------------
# contains_list
# ---------------------------------------------------------------------------

class TestContainsList:
    def test_all_present(self):
        missing, ok = contains_list(["a", "b", "c"], ["a", "b"])
        assert ok is True
        assert missing == []

    def test_some_missing(self):
        missing, ok = contains_list(["a"], ["a", "b"])
        assert ok is False
        assert "b" in missing

    def test_empty_needle(self):
        _, ok = contains_list(["a"], [])
        assert ok is True


# ---------------------------------------------------------------------------
# merge_two_dict
# ---------------------------------------------------------------------------

class TestMergeTwoDict:
    def test_disjoint_dicts_merged(self):
        result = merge_two_dict({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_overlapping_keys_summed(self):
        result = merge_two_dict({"a": [1]}, {"a": [2]})
        assert result == {"a": [1, 2]}

    def test_original_not_mutated(self):
        x = {"a": 1}
        merge_two_dict(x, {"b": 2})
        assert x == {"a": 1}


# ---------------------------------------------------------------------------
# sum_dict_values
# ---------------------------------------------------------------------------

def test_sum_dict_values():
    assert sum_dict_values({"a": 1, "b": 2, "c": 3}) == 6


def test_sum_dict_values_empty():
    assert sum_dict_values({}) == 0


# ---------------------------------------------------------------------------
# split_path
# ---------------------------------------------------------------------------

def test_split_path_unix():
    assert split_path("foo/bar/baz") == ["foo", "bar", "baz"]


def test_split_path_windows_backslash():
    assert split_path("foo\\bar\\baz") == ["foo", "bar", "baz"]


# ---------------------------------------------------------------------------
# Filesystem helpers (use tmp_path)
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_tree(tmp_path):
    (tmp_path / "subA").mkdir()
    (tmp_path / "subB").mkdir()
    (tmp_path / "subA" / "file1.txt").write_text("x")
    (tmp_path / "subA" / "file2.txt").write_text("y")
    (tmp_path / "subB").mkdir(exist_ok=True)
    return tmp_path


def test_list_dir(sample_tree):
    dirs = list_dir(str(sample_tree))
    assert dirs == ["subA", "subB"]


def test_list_files(sample_tree):
    files = list_files(str(sample_tree / "subA"))
    assert files == ["file1.txt", "file2.txt"]


def test_list_empty_dirs(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    (tmp_path / "nonempty").mkdir()
    (tmp_path / "nonempty" / "f.txt").write_text("x")
    empties = list_empty_dirs_recursively(str(tmp_path))
    assert str(empty) in empties
    assert str(tmp_path / "nonempty") not in empties


def test_list_files_recursively(sample_tree):
    files = list_files_recursively(str(sample_tree))
    names = [os.path.basename(f) for f in files]
    assert "file1.txt" in names
    assert "file2.txt" in names
