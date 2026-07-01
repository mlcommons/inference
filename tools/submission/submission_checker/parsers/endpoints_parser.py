import json
import logging
import os
import sys
import yaml

from .base import BaseParser
from ..constants import (
    ENDPOINTS_YAML_FIELD_MAP,
    ENDPOINTS_JSON_ALT_PATHS,
    ENDPOINTS_MAPPINGS,
    ENDPOINTS_INFERRED_FIELDS,
)

_FIELDS_MAP_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "helper",
    "fields_map")
_SAMPLE_LOGS_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "helper",
    "sample_logs")

_RESULT_SUMMARY_FILE = "result_summary.json"
_RESULTS_FILE = "results.json"
_CONFIG_FILES = ("config.yaml", "config.yml")


def _load_field_map(filename):
    with open(os.path.join(_FIELDS_MAP_DIR, filename), "r", encoding="utf-8") as f:
        return json.load(f)


def _get_nested(data, dotted_key):
    """Navigate a nested dict using a dot-notation key.

    Uses a greedy left-to-right match so dotted numeric keys like '99.9' are
    handled correctly: the longest matching key at each level wins.

    Also handles float-formatted integer keys: '50.0' resolves to key '50'
    (common in the ENDPOINTS_MAPPINGS percentile entries).
    """
    if not isinstance(data, dict):
        return None
    parts = dotted_key.split(".")
    current = data
    i = 0
    while i < len(parts):
        if not isinstance(current, dict):
            # Trailing '.0' on a float-formatted integer key: treat as
            # consumed.
            if parts[i:] == ["0"] and not isinstance(current, (dict, list)):
                return current
            return None
        found = False
        for j in range(len(parts), i, -1):
            candidate = ".".join(parts[i:j])
            if candidate in current:
                current = current[candidate]
                i = j
                found = True
                break
        if not found:
            return None
    if isinstance(current, (dict, list)) and not current:
        return None
    return current


def _resolve_value(stripped, summary_data, results_data, yaml_data):
    """Look up a field in three data sources in priority order.

    Priority: result_summary.json > results.json > config.yaml
    Within each JSON source, a direct dot-notation path is tried first,
    then the alternative paths from ENDPOINTS_JSON_ALT_PATHS.
    For the YAML source, the explicit path overrides in ENDPOINTS_YAML_FIELD_MAP
    are tried first, then a direct dot-notation path.
    """
    for data in (summary_data, results_data):
        value = _get_nested(data, stripped)
        if value is None and stripped in ENDPOINTS_JSON_ALT_PATHS:
            value = _get_nested(data, ENDPOINTS_JSON_ALT_PATHS[stripped])
        if value is not None:
            return value

    # YAML: explicit path map first, then direct
    if stripped in ENDPOINTS_YAML_FIELD_MAP:
        value = _get_nested(yaml_data, ENDPOINTS_YAML_FIELD_MAP[stripped])
        if value is not None:
            return value
    return _get_nested(yaml_data, stripped)


class EndpointsParser(BaseParser):
    def __init__(self, run_dir):
        """
        run_dir: path to the run directory containing:
          - result_summary.json  (highest priority)
          - results.json
          - config.yaml / config.yml  (lowest priority)
        """
        super().__init__(run_dir)

        self.logger = logging.getLogger("MLPerfLog")
        self.messages = {}

        summary_data = self._load_json(
            os.path.join(run_dir, _RESULT_SUMMARY_FILE))
        results_data = self._load_json(os.path.join(run_dir, _RESULTS_FILE))
        yaml_data = self._load_yaml(run_dir)

        for endpoints_key, loadgen_key in ENDPOINTS_MAPPINGS.items():
            stripped = endpoints_key.strip()
            value = _resolve_value(
                stripped, summary_data, results_data, yaml_data)
            if value is not None:
                self.messages.setdefault(loadgen_key, []).append(
                    {"key": loadgen_key, "value": value}
                )

        self.keys = set(self.messages.keys())

        # Inferred fields: copy the value of one loadgen key to another
        for inferred_key, source_key in ENDPOINTS_INFERRED_FIELDS.items():
            value = self[source_key]
            if value is not None:
                self.messages.setdefault(inferred_key, []).append(
                    {"key": inferred_key, "value": value}
                )

        # Infer QPS from count / duration when not directly available
        duration_ns = self["generated_query_duration"]
        count = self["generated_query_count"]
        scenario = self["effective_scenario"]
        if duration_ns and count and scenario:
            qps_key = (
                "result_samples_per_second"
                if scenario.lower() == "offline"
                else "result_completed_samples_per_sec"
            )
            if qps_key not in self.messages:
                qps = count / (duration_ns / 1e9)
                self.messages[qps_key] = [{"key": qps_key, "value": qps}]

        # Expose accuracy scores stored in results.json
        for result in results_data.get("accuracy_scores", {}).values():
            score = result.get("score")
            if score is not None:
                self.messages.setdefault("accuracy_score", []).append(
                    {"key": "accuracy_score", "value": score}
                )

        self.keys = set(self.messages.keys())
        self.logger.info("Successfully loaded endpoints log from %s.", run_dir)

    def _load_json(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except BaseException:
            self.logger.error("Could not load json file from %s", path)
            return {}
        return {}

    def _load_yaml(self, run_dir):
        for name in _CONFIG_FILES:
            path = os.path.join(run_dir, name)
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        return yaml.safe_load(f) or {}
                except BaseException:
                    pass
        self.logger.error("Yaml file not found in directory %s", run_dir)
        return {}

    def __getitem__(self, key):
        if key not in self.keys:
            return None
        results = self.messages[key]
        if len(results) > 1:
            self.logger.warning(
                "Multiple messages with key %s in the log. Empirically choosing the first one.",
                key,
            )
        return results[0]["value"]

    def get(self, key):
        return self[key]

    def get_messages(self):
        return self.messages

    def get_keys(self):
        return self.keys

    def num_errors(self):
        return self["num_errors"]

    def has_error(self):
        return self.num_errors() != 0


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s %(filename)s:%(lineno)d %(levelname)s] %(message)s",
    )
    logger = logging.getLogger("main")

    backwards_map = _load_field_map("backwards.json")

    # Collect all run directories (those containing at least one JSON and one
    # YAML)
    run_dirs = []
    for root, _dirs, files in os.walk(_SAMPLE_LOGS_DIR):
        has_json = any(f.endswith(".json") for f in files)
        has_yaml = any(f.endswith(".yaml") or f.endswith(".yml")
                       for f in files)
        if has_json and has_yaml:
            run_dirs.append(root)

    if not run_dirs:
        logger.error("No run directories found under %s.", _SAMPLE_LOGS_DIR)
        return 1

    for run_dir in sorted(run_dirs):
        folder = os.path.relpath(run_dir, _SAMPLE_LOGS_DIR)
        print(f"\n{'=' * 70}")
        print(f"Directory: {folder}")
        print(f"{'=' * 70}")

        parser = EndpointsParser(run_dir)

        found = []
        not_found = []
        for loadgen_key, endpoints_key in backwards_map.items():
            value = parser[loadgen_key]
            if value is not None:
                found.append((loadgen_key, endpoints_key, value))
            else:
                not_found.append((loadgen_key, endpoints_key))

        total = len(backwards_map)
        print(f"\nFound ({len(found)}/{total}):")
        for loadgen_key, endpoints_key, value in found:
            print(f"  {loadgen_key:<55} = {value}")

        print(f"\nNot found ({len(not_found)}/{total}):")
        for loadgen_key, endpoints_key in not_found:
            label = endpoints_key if endpoints_key != "None" else "(no endpoints mapping)"
            print(f"  {loadgen_key:<55}  [{label}]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
