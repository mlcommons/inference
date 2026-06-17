import json
import logging
import os
import sys
import yaml

from .base import BaseParser
from ..constants import ENDPOINTS_YAML_FIELD_MAP, ENDPOINTS_JSON_ALT_PATHS, ENDPOINTS_MAPPINGS, ENDPOINTS_INFERRED_FIELDS

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


def _load_field_map(filename):
    with open(os.path.join(_FIELDS_MAP_DIR, filename), "r", encoding="utf-8") as f:
        return json.load(f)


def _get_nested(data, dotted_key):
    """Navigate a nested dict using a dot-notation key.

    Uses a greedy left-to-right match so dotted numeric keys like '99.9' are
    handled correctly: the longest matching key at each level wins.
    """
    if not isinstance(data, dict):
        return None
    parts = dotted_key.split(".")
    current = data
    i = 0
    while i < len(parts):
        if not isinstance(current, dict):
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


class EndpointsParser(BaseParser):
    def __init__(self, log_paths):
        """
        log_paths: [json_path, yaml_path]
          json_path - path to the JSON results file (result_summary.json or results.json)
          yaml_path - path to the YAML config file (config.yaml)
        """
        json_path, yaml_path = log_paths
        super().__init__(json_path)

        self.logger = logging.getLogger("MLPerfLog")
        self.messages = {}

        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        with open(yaml_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)

        forwards_map = ENDPOINTS_MAPPINGS

        for endpoints_key, loadgen_key in forwards_map.items():
            stripped = endpoints_key.strip()
            value = None

            # 1. Direct dot-notation path in the JSON result file
            value = _get_nested(json_data, stripped)

            # 2. Alternative JSON paths for known structural mismatches
            if value is None and stripped in ENDPOINTS_JSON_ALT_PATHS:
                value = _get_nested(
                    json_data, ENDPOINTS_JSON_ALT_PATHS[stripped])

            # 3. Explicit YAML field path overrides
            if value is None and stripped in ENDPOINTS_YAML_FIELD_MAP:
                value = _get_nested(
                    yaml_data, ENDPOINTS_YAML_FIELD_MAP[stripped])

            # 4. Fallback: direct dot-notation path in the YAML config
            if value is None:
                value = _get_nested(yaml_data, stripped)

            if value is not None:
                entry = {"key": loadgen_key, "value": value}
                self.messages.setdefault(loadgen_key, []).append(entry)

        self.keys = set(self.messages.keys())
        # Additional values that can be inferred from other values
        inferred_map = ENDPOINTS_INFERRED_FIELDS
        for inferred, key in inferred_map.items():
            value = self.__getitem__(key)
            if value is not None:
                entry = {"key": inferred, "value": value}
                self.messages.setdefault(inferred, []).append(entry)

        # Infer QPS from sample count / duration when not directly available.
        # generated_query_duration is in nanoseconds; divide by 1e9 for seconds.
        if self.__getitem__("generated_query_duration") and self.__getitem__(
                "generated_query_count"):
            key = "result_samples_per_second" if self.__getitem__(
                "effective_scenario").lower() == "offline" else "result_completed_samples_per_sec"
            duration_s = self.__getitem__("generated_query_duration") / 1e9
            value = self.__getitem__("generated_query_count") / duration_s
            if key not in self.messages:
                entry = {"key": key, "value": value}
                self.messages[key] = [entry]

        # Extract accuracy scores if possible
        if "accuracy_scores" in json_data:
            for dataset_name, result in json_data["accuracy_scores"].items():
                value = result.get("score", None)
                entry = {"key": "accuracy_score", "value": value}
                self.messages.setdefault("accuracy_score", []).append(entry)

        self.keys = set(self.messages.keys())
        self.logger.info(
            "Successfully loaded endpoints log from %s.",
            json_path)

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
        return self.get("num_errors")

    def has_error(self):
        """Check if the log contains any errors."""
        return self.num_errors() != 0


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s %(filename)s:%(lineno)d %(levelname)s] %(message)s",
    )
    logger = logging.getLogger("main")

    backwards_map = _load_field_map("backwards.json")

    # Collect all (json_file, yaml_file) pairs from leaf subdirectories
    pairs = []
    for root, _dirs, files in os.walk(_SAMPLE_LOGS_DIR):
        json_files = sorted(f for f in files if f.endswith(".json"))
        yaml_files = sorted(f for f in files if f.endswith(
            ".yaml") or f.endswith(".yml"))
        if json_files and yaml_files:
            pairs.append(
                (
                    os.path.join(root, json_files[0]),
                    os.path.join(root, yaml_files[0]),
                )
            )

    if not pairs:
        logger.error("No JSON+YAML pairs found under %s.", _SAMPLE_LOGS_DIR)
        return 1

    for json_path, yaml_path in sorted(pairs):
        folder = os.path.relpath(os.path.dirname(json_path), _SAMPLE_LOGS_DIR)
        print(f"\n{'=' * 70}")
        print(f"Folder : {folder}")
        print(f"JSON   : {os.path.basename(json_path)}")
        print(f"YAML   : {os.path.basename(yaml_path)}")
        print(f"{'=' * 70}")

        parser = EndpointsParser([json_path, yaml_path])

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
