
from .loader import SubmissionLogs
from .configuration.configuration import Config
import os
import csv
from .constants import *
import json


class ResultExporter:
    """Exports submission validation results to CSV format.

    The `ResultExporter` class collects validated submission data and
    exports it to a CSV file with standardized columns for MLPerf
    submission summaries. It handles both performance and power results,
    duplicating rows for power submissions with power-specific metrics.
    """

    def __init__(self, csv_path, config: Config) -> None:
        """Initialize the result exporter.

        Sets up the CSV header columns and prepares for result collection.

        Args:
            csv_path (str): Path to the output CSV file.
            config (Config): Configuration helper for model mappings.
        """
        self.head = [
            "Organization",
            "Availability",
            "Division",
            "SystemType",
            "SystemName",
            "Platform",
            "Model",
            "MlperfModel",
            "Scenario",
            "Result",
            "Accuracy",
            "number_of_nodes",
            "host_processor_model_name",
            "host_processors_per_node",
            "host_processor_core_count",
            "accelerator_model_name",
            "accelerators_per_node",
            "Location",
            "framework",
            "operating_system",
            "notes",
            "compliance",
            "errors",
            "version",
            "inferred",
            "has_power",
            "Units",
            "weight_data_types",
        ]
        self.rows = []
        self.csv_path = csv_path
        self.config = config

    def add_result(self, submission_logs: SubmissionLogs):
        """Add a validated submission result to the export queue.

        Extracts relevant fields from submission logs and system JSON,
        formats them into a CSV row, and appends to the rows list. For
        power submissions, adds an additional row with power metrics.

        Args:
            submission_logs (SubmissionLogs): Validated submission data
                and metadata.
        """
        row = {key: "" for key in self.head}
        row["Organization"] = submission_logs.loader_data["submitter"]
        row["Availability"] = submission_logs.system_json["status"]
        row["Division"] = submission_logs.loader_data["division"]
        row["SystemType"] = submission_logs.system_json["system_type"]
        row["SystemName"] = submission_logs.system_json["system_name"]
        row["Platform"] = submission_logs.loader_data["system"]
        row["Model"] = submission_logs.loader_data["benchmark"]
        row["MlperfModel"] = self.config.get_mlperf_model(
            row["Model"], submission_logs.loader_data.get("model_mapping", {}))
        row["Scenario"] = submission_logs.loader_data["scenario"]
        row["Result"] = submission_logs.loader_data["performance_metric"]
        row["Accuracy"] = json.dumps(
            submission_logs.loader_data["accuracy_metrics"]).replace(
            ",",
            " ").replace(
            '"',
            "").replace(
                "{",
                "").replace(
                    "}",
            "").strip()
        row["number_of_nodes"] = submission_logs.system_json["number_of_nodes"]
        row["host_processor_model_name"] = submission_logs.system_json["host_processor_model_name"]
        row["host_processors_per_node"] = submission_logs.system_json["host_processors_per_node"]
        row["host_processor_core_count"] = submission_logs.system_json["host_processor_core_count"]
        row["accelerator_model_name"] = submission_logs.system_json["accelerator_model_name"]
        row["accelerators_per_node"] = submission_logs.system_json["accelerators_per_node"]
        row["Location"] = os.path.dirname(
            submission_logs.loader_data["perf_path"])
        row["framework"] = submission_logs.system_json["framework"]
        row["operating_system"] = submission_logs.system_json["operating_system"]
        notes = submission_logs.system_json.get("hw_notes", "")
        if submission_logs.system_json.get("sw_notes"):
            notes = notes + ". " if notes else ""
            notes = notes + submission_logs.system_json.get("sw_notes")
        row["notes"] = notes
        row["compliance"] = submission_logs.loader_data["division"]  # TODO
        row["errors"] = 0
        row["version"] = self.config.version
        row["inferred"] = 1 if row["Scenario"] != submission_logs.performance_log["effective_scenario"] and (
            submission_logs.performance_log["effective_scenario"], row["Scenario"]) != ("server", "interactive") else 0
        row["has_power"] = os.path.exists(
            submission_logs.loader_data["power_dir_path"])
        unit = SPECIAL_UNIT_DICT.get(
            row["MlperfModel"], UNIT_DICT).get(
            row["Scenario"], UNIT_DICT[row["Scenario"]]
        )
        row["Units"] = unit
        row["weight_data_types"] = submission_logs.measurements_json["weight_data_types"]
        self.rows.append(row.copy())
        if row["has_power"]:
            row["Result"] = submission_logs.loader_data["power_metric"]
            power_unit = POWER_UNIT_DICT[row["Scenario"]]
            row["Units"] = power_unit
            self.rows.append(row.copy())

    def export_row(self, row: dict):
        """Write a single result row to the CSV file.

        Formats the row dictionary into a quoted CSV line and appends it
        to the output file.

        Args:
            row (dict): Result row data keyed by column headers.
        """
        values = [f'"{row.get(key, "")}"' for key in self.head]
        csv_row = ",".join(values) + "\n"
        with open(self.csv_path, "+a") as csv:
            csv.write(csv_row)

    def export(self):
        """Export all accumulated results to the CSV file.

        Writes the header row first, then iterates through all collected
        rows, exporting each one.
        """
        csv_header = ",".join(self.head) + "\n"
        with open(self.csv_path, "w") as csv:
            csv.write(csv_header)
        for row in self.rows:
            self.export_row(row)
