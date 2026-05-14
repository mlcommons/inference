#!/usr/bin/env python3
# Copyright 2018-2025 The MLPerf Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""
TEST07 Verification Script

Verifies accuracy in performance mode for workloads with separate accuracy/performance
datasets. This test logs all samples and verifies the accuracy score meets a compliance
threshold.

This script is generic and calls an external accuracy evaluation script provided by
the user. The accuracy script should output a line containing the accuracy score
in a parseable format.

The compliance threshold can be specified via:
1. The audit.config file (test07_accuracy_threshold field) - recommended
2. The --accuracy-threshold CLI argument (overrides audit.config)

Usage:
    python3 run_verification.py \
        -c COMPLIANCE_DIR \
        -o OUTPUT_DIR \
        --accuracy-script "python3 /path/to/eval_accuracy.py --mlperf-log {accuracy_log} ..." \
        [--audit-config /path/to/audit.config]
"""

import os
import sys
import shutil
import subprocess
import argparse
import re

sys.path.append(os.getcwd())


def parse_audit_config(config_path):
    """
    Parse audit.config file and extract TEST07-specific settings.

    Returns:
        dict: Parsed configuration values
    """
    config = {}

    if not os.path.isfile(config_path):
        return config

    try:
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue

                # Parse key = value
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()

                    # Extract the setting name (last part of key like
                    # *.*.setting_name)
                    parts = key.split('.')
                    if len(parts) >= 3:
                        setting_name = parts[-1]

                        # Parse test07_accuracy_threshold
                        if setting_name == 'test07_accuracy_threshold':
                            try:
                                config['accuracy_threshold'] = float(value)
                            except ValueError:
                                print(
                                    f"Warning: Invalid threshold value in audit.config: {value}")
    except Exception as e:
        print(f"Warning: Error parsing audit.config: {e}")

    return config


def main():
    parser = argparse.ArgumentParser(
        description="TEST07: Verify accuracy in performance mode for workloads with separate datasets"
    )
    parser.add_argument(
        "--compliance_dir", "-c",
        required=True,
        help="Specifies the path to the directory containing the logs from the compliance test run."
    )
    parser.add_argument(
        "--output_dir", "-o",
        required=True,
        help="Specifies the path to the output directory where compliance logs will be uploaded from, "
             "e.g., inference_results/closed/ORG/compliance/SYSTEM/benchmark/Offline."
    )
    parser.add_argument(
        "--accuracy-script",
        required=True,
        help="Command to run the accuracy evaluation script. Use {accuracy_log} as placeholder for "
             "the path to mlperf_log_accuracy.json. The script should output 'exact_match': <score> "
             "or similar parseable accuracy metric."
    )
    parser.add_argument(
        "--audit-config",
        default=None,
        help="Path to audit.config file containing test07_accuracy_threshold. "
             "If not specified, --accuracy-threshold must be provided."
    )
    parser.add_argument(
        "--accuracy-threshold",
        type=float,
        default=None,
        help="Minimum accuracy score required to pass. Overrides value from audit.config if both provided."
    )
    parser.add_argument(
        "--score-pattern",
        default=r"'exact_match':\s*([\d.]+)",
        help="Regex pattern to extract accuracy score from script output. "
             "Default: \"'exact_match': <score>\" format."
    )

    args = parser.parse_args()

    print("Parsing arguments.")
    compliance_dir = args.compliance_dir
    output_dir = os.path.join(args.output_dir, "TEST07")

    # Verify compliance directory exists
    if not os.path.isdir(compliance_dir):
        print(f"Error: Compliance directory does not exist: {compliance_dir}")
        sys.exit(1)

    accuracy_log = os.path.join(compliance_dir, "mlperf_log_accuracy.json")
    if not os.path.isfile(accuracy_log):
        print(f"Error: Accuracy log not found: {accuracy_log}")
        sys.exit(1)

    # Determine accuracy threshold
    accuracy_threshold = args.accuracy_threshold
    audit_config_path = args.audit_config

    # Try to read threshold from audit.config if provided
    if audit_config_path:
        print(f"Reading audit.config from: {audit_config_path}")
        audit_config = parse_audit_config(audit_config_path)

        if 'accuracy_threshold' in audit_config:
            config_threshold = audit_config['accuracy_threshold']
            print(f"Found threshold in audit.config: {config_threshold}")

            # CLI argument overrides config file
            if accuracy_threshold is None:
                accuracy_threshold = config_threshold
            else:
                print(
                    f"CLI threshold ({accuracy_threshold}) overrides audit.config ({config_threshold})")

    # Validate we have a threshold
    if accuracy_threshold is None:
        print("Error: No accuracy threshold specified.")
        print("Provide --accuracy-threshold or --audit-config with test07_accuracy_threshold field.")
        sys.exit(1)

    print(f"Using accuracy threshold: {accuracy_threshold}")

    # Build accuracy script command with placeholder substitution
    accuracy_command = args.accuracy_script.format(accuracy_log=accuracy_log)

    print(f"\nRunning accuracy evaluation script...")
    print(f"Command: {accuracy_command}")
    print("=" * 80)

    # Run verify accuracy script
    verify_accuracy_output = ""
    try:
        with open("verify_accuracy.txt", "w") as f:
            process = subprocess.Popen(
                accuracy_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                shell=True,
                text=True
            )
            for line in process.stdout:
                print(line, end="")
                f.write(line)
                verify_accuracy_output += line
            process.wait()
    except Exception as e:
        print(f"Exception occurred trying to execute:\n  {accuracy_command}")
        print(f"Error: {e}")

    # Parse accuracy score from output
    accuracy_score = None
    try:
        match = re.search(args.score_pattern, verify_accuracy_output)
        if match:
            accuracy_score = float(match.group(1))
    except Exception as e:
        print(f"Error parsing accuracy score: {e}")

    # Determine pass/fail
    accuracy_pass = False
    if accuracy_score is not None:
        accuracy_pass = accuracy_score >= accuracy_threshold
        print("\n" + "=" * 80)
        print(f"Accuracy score: {accuracy_score}")
        print(f"Accuracy threshold: {accuracy_threshold}")
    else:
        print("\n" + "=" * 80)
        print("Error: Could not parse accuracy score from output.")
        print(f"Expected pattern: {args.score_pattern}")

    # Append pass/fail result to verify_accuracy.txt
    with open("verify_accuracy.txt", "a") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write("TEST07 Verification Summary\n")
        f.write("=" * 80 + "\n")
        if accuracy_score is not None:
            f.write(f"Accuracy score: {accuracy_score}\n")
            f.write(f"Accuracy threshold: {accuracy_threshold}\n")
            f.write(f"Accuracy check pass: {accuracy_pass}\n")
            if accuracy_pass:
                f.write("TEST PASS\n")
            else:
                f.write("TEST FAIL\n")
        else:
            f.write("Error: Could not parse accuracy score\n")
            f.write("TEST FAIL\n")

    # Setup output compliance directory structure
    output_accuracy_dir = os.path.join(output_dir, "accuracy")
    output_performance_dir = os.path.join(output_dir, "performance", "run_1")

    try:
        if not os.path.isdir(output_accuracy_dir):
            os.makedirs(output_accuracy_dir)
    except Exception:
        print(f"Exception occurred trying to create {output_accuracy_dir}")

    try:
        if not os.path.isdir(output_performance_dir):
            os.makedirs(output_performance_dir)
    except Exception:
        print(f"Exception occurred trying to create {output_performance_dir}")

    # Copy compliance logs to output compliance directory
    shutil.copy2("verify_accuracy.txt", output_dir)

    accuracy_file = os.path.join(compliance_dir, "mlperf_log_accuracy.json")
    summary_file = os.path.join(compliance_dir, "mlperf_log_summary.txt")
    detail_file = os.path.join(compliance_dir, "mlperf_log_detail.txt")

    try:
        shutil.copy2(accuracy_file, output_accuracy_dir)
    except Exception:
        print(
            f"Exception occurred trying to copy {accuracy_file} to {output_accuracy_dir}")

    try:
        if os.path.exists(summary_file):
            shutil.copy2(summary_file, output_performance_dir)
    except Exception:
        print(
            f"Exception occurred trying to copy {summary_file} to {output_performance_dir}")

    try:
        if os.path.exists(detail_file):
            shutil.copy2(detail_file, output_performance_dir)
    except Exception:
        print(
            f"Exception occurred trying to copy {detail_file} to {output_performance_dir}")

    print(f"\nAccuracy check pass: {accuracy_pass}")
    print("TEST07 verification complete")


if __name__ == "__main__":
    main()
