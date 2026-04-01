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
TEST09 Verification Script

Verifies output token length in performance mode for LLM workloads.
This test logs all samples and verifies the mean output token length
is within the specified bounds (min_output_tokens <= mean <= max_output_tokens).

This prevents cheating by truncating outputs to improve throughput metrics.

The compliance thresholds can be specified via:
1. The audit.config file (test09_min_output_tokens, test09_max_output_tokens) - recommended
2. CLI arguments (--min-output-tokens, --max-output-tokens) - overrides audit.config

Usage:
    python3 run_verification.py \
        -c COMPLIANCE_DIR \
        -o OUTPUT_DIR \
        --tokenizer openai/gpt-oss-120b \
        [--audit-config /path/to/audit.config]
"""

import os
import sys
import shutil
import argparse
import json
from typing import List, Dict, Any, Optional, Tuple

sys.path.append(os.getcwd())


def decode_hex_to_tokens(hex_string: str) -> List[int]:
    """Decode hex-encoded byte array to list of token IDs.

    MLPerf stores token IDs as hex-encoded bytes where each token is a 4-byte
    little-endian integer.

    Args:
        hex_string: Hex-encoded string from MLPerf log

    Returns:
        List of token IDs
    """
    hex_string = hex_string.strip()
    byte_data = bytes.fromhex(hex_string)

    token_ids = []
    for i in range(0, len(byte_data), 4):
        if i + 4 <= len(byte_data):
            token_id = int.from_bytes(
                byte_data[i:i + 4], byteorder='little', signed=True)
            token_ids.append(token_id)

    return token_ids


def parse_mlperf_log(log_path: str) -> List[Dict[str, Any]]:
    """Parse MLPerf accuracy log file.

    Handles multiple formats:
    - JSON array: [{"qsl_idx": 0, ...}, ...]
    - JSONL: one JSON object per line
    - Concatenated JSON: multiple JSON objects on same line

    Args:
        log_path: Path to mlperf_log_accuracy.json

    Returns:
        List of entries with qsl_idx and hex-encoded data
    """
    print(f"Reading MLPerf log: {log_path}")

    entries = []

    # First try to load as a single JSON array
    try:
        with open(log_path, 'r') as f:
            log_data = json.load(f)
        if isinstance(log_data, list):
            print(f"Loaded {len(log_data)} entries as JSON array")
            return log_data
    except json.JSONDecodeError:
        pass

    # Parse line by line (JSONL or concatenated JSON)
    decoder = json.JSONDecoder()
    with open(log_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
                entries.append(entry)
            except json.JSONDecodeError:
                remaining = line
                parsed_count = 0
                while remaining:
                    remaining = remaining.lstrip()
                    if not remaining:
                        break
                    try:
                        obj, end_idx = decoder.raw_decode(remaining)
                        entries.append(obj)
                        remaining = remaining[end_idx:]
                        parsed_count += 1
                    except json.JSONDecodeError:
                        if parsed_count == 0:
                            print(
                                f"Warning: Line {line_num}: Could not parse JSON")
                        break

    print(f"Loaded {len(entries)} entries from MLPerf log")
    return entries


def compute_output_token_lengths(
        entries: List[Dict[str, Any]]) -> Tuple[List[int], float, int, int]:
    """Compute output token lengths from MLPerf log entries.

    Args:
        entries: List of MLPerf log entries with hex-encoded token data

    Returns:
        Tuple of (token_lengths_list, mean_length, min_length, max_length)
    """
    token_lengths = []

    for entry in entries:
        hex_data = entry.get("data", "")
        if hex_data:
            try:
                token_ids = decode_hex_to_tokens(hex_data)
                token_lengths.append(len(token_ids))
            except Exception as e:
                print(
                    f"Warning: Error decoding entry {entry.get('qsl_idx')}: {e}")
                token_lengths.append(0)
        else:
            token_lengths.append(0)

    if not token_lengths:
        return [], 0.0, 0, 0

    mean_length = sum(token_lengths) / len(token_lengths)
    min_length = min(token_lengths)
    max_length = max(token_lengths)

    return token_lengths, mean_length, min_length, max_length


def parse_audit_config(config_path: str) -> Dict[str, Any]:
    """
    Parse audit.config file and extract TEST09-specific settings.

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
                if not line or line.startswith('#'):
                    continue

                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()

                    # Strip inline comments (everything after #)
                    if '#' in value:
                        value = value.split('#', 1)[0].strip()

                    parts = key.split('.')
                    if len(parts) >= 3:
                        setting_name = parts[-1]

                        if setting_name == 'test09_min_output_tokens':
                            try:
                                config['min_output_tokens'] = float(value)
                            except ValueError:
                                print(
                                    f"Warning: Invalid min_output_tokens value: {value}")

                        elif setting_name == 'test09_max_output_tokens':
                            try:
                                config['max_output_tokens'] = float(value)
                            except ValueError:
                                print(
                                    f"Warning: Invalid max_output_tokens value: {value}")

    except Exception as e:
        print(f"Warning: Error parsing audit.config: {e}")

    return config


def main():
    parser = argparse.ArgumentParser(
        description="TEST09: Verify output token length in performance mode for LLM workloads"
    )
    parser.add_argument(
        "--compliance_dir", "-c",
        required=True,
        help="Path to the directory containing the logs from the compliance test run."
    )
    parser.add_argument(
        "--output_dir", "-o",
        required=True,
        help="Path to the output directory where compliance logs will be uploaded from."
    )
    parser.add_argument(
        "--audit-config",
        default=None,
        help="Path to audit.config file containing test09_min_output_tokens and test09_max_output_tokens."
    )
    parser.add_argument(
        "--min-output-tokens",
        type=float,
        default=None,
        help="Minimum mean output tokens required. Overrides audit.config if provided."
    )
    parser.add_argument(
        "--max-output-tokens",
        type=float,
        default=None,
        help="Maximum mean output tokens allowed. Overrides audit.config if provided."
    )

    args = parser.parse_args()

    print("=" * 80)
    print("TEST09: Verify Output Token Length in Performance Mode")
    print("=" * 80)

    compliance_dir = args.compliance_dir
    output_dir = os.path.join(args.output_dir, "TEST09")

    # Verify compliance directory exists
    if not os.path.isdir(compliance_dir):
        print(f"Error: Compliance directory does not exist: {compliance_dir}")
        sys.exit(1)

    accuracy_log = os.path.join(compliance_dir, "mlperf_log_accuracy.json")
    if not os.path.isfile(accuracy_log):
        print(f"Error: Accuracy log not found: {accuracy_log}")
        sys.exit(1)

    # Determine thresholds
    min_output_tokens = args.min_output_tokens
    max_output_tokens = args.max_output_tokens
    audit_config_path = args.audit_config

    # Try to read thresholds from audit.config if provided
    if audit_config_path:
        print(f"Reading audit.config from: {audit_config_path}")
        audit_config = parse_audit_config(audit_config_path)

        if 'min_output_tokens' in audit_config:
            config_min = audit_config['min_output_tokens']
            print(f"Found min_output_tokens in audit.config: {config_min}")
            if min_output_tokens is None:
                min_output_tokens = config_min
            else:
                print(
                    f"CLI min ({min_output_tokens}) overrides audit.config ({config_min})")

        if 'max_output_tokens' in audit_config:
            config_max = audit_config['max_output_tokens']
            print(f"Found max_output_tokens in audit.config: {config_max}")
            if max_output_tokens is None:
                max_output_tokens = config_max
            else:
                print(
                    f"CLI max ({max_output_tokens}) overrides audit.config ({config_max})")

    # Validate we have thresholds
    if min_output_tokens is None or max_output_tokens is None:
        print("Error: Output token thresholds not specified.")
        print("Provide --min-output-tokens and --max-output-tokens, or --audit-config with test09_* fields.")
        sys.exit(1)

    print(f"\nUsing thresholds:")
    print(f"  Min output tokens: {min_output_tokens}")
    print(f"  Max output tokens: {max_output_tokens}")
    print("=" * 80)

    # Parse MLPerf log and compute token lengths
    print("\nParsing MLPerf accuracy log...")
    try:
        entries = parse_mlperf_log(accuracy_log)
    except Exception as e:
        print(f"Error parsing MLPerf log: {e}")
        sys.exit(1)

    if not entries:
        print("Error: No entries found in MLPerf log")
        sys.exit(1)

    print(f"\nComputing output token lengths for {len(entries)} samples...")
    token_lengths, mean_length, min_length, max_length = compute_output_token_lengths(
        entries)

    # Print statistics
    print("\n" + "=" * 80)
    print("Output Token Length Statistics")
    print("=" * 80)
    print(f"Total samples: {len(token_lengths)}")
    print(f"Mean output tokens: {mean_length:.2f}")
    print(f"Min output tokens: {min_length}")
    print(f"Max output tokens: {max_length}")

    # Compute standard deviation
    if token_lengths:
        variance = sum((x - mean_length) **
                       2 for x in token_lengths) / len(token_lengths)
        std_dev = variance ** 0.5
        print(f"Std deviation: {std_dev:.2f}")

    # Check pass/fail
    print("\n" + "=" * 80)
    print("Verification Results")
    print("=" * 80)

    min_check_pass = mean_length >= min_output_tokens
    max_check_pass = mean_length <= max_output_tokens
    overall_pass = min_check_pass and max_check_pass

    print(f"Mean output tokens: {mean_length:.2f}")
    print(
        f"Min threshold: {min_output_tokens} -> {'PASS' if min_check_pass else 'FAIL'}")
    print(
        f"Max threshold: {max_output_tokens} -> {'PASS' if max_check_pass else 'FAIL'}")
    print(f"\nOverall: {'TEST PASS' if overall_pass else 'TEST FAIL'}")

    # Write verification results
    with open("verify_output_len.txt", "w") as f:
        f.write("=" * 80 + "\n")
        f.write("TEST09: Verify Output Token Length in Performance Mode\n")
        f.write("=" * 80 + "\n\n")

        f.write("Output Token Length Statistics\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total samples: {len(token_lengths)}\n")
        f.write(f"Mean output tokens: {mean_length:.2f}\n")
        f.write(f"Min output tokens: {min_length}\n")
        f.write(f"Max output tokens: {max_length}\n")
        if token_lengths:
            f.write(f"Std deviation: {std_dev:.2f}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Verification Results\n")
        f.write("=" * 80 + "\n")
        f.write(f"Mean output tokens: {mean_length:.2f}\n")
        f.write(f"Min threshold: {min_output_tokens}\n")
        f.write(f"Max threshold: {max_output_tokens}\n")
        f.write(f"Min check pass: {min_check_pass}\n")
        f.write(f"Max check pass: {max_check_pass}\n")
        f.write(f"\n{'TEST PASS' if overall_pass else 'TEST FAIL'}\n")

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

    # Copy compliance logs to output directory
    shutil.copy2("verify_output_len.txt", output_dir)

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

    print("\n" + "=" * 80)
    print("TEST09 verification complete")
    print(f"Results written to: {output_dir}")
    print("=" * 80)

    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
