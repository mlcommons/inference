#! /usr/bin/env python3
# Copyright 2018 The MLPerf Authors. All Rights Reserved.
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
import argparse
import json
import sys

import numpy as np


def hex_to_array(hex_data):
    """Convert hex string to numpy float32 array."""
    return np.frombuffer(bytes.fromhex(hex_data), np.float32)


def parse_sample(data):
    """
    Parse a sample's data array into its components.

    Format: [ts_idx, query_idx, predictions..., labels..., weights..., candidate_size]

    Returns:
        tuple: (ts_idx, query_idx, predictions, labels, weights)
    """
    num_candidates = int(data[-1])
    assert len(data) == 3 + num_candidates * 3

    ts_idx = int(data[0])
    query_idx = int(data[1])
    predictions = data[2: 2 + num_candidates]
    labels = data[2 + num_candidates: 2 + num_candidates * 2]
    weights = data[2 + num_candidates * 2: 2 + num_candidates * 3]

    return ts_idx, query_idx, predictions, labels, weights


def parse_sample_perf(data):
    """
    Parse a performance sample's data array into its components.

    Format: [ts_idx, query_idx, predictions...]

    Returns:
        tuple: (ts_idx, query_idx, predictions)
    """
    ts_idx = int(data[0])
    query_idx = int(data[1])
    predictions = data[2:]
    return ts_idx, query_idx, predictions


def compute_ne(predictions, labels, weights):
    """
    Compute Normalized Entropy (NE) for a single sample.

    NE = cross_entropy / baseline_entropy
    where baseline_entropy = -p*log(p) - (1-p)*log(1-p) with p = weighted mean of labels
    """
    eps = 1e-7
    predictions = np.clip(predictions, eps, 1 - eps)
    labels = np.clip(labels, eps, 1 - eps)

    total_weight = np.sum(weights)
    if total_weight == 0:
        return 0.0

    cross_entropy = -np.sum(
        weights * (labels * np.log(predictions) +
                   (1 - labels) * np.log(1 - predictions))
    )

    p = np.sum(weights * labels) / total_weight
    p = np.clip(p, eps, 1 - eps)
    baseline_entropy = -p * np.log(p) - (1 - p) * np.log(1 - p)

    if baseline_entropy == 0:
        return 0.0

    ne = cross_entropy / (baseline_entropy * total_weight)
    return ne


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference_accuracy",
        "-r",
        help="Specifies the path to the accuracy log from a submission/accuracy run.",
        default="",
    )
    parser.add_argument(
        "--test_accuracy",
        "-t",
        help="Specifies the path to the accuracy log from a performance run with accuracy log sampling enabled.",
        default="",
    )
    parser.add_argument(
        "--tolerance",
        default=0.001,
        type=float,
        help="Relative tolerance for NE comparison (default: 0.001 = 0.1%% difference allowed)",
    )
    return parser.parse_args()


def run(acc_log, perf_log, tolerance):
    """
    Run verification by matching samples on (ts_idx, query_idx) and comparing NE values.

    Samples are matched by their (ts_idx, query_idx) pair. For matched samples,
    the NE values must be within the specified relative tolerance.
    """
    with open(acc_log, "r") as f:
        acc_data = json.load(f)
    with open(perf_log, "r") as f:
        perf_data = json.load(f)
    print("Reading accuracy mode results...")
    acc_samples = {}
    for sample in acc_data:
        data = hex_to_array(sample["data"])
        ts_idx, query_idx, predictions, labels, weights = parse_sample(data)
        ne = compute_ne(predictions, labels, weights)
        key = (ts_idx, query_idx)
        acc_samples[key] = {
            "ne": ne,
            "labels": labels,
            "weights": weights,
        }

    print("Reading performance mode results...")
    num_matched = 0
    num_unmatched = 0
    num_ne_mismatch = 0
    for sample in perf_data:
        data = hex_to_array(sample["data"])
        ts_idx, query_idx, predictions = parse_sample_perf(data)
        key = (ts_idx, query_idx)
        if key not in acc_samples:
            num_unmatched += 1
            continue

        acc_ne = acc_samples[key]["ne"]
        num_matched += 1

        labels = acc_samples[key]["labels"]
        weights = acc_samples[key]["weights"]
        perf_ne = compute_ne(predictions, labels, weights)

        if acc_ne == 0 and perf_ne == 0:
            continue

        if acc_ne == 0 or perf_ne == 0:
            num_ne_mismatch += 1
            print(
                f"  NE mismatch at {key}: acc_ne={acc_ne}, perf_ne={perf_ne}")
            continue

        relative_diff = abs(perf_ne - acc_ne) / abs(acc_ne)
        if relative_diff > tolerance:
            num_ne_mismatch += 1
            print(
                f"  NE mismatch at {key}: acc_ne={acc_ne:.6f}, perf_ne={perf_ne:.6f}, "
                f"diff={relative_diff * 100:.4f}%"
            )

    print(f"\nnum_acc_log_entries = {len(acc_data)}")
    print(f"num_perf_log_entries = {len(perf_data)}")
    print(f"num_matched = {num_matched}")
    print(f"num_unmatched = {num_unmatched}")
    print(f"num_ne_mismatch = {num_ne_mismatch}")
    print(f"tolerance = {tolerance * 100:.2f}%")

    if num_ne_mismatch == 0 and num_matched > 0 and num_unmatched == 0:
        print("\nTEST PASS")
        return True
    else:
        print("\nTEST FAIL")
        return False


def main():
    args = parse_args()

    print("Verifying accuracy. This might take a while...")
    acc_log = args.reference_accuracy
    perf_log = args.test_accuracy

    success = run(acc_log, perf_log, args.tolerance)

    print("TEST08 verification complete")
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
