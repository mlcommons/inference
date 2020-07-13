"""
Tool to calculate accuracy for loadgen accuracy output found in mlperf_log_accuracy.json
We assume that loadgen's query index is in the same order as samples in the last 24th day
of the pre-processed data set.
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import json

import numpy as np
import sklearn.metrics

# pylint: disable=missing-docstring

def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlperf-accuracy-file", required=True, help="path to mlperf_log_accuracy.json")
    parser.add_argument("--day-23-file", default=None,
        help="path to day_23 file. If present, it is assumed that the accuracy log contains only the prediction, not the ground truth label.")
    parser.add_argument("--aggregation-trace-file", default=None,
        help="path to dlrm_trace_of_aggregated_samples.txt. Only needed if --day-23-file is specified")
    parser.add_argument("--verbose", action="store_true", help="verbose messages")
    parser.add_argument("--dtype", default="float32", choices=["float32", "int32", "int64"], help="data type of the label")
    args = parser.parse_args()
    return args


dtype_map = {
    "float32": np.float32,
    "int32": np.int32,
    "int64": np.int64
}

def get_targets(args, qsl_indices):
    # Parse aggregation trace file to know the sample -> user-item pair mapping
    print("Parsing aggregation trace file...")
    sample_boundaries = [0]
    with open(args.aggregation_trace_file) as f:
        for line in f:
            sample_boundaries.append(sample_boundaries[-1] + int(line.split(", ")[2]))
    assert len(sample_boundaries) == len(qsl_indices) + 1, "Number of samples in trace file does not match number of samples in loadgen accuracy log!"
    # Get all the ground truth labels in the original order in day_23
    print("Parsing ground truth labels from day_23 file...")
    ground_truths = []
    with open(args.day_23_file) as f:
        for line_idx, line in enumerate(f):
            if line_idx >= sample_boundaries[-1]:
                break
            ground_truths.append(int(line.split("\t")[0]))
    # Re-order the ground truth labels according to the qsl indices in the loadgen log.
    print("Re-ordering ground truth labels...")
    targets = []
    for qsl_idx in qsl_indices:
        for i in range(sample_boundaries[qsl_idx], sample_boundaries[qsl_idx + 1]):
            targets.append(ground_truths[i])
    return targets

def main():
    args = get_args()

    # If "--day-23-file" is specified, assume that the accuracy log contains only the prediction, not the ground truth label.
    log_contains_gt = args.day_23_file is None

    if log_contains_gt:
        print("Assuming loadgen accuracy log contains ground truth labels.")
    else:
        print("Assuming loadgen accuracy log does not contain ground truth labels.")

    print("Parsing loadgen accuracy log...")
    with open(args.mlperf_accuracy_file, "r") as f:
        results = json.load(f)

    seen = set()
    good = 0
    total= 0
    all_results = []
    all_targets = []
    qsl_indices = []
    for j in results:
        idx = j['qsl_idx']

        # de-dupe in case loadgen sends the same sample multiple times
        if idx in seen:
            continue
        seen.add(idx)
        qsl_indices.append(idx)

        # reconstruct label from mlperf accuracy log
        data = np.frombuffer(bytes.fromhex(j['data']), dtype_map[args.dtype])

        # data stores both predictions and targets
        output_count = 2 if log_contains_gt else 1
        query_length = data.size // output_count
        data = data.reshape((query_length, output_count))

        # go through the query elements
        for k in range(query_length):
            total += 1

            result = data[k][0]
            all_results.append(result)

            if log_contains_gt:
                target = data[k][1]
                all_targets.append(target)

                # count correct predictions
                if result.round() == target:
                    good += 1
                else:
                    if args.verbose:
                        print("{}:{}, expected: {}, found {}".format(idx, k, target, result.round()))

    if not log_contains_gt:
        all_targets = get_targets(args, qsl_indices)
        for i in range(len(all_targets)):
            if all_results[i].round() == all_targets[i]:
                good += 1

    # compute AUC metric
    print("Calculating AUC metric...")
    all_results = np.array(all_results)
    all_targets = np.array(all_targets)
    roc_auc = sklearn.metrics.roc_auc_score(all_targets, all_results)
    # compute accuracy metric
    acc = good / total
    print("AUC={:.3f}%, accuracy={:.3f}%, good={}, total={}, queries={}".format(100. * roc_auc, 100. * acc, good, total, len(seen)))
    if args.verbose:
        print("found and ignored {} query dupes".format(len(results) - len(seen)))


if __name__ == "__main__":
    main()
