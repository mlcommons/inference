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
    parser.add_argument("--verbose", action="store_true", help="verbose messages")
    parser.add_argument("--dtype", default="float32", choices=["float32", "int32", "int64"], help="data type of the label")
    args = parser.parse_args()
    return args


dtype_map = {
    "float32": np.float32,
    "int32": np.int32,
    "int64": np.int64
}

def main():
    args = get_args()

    with open(args.mlperf_accuracy_file, "r") as f:
        results = json.load(f)

    seen = set()
    good = 0
    total= 0
    all_results = []
    all_targets = []
    for j in results:
        idx = j['qsl_idx']

        # de-dupe in case loadgen sends the same sample multiple times
        if idx in seen:
            continue
        seen.add(idx)

        # reconstruct label from mlperf accuracy log
        data = np.frombuffer(bytes.fromhex(j['data']), dtype_map[args.dtype])

        # data stores both predictions and targets
        query_length = data.size // 2
        data = data.reshape((query_length, 2))

        # go through the query elements
        for k in range(query_length):
            total += 1

            result = data[k][0]
            all_results.append(result)

            target = data[k][1]
            all_targets.append(target)

            # count correct predictions
            if result.round() == target:
                good += 1
            else:
                if args.verbose:
                    print("{}:{}, expected: {}, found {}".format(idx, k, target, result.round()))

    # compute AUC metric
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
