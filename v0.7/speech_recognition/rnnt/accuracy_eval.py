#!/usr/bin/env python

import argparse
import array
import json
import sys
import os

from QSL import AudioQSL

sys.path.insert(0, os.path.join(os.getcwd(), "pytorch"))
from helpers import process_evaluation_epoch, __gather_predictions
from parts.manifest import Manifest


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    labels = [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'"]
    qsl = AudioQSL(args.dataset_dir, args.manifest, labels)
    manifest = qsl.manifest
    with open(os.path.join(args.log_dir, "mlperf_log_accuracy.json")) as fh:
        results = json.load(fh)
    hypotheses = []
    references = []
    for result in results:
        hypotheses.append(array.array('q', bytes.fromhex(result["data"])).tolist())
        references.append(manifest[result["qsl_idx"]]["transcript"])
    hypotheses = __gather_predictions([hypotheses], labels=labels)
    references = __gather_predictions([references], labels=labels)
    d = dict(predictions=hypotheses,
             transcripts=references)
    print("Word Error Rate:", process_evaluation_epoch(d))

if __name__ == '__main__':
    main()
