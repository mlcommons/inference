"""
Tool to calculate accuracy for loadgen accuracy output found in mlperf_log_accuracy.json
We assume that loadgen's query index is in the same order as
the images in coco's annotations/instances_val2017.json.
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import json
import os

import numpy as np
from waymo import Waymo
from tools.evaluate import do_eval
# pylint: disable=missing-docstring
CLASSES = Waymo.CLASSES
LABEL2CLASSES = {v: k for k, v in CLASSES.items()}


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mlperf-accuracy-file",
        required=True,
        help="path to mlperf_log_accuracy.json")
    parser.add_argument(
        "--waymo-dir",
        required=True,
        help="waymo dataset directory")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="verbose messages")
    parser.add_argument(
        "--output-file",
        default="openimages-results.json",
        help="path to output file")
    parser.add_argument(
        "--use-inv-map",
        action="store_true",
        help="use inverse label map")
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    with open(args.mlperf_accuracy_file, "r") as f:
        results = json.load(f)

    detections = {}
    image_ids = set()
    seen = set()
    no_results = 0

    val_dataset = Waymo(
        data_root=args.waymo_dir,
        split='val',
        painted=True,
        cam_sync=False)

    for j in results:
        idx = j['qsl_idx']
        # de-dupe in case loadgen sends the same image multiple times
        if idx in seen:
            continue
        seen.add(idx)

        # reconstruct from mlperf accuracy log
        # what is written by the benchmark is an array of float32's:
        # id, box[0], box[1], box[2], box[3], score, detection_class
        # note that id is a index into instances_val2017.json, not the actual
        # image_id
        data = np.frombuffer(bytes.fromhex(j['data']), np.float32)

        for i in range(0, len(data), 14):
            dimension = [float(x) for x in data[i:i + 3]]
            location = [float(x) for x in data[i + 3:i + 6]]
            rotation_y = float(data[i + 6])
            bbox = [float(x) for x in data[i + 7:i + 11]]
            label = int(data[i + 11])
            score = float(data[i + 12])
            image_idx = int(data[i + 13])
            if image_idx not in detections:
                detections[image_idx] = {
                    'name': [],
                    'dimensions': [],
                    'location': [],
                    'rotation_y': [],
                    'bbox': [],
                    'score': []
                }
            if dimension[0] > 0:
                detections[image_idx]['name'].append(LABEL2CLASSES[label])
                detections[image_idx]['dimensions'].append(dimension)
                detections[image_idx]['location'].append(location)
                detections[image_idx]['rotation_y'].append(rotation_y)
                detections[image_idx]['bbox'].append(bbox)
                detections[image_idx]['score'].append(score)
            image_ids.add(image_idx)

    with open(args.output_file, "w") as fp:
        json.dump(detections, fp, sort_keys=True, indent=4)
    format_results = {}
    for key in detections.keys():
        format_results[key] = {k: np.array(v)
                               for k, v in detections[key].items()}
    map_stats = do_eval(
        format_results,
        val_dataset.data_infos,
        CLASSES,
        cam_sync=False)
    map_stats['Total'] = np.mean(list(map_stats.values()))

    print(map_stats)
    if args.verbose:
        print("found {} results".format(len(results)))
        print("found {} images".format(len(image_ids)))
        print("found {} images with no results".format(no_results))
        print("ignored {} dupes".format(len(results) - len(seen)))


if __name__ == "__main__":
    main()
