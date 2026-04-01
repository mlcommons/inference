from ultralytics import YOLO
import mlperf_loadgen as lg
import numpy as np
from pathlib import Path
from datetime import datetime
import time
import struct
import sys
import os
import json
import array
import argparse

"""
YOLOv11 LoadGen MLPerf
"""


# Standard YOLO (80 classes) to COCO (91 classes) mapping
COCO_80_TO_91 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
    64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90
]


class Coco:
    def __init__(self, data_path, annotation_file, count=None):
        self.image_list = []
        self.image_ids = []
        self.data_path = data_path

        print(f"Loading official COCO annotations from: {annotation_file}")
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)

        # image order needs to match the JSON for correct qsl_idx mapping
        for img_info in coco_data['images']:
            img_name = img_info['file_name']
            img_path = os.path.join(data_path, img_name)

            if os.path.exists(img_path):
                self.image_list.append(img_path)
                self.image_ids.append(img_info['id'])

            # Stop if we hit the requested count
            if count and len(self.image_list) >= count:
                break

        self.count = len(self.image_list)
        print(f"Loaded {self.count} images in official COCO order.")

    def get_item_loc(self, index):
        return self.image_list[index], self.image_ids[index]


class Runner:
    def __init__(self, model_path, dataset):
        self.model = YOLO(model_path)
        self.ds = dataset

    def enqueue(self, query_samples):
        for qitem in query_samples:
            img_path, img_id = self.ds.get_item_loc(qitem.index)

            # low confidence threshold set to capture enough detection for
            # valid mAP for accuracy runs
            results = self.model.predict(
                img_path, conf=0.001, verbose=False)[0]

            h, w = results.orig_shape
            response_payload = b""

            if len(results.boxes) > 0:
                boxes = results.boxes.xyxy.cpu().numpy()
                scores = results.boxes.conf.cpu().numpy()
                classes = results.boxes.cls.cpu().numpy().astype(int)

                for box, score, cls in zip(boxes, scores, classes):
                    category_id = COCO_80_TO_91[cls]
                    # h, w = results.orig_shape

                    # pack as [qsl_idx, ymin, xmin, ymax, xmax, score, cat_id]
                    # - 7f format is required for MLPerf accuracy scripts
                    response_payload += struct.pack("7f",
                                                    float(qitem.index),
                                                    # ymin, xmin
                                                    float(
                                                        box[1] / h), float(box[0] / w),
                                                    # ymax, xmax
                                                    float(
                                                        box[3] / h), float(box[2] / w),
                                                    float(score),
                                                    float(category_id)
                                                    )

            response_array = array.array('B', response_payload)
            bi = response_array.buffer_info()
            lg.QuerySamplesComplete(
                [lg.QuerySampleResponse(qitem.id, bi[0], bi[1])])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--annotation-file", type=str, required=True)
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of samples to run")
    parser.add_argument("--output", type=str, help="Directory for MLPerf logs")
    parser.add_argument(
        "--user_conf",
        default="user.conf",
        help="user config for user LoadGen settings such as target QPS",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=False,
        default="yolo"
    )
    parser.add_argument(
        "--enable-log-trace",
        action="store_true",
        help="Enable log tracing. This file can become quite large",
    )
    parser.add_argument(
        "--audit-conf",
        type=str,
        default="audit.conf",
        help="audit config for LoadGen settings during compliance runs",
    )
    # mode flags
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--AccuracyOnly", action="store_true")
    mode_group.add_argument("--PerformanceOnly", action="store_true")

    # scenario selection
    parser.add_argument(
        "--scenario",
        type=str,
        choices=[
            "SingleStream",
            "MultiStream",
            "Offline"],
        default="SingleStream")
    args = parser.parse_args()

    # output logs
    if args.output:
        log_path = os.path.abspath(args.output)
    else:
        log_path = os.path.abspath(
            f"logs_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")

    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)

    # initialize dataset
    ds = Coco(args.dataset_path, args.annotation_file, args.count)
    runner = Runner(args.model, ds)

    # MLPerf LoadGen Setup
    def flush_queries(): pass
    sut = lg.ConstructSUT(runner.enqueue, flush_queries)

    # standard edge QSL: min of 500 performance_sample_count and the full
    # dataset count
    qsl = lg.ConstructQSL(ds.count, min(ds.count, 500),
                          lambda x: None, lambda x: None)

    settings = lg.TestSettings()

    # Load user configuration
    user_conf = os.path.abspath(args.user_conf)
    if not os.path.exists(user_conf):
        print("{} not found".format(user_conf))
        sys.exit(1)
    settings.FromConfig(user_conf, args.model_name, args.scenario)

    # scenario configurations
    scenario_map = {
        "SingleStream": lg.TestScenario.SingleStream,
        "MultiStream": lg.TestScenario.MultiStream,
        "Offline": lg.TestScenario.Offline
    }
    settings.scenario = scenario_map[args.scenario]

    # MultiStream samples per query
    if args.scenario == "MultiStream":
        # NOTE: set to 8 for Edge submission
        settings.multi_stream_samples_per_query = 8

    # mode and duration
    if args.AccuracyOnly:
        settings.mode = lg.TestMode.AccuracyOnly
    else:
        settings.mode = lg.TestMode.PerformanceOnly
        # NOTE MLPerf requirement: minimum 10 minute run for performance
        settings.min_duration_ms = 600000

        # NOTE: user configs can override this in submission, this is the reference implementation so purposely left barebones
        # settings.target_qps = ...
        # ...

    # configure logs
    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = log_path
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings
    log_settings.enable_trace = args.enable_log_trace

    print(f"Starting MLPerf run")
    print(f"Scenario: {args.scenario}")
    print(f"{'Accuracy' if args.AccuracyOnly else 'Performance'} run")
    print(f"Log directory: {log_path}")

    try:
        lg.StartTestWithLogSettings(
            sut, qsl, settings, log_settings, args.audit_conf)
        print(f"MLPerf run complete - cleaning up")
    except Exception as e:
        print(f"An error occured during StartTest: {e}")
    finally:
        print(f"cleaning up LoadGen and flushing logs")
        lg.DestroyQSL(qsl)
        lg.DestroySUT(sut)

        del qsl
        del sut

        time.sleep(2)

    # final sanity check for logs
    expected_files = ["mlperf_log_summary.txt", "mlperf_log_detail.txt"]
    if args.AccuracyOnly:
        expected_files.append("mlperf_log_accuracy.json")

    print(f"Checking for logs in {log_path}:")
    for f in expected_files:
        fpath = os.path.join(log_path, f)
        if os.path.exists(fpath):
            print(f" [OK] {f} present ({os.path.getsize(fpath)} bytes)")
        else:
            if os.path.exists(f):
                print(f" [!!] Found {f} in CURRENT DIR instead of output dir!")
            else:
                print(f" [!!] {f} MISSING")


if __name__ == "__main__":
    main()
