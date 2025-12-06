"""
YOLOv11 LoadGen MLPerf 
"""

import argparse
import array
import json
import logging
import os 
import sys
import time
from pathlib import Path

import numpy as np
import mlperf_loadgen as lg
from ultralytics import YOLO


# COCO Dataset handler for YOLO
class Coco:
    def __init__(self, data_path, count=None):
        self.image_list = []
        self.image_ids = []
        self.data_path = data_path

        # load from annotations
        annotations_file = Path(data_path).parent.parent / "annotations" / "instances_val2017.json"
        if not annotations_file.exists():
            annotations_file = Path(data_path).parent.parent / "annotations" / "image_info_test-dev2017.json"
       
        if annotations_file.exists():
            with open(annotations_file, 'r') as f:
                coco = json.load(f)
            for img_info in coco['images']:
                img_path = os.path.join(data_path, img_info['file_name'])
                if os.path.exists(img_path):
                    self.image_list.append(img_path)
                    self.image_ids.append(img_info['id'])
        else:
            # load from directory
            for img_path in sorted(Path(data_path).glob("*.jpg")):
                self.image_list.append(str(img_path))
                self.image_ids.append(int(img_path.stem))
       
        self.count = len(self.image_list) if count is None else min(count, len(self.image_list))
        print(f"Loaded {self.count} images")

    def get_item_count(self):
        return self.count

    def get_item_loc(self, idx):
        return self.image_list[idx], self.image_ids[idx]


# post process COCO - convert YOLO outputs to COCO format
class PostProcessCoco:
    def __init__(self):
        self.results = []

    def start(self):
        self.results = []

    def add_results(self, results):
        self.results.extend(results)

    def finalize(self, output_dir):
        if output_dir:
            output_file = os.path.join(output_dir, "predictions.json")
            with open(output_file, 'w') as f:
                json.dump(self.results, f)
            print(f"saved {len(self.results)} predictions to {output_file}")


# YOLO inference engine backend
class BackendYOLO:
    def __init__(self, model_path, device="cuda:0"):
        print(f"loading model: {model_path}")
        self.model = YOLO(model_path)
        self.model.to(device)
        print("model has been loaded")

    def predict(self, img_path):
        results = self.model.predict(
                img_path,
                conf=0.001,
                iou=0.6,
                max_det=300,
                imgsz=640,
                verbose=False
        )
        return results[0]


# runner for orchestration, dataset, model and LoadGen - based on inference/vision/classification_and_detection/python/main.py
class Runner:
    def __init__(self, model, ds, post_proc):
        self.model = model
        self.ds = ds
        self.post_proc = post_proc
        self.take_accuracy = False

    def start_run(self, take_accuracy):
        self.take_accuracy = take_accuracy
        self.post_proc.start()

    # convert YOLO result to COCO format
    def convert_to_coco(self, result, image_id):
        detections = []

        if len(result.boxes) == 0:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box
            detection = {
                "image_id": int(image_id),
                "category_id": int(cls) + 1,  # COCO is 1-indexed
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(score)
            }
            detections.append(detection)

        return detections

    # to process the query samples
    def enqueue(self, query_samples):
        for qitem in query_samples:
            img_path, img_id = self.ds.get_item_loc(qitem.index)

            # run inference
            result = self.model.predict(img_path)

            # convert to COCO format
            detections = self.convert_to_coco(result, img_id)

            # store for accuracy
            if self.take_accuracy:
                self.post_proc.add_results(detections)

            # prepare response for LoadGen
            response_data = json.dumps(detections).encode('utf-8')
            response_array = array.array('B', response_data)
            bi = response_array.buffer_info()

            response = lg.QuerySampleResponse(qitem.id, bi[0], bi[1])
            lg.QuerySamplesComplete([response])


# QSL/SUT LoadGen
class QueueRunner:
    def __init__(self, runner):
        self.runner = runner
        self.qsl = None
        self.sut = None

    def load_query_samples(self, sample_list):
        pass

    def unload_query_samples(self, sample_list):
        pass
   
    def issue_queries(self, query_samples):
        self.runner.enqueue(query_samples)
   
    def flush_queries(self):
        pass

# creaet SUT 
def get_sut(ds, runner):
    queue_runner = QueueRunner(runner)
   
    qsl = lg.ConstructQSL(
        ds.get_item_count(),
        ds.get_item_count(),
        queue_runner.load_query_samples,
        queue_runner.unload_query_samples
    )
    queue_runner.qsl = qsl
   
    sut = lg.ConstructSUT(
        queue_runner.issue_queries,
        queue_runner.flush_queries
    )
    queue_runner.sut = sut

   
    return qsl, sut, queue_runner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True, help="path to dataset images")
    parser.add_argument("--model", required=True, help="path to YOLO model")
    parser.add_argument("--device", default="cuda:0", help="device")
    parser.add_argument("--scenario", default="Offline", choices=["Offline", "SingleStream", "MultiStream"])
    parser.add_argument("--accuracy", action="store_true", help="run accuracy mode")
    parser.add_argument("--count", type=int, help="number of samples")
    parser.add_argument("--output", default="output", help="output directory")
    args = parser.parse_args()
    
    print("=" * 60)
    print("YOLOv11 MLC LoadGen POC")
    print("=" * 60)

    os.makedirs(args.output, exist_ok=True)

    # load dataset
    ds = Coco(args.dataset_path, count=args.count)

    # load model
    backend = BackendYOLO(args.model, device=args.device)

    # create post-processor and runner
    post_proc = PostProcessCoco()
    runner = Runner(backend, ds, post_proc)

    # create QSL and SUT
    qsl, sut, queue_runner = get_sut(ds, runner)

    # configure LoadGen
    settings = lg.TestSettings()
    settings.scenario = getattr(lg.TestScenario, args.scenario)
    settings.mode = lg.TestMode.AccuracyOnly if args.accuracy else lg.TestMode.PerformanceOnly

    if args.scenario == "Offline":
        settings.offline_expected_qps = 100
    elif args.scenario == "SingleStream":
        settings.single_stream_expected_latency_ns = 10000000  # 10ms
    elif args.scenario == "MultiStream":
        settings.multi_stream_samples_per_query = 8
        settings.multi_stream_target_latency_ns = 50000000  # 50ms

    settings.min_duration_ms = 60000
    settings.min_query_count = 100

    # logging - come back to this
    log_settings = lg.LogSettings()
    log_settings.log_output.outdir = args.output
    log_settings.log_output.copy_summary_to_stdout = True
    log_settings.enable_trace = False

    # run
    print(f"\nRunning {args.scenario} scenario...")
    print("-" * 60)

    runner.start_run(args.accuracy)

    start = time.time()
    lg.StartTestWithLogSettings(sut, qsl, settings, log_settings)
    elapsed = time.time() - start

    print("-" * 60)
    print(f"Completed in {elapsed:.2f}s\n")

    # save results
    if args.accuracy:
        post_proc.finalize(args.output)

    print("=" * 60)
    print(f"Results: {args.output}")
    print("=" * 60)

    # destroy qsl and sut cleanup
    lg.DestroyQSL(qsl)
    lg.DestroySUT(sut)


if __name__ == "__main__":
    main()

