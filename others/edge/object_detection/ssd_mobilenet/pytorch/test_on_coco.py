# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import json
import time
import tempfile

import torch
import torchvision
import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from ssd_mobilenet_v1 import get_tf_pretrained_mobilenet_ssd


def pil_to_tensor(image):
    x = np.asarray(image).astype(np.float32)
    x = torch.as_tensor(x).permute(2, 0, 1)
    return x


def evaluate_predictions_on_coco(
    coco_gt, coco_results, json_result_file, imgs_to_evaluate
):

    with open(json_result_file, "w") as f:
        json.dump(coco_results, f)

    coco_dt = coco_gt.loadRes(str(json_result_file)) if coco_results else COCO()

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = coco_eval.params.imgIds[:imgs_to_evaluate]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--dataset-root",
        default="datasets/coco/val2014",
        metavar="PATH",
        help="path to COCO image folder",
    )
    parser.add_argument(
        "--ann-file",
        default="datasets/coco/annotations/instances_minival2014.json",
        metavar="PATH",
        help="path to COCO annotation file",
    )
    parser.add_argument(
        "--device", default="cuda", help="torch.device to use for inference [cpu, cuda]"
    )
    parser.add_argument(
        "--imgs-to-evaluate", default=50, type=int, help="number of images to evaluate"
    )
    parser.add_argument(
        "--weights-file",
        default="ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb",
        help="path to pre-trained weights",
    )

    args = parser.parse_args()

    setup_time_begin = time.time()
    device = torch.device(args.device)

    model = get_tf_pretrained_mobilenet_ssd(args.weights_file)
    model.eval()
    model.to(device)

    dataset = torchvision.datasets.CocoDetection(args.dataset_root, args.ann_file)
    # for reproducibility
    dataset.ids = sorted(dataset.ids)

    setup_time = time.time() - setup_time_begin

    results = []
    test_time_begin = time.time()
    load_time_total = 0
    detect_time_total = 0
    images_processed = 0
    for idx in range(args.imgs_to_evaluate):
        # Load image
        load_time_begin = time.time()
        image, _ = dataset[idx]
        image = pil_to_tensor(image)[None]
        load_time = time.time() - load_time_begin
        load_time_total += load_time

        # Detect image
        detect_time_begin = time.time()
        image = image.to(device)
        with torch.no_grad():
            boxes, labels, scores = model.predict(image)
        if device.type == "cuda":
            torch.cuda.synchronize()
        detect_time = time.time() - detect_time_begin

        # Exclude first image from averaging
        if idx > 0 or args.imgs_to_evaluate == 1:
            detect_time_total += detect_time
            images_processed += 1

        # convert predictions from xyxy to xywh
        boxes[:, 2:] -= boxes[:, :2]

        boxes = boxes.tolist()
        scores = scores.tolist()
        labels = labels.tolist()

        results.extend(
            [
                {
                    "image_id": dataset.ids[idx],
                    "category_id": labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )

    test_time = time.time() - test_time_begin
    detect_avg_time = detect_time_total / images_processed
    load_avg_time = load_time_total / args.imgs_to_evaluate

    with tempfile.NamedTemporaryFile() as f:
        file_path = f.name
        res = evaluate_predictions_on_coco(
            dataset.coco, results, file_path, args.imgs_to_evaluate
        )

    print("Summary:")
    print("-------------------------------")
    print("Setup time {}s".format(setup_time))
    print("All images loaded in {}s".format(load_time_total))
    print("All images detected in {}s".format(detect_time_total))
    print("Average detection time: {}s".format(detect_avg_time))
    print("mAP: {}".format(res.stats[0]))
    print("Recall: {}".format(res.stats[6]))
    print("-------------------------------")


if __name__ == "__main__":
    main()
