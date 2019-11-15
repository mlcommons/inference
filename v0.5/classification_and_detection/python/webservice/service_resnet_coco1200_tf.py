"""
mlperf inference benchmarking tool
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import time

import numpy as np
from PIL import Image
import dataset
import imagenet
import coco
from backend_tf import BackendTensorflow

NANO_SEC = 1e9
MILLI_SEC = 1000



# pylint: disable=missing-docstring

# the datasets we support
SUPPORTED_DATASETS = {
    "imagenet":
        (imagenet.Imagenet, dataset.pre_process_vgg, dataset.PostProcessCommon(offset=-1),
         {"image_size": [224, 224, 3]}),
    "imagenet_mobilenet":
        (imagenet.Imagenet, dataset.pre_process_mobilenet, dataset.PostProcessArgMax(offset=-1),
         {"image_size": [224, 224, 3]}),
    "coco-300":
        (coco.Coco, dataset.pre_process_coco_mobilenet, coco.PostProcessCoco(),
         {"image_size": [300, 300, 3]}),
    "coco-300-pt":
        (coco.Coco, dataset.pre_process_coco_pt_mobilenet, coco.PostProcessCocoPt(False,0.3),
         {"image_size": [300, 300, 3]}),         
    "coco-1200":
        (coco.Coco, dataset.pre_process_coco_resnet34, coco.PostProcessCoco(),
         {"image_size": [1200, 1200, 3]}),
    "coco-1200-onnx":
        (coco.Coco, dataset.pre_process_coco_resnet34, coco.PostProcessCocoOnnx(),
         {"image_size": [1200, 1200, 3]}),
    "coco-1200-pt":
        (coco.Coco, dataset.pre_process_coco_resnet34, coco.PostProcessCocoPt(True,0.05),
         {"image_size": [1200, 1200, 3],"use_label_map": True}),
    "coco-1200-tf":
        (coco.Coco, dataset.pre_process_coco_resnet34, coco.PostProcessCocoTf(),
         {"image_size": [1200, 1200, 3],"use_label_map": False}),
}

# pre-defined command line options so simplify things. They are used as defaults and can be
# overwritten from command line

SUPPORTED_PROFILES = {
    "default": {
        "inputs": "image_tensor:0",
        "outputs": "num_detections:0,detection_boxes:0,detection_scores:0,detection_classes:0",
        "dataset": "coco-300",
        "backend": "tensorflow",
        "model-name": "ssd-mobilenet",
    },

    # resnet
    "resnet50-tf": {
        "inputs": "input_tensor:0",
        "outputs": "ArgMax:0",
        "dataset": "imagenet",
        "backend": "tensorflow",
        "model-name": "resnet50",
    },
    "resnet50-onnxruntime": {
        "dataset": "imagenet",
        "outputs": "ArgMax:0",
        "backend": "onnxruntime",
        "model-name": "resnet50",
    },

    # mobilenet
    "mobilenet-tf": {
        "inputs": "input:0",
        "outputs": "MobilenetV1/Predictions/Reshape_1:0",
        "dataset": "imagenet_mobilenet",
        "backend": "tensorflow",
        "model-name": "mobilenet",
    },
    "mobilenet-onnxruntime": {
        "dataset": "imagenet_mobilenet",
        "outputs": "MobilenetV1/Predictions/Reshape_1:0",
        "backend": "onnxruntime",
        "model-name": "mobilenet",
    },

    # ssd-mobilenet
    "ssd-mobilenet-tf": {
        "inputs": "image_tensor:0",
        "outputs": "num_detections:0,detection_boxes:0,detection_scores:0,detection_classes:0",
        "dataset": "coco-300",
        "backend": "tensorflow",
        "model-name": "ssd-mobilenet",
    },
    "ssd-mobilenet-pytorch": {
        "inputs": "image",
        "outputs": "bboxes,labels,scores",
        "dataset": "coco-300-pt",
        "backend": "pytorch-native",
        "model-name": "ssd-mobilenet",
    },
    "ssd-mobilenet-onnxruntime": {
        "dataset": "coco-300",
        "outputs": "num_detections:0,detection_boxes:0,detection_scores:0,detection_classes:0",
        "backend": "onnxruntime",
        "data-format": "NHWC",
        "model-name": "ssd-mobilenet",
    },

    # ssd-resnet34
    "ssd-resnet34-tf": {
        "inputs": "image:0",
        "outputs": "detection_bboxes:0,detection_classes:0,detection_scores:0",
        "dataset": "coco-1200-tf",
        "backend": "tensorflow",
        "data-format": "NCHW",
        "model-name": "ssd-resnet34",
    },
    "ssd-resnet34-pytorch": {
        "inputs": "image",
        "outputs": "bboxes,labels,scores",
        "dataset": "coco-1200-pt",
        "backend": "pytorch-native",
        "model-name": "ssd-resnet34",
    },
    "ssd-resnet34-onnxruntime": {
        "dataset": "coco-1200-onnx",
        "inputs": "image",
        "outputs": "bboxes,labels,scores",
        "backend": "onnxruntime",
        "data-format": "NCHW",
        "max-batchsize": 1,
        "model-name": "ssd-resnet34",
    },
    "ssd-resnet34-onnxruntime-tf": {
        "dataset": "coco-1200-tf",
        "inputs": "image:0",
        "outputs": "detection_bboxes:0,detection_classes:0,detection_scores:0",
        "backend": "onnxruntime",
        "data-format": "NHWC",
        "model-name": "ssd-resnet34",
    },
}


# args = ''
# def get_args():
#     """Parse commandline."""
#     #python python/main.py
#     # --profile ssd-mobilenet-tf
#     # --config ../mlperf.conf
#     # --model /model/ssd_mobilenet_v1_coco_2018_01_28.pb
#     # --dataset-path /data/coco-300
#     # --output /xxx/output/tf-gpu/ssd-mobilenet
#     # [--profile ssd-mobilenet-tf --config ../mlperf.conf --model /model/ssd_mobilenet_v1_coco_2018_01_28.pb --dataset-path /data/coco-300 --output /xxx/output/tf-gpu/ssd-mobilenet]
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--profile", default="ssd-mobilenet-tf", help="standard profiles")
#     parser.add_argument("--config", default="../../mlperf.conf", help="mlperf rules config")
#     parser.add_argument("--model", default="F:/MLperf/model/ssd_mobilenet_v1_coco_2018_01_28.pb", help="model file")
#     parser.add_argument("--dataset-path", default="F:/MLperf/data/coco-300/val2017/", help="path to the dataset")
#     parser.add_argument("--output", default="F:/MLperf/output/tf-gpu/ssd-mobilenet", help="test results")
#
#     parser.add_argument("--dataset", choices=SUPPORTED_DATASETS.keys(), help="dataset")
#     parser.add_argument("--dataset-list", help="path to the dataset list")
#     parser.add_argument("--data-format", choices=["NCHW", "NHWC"], help="data format")
#     parser.add_argument("--scenario", default="SingleStream", help="mlperf benchmark scenario, one of xxx")
#     parser.add_argument("--max-batchsize", type=int, help="max batch size in a single inference")
#     parser.add_argument("--inputs", help="model inputs")
#     parser.add_argument("--outputs", help="model outputs")
#     parser.add_argument("--backend", help="runtime to use")
#     parser.add_argument("--model-name", help="name of the mlperf model, ie. resnet50")
#
#     parser.add_argument("--count", default=1,type=int, help="dataset items to use")
#
#     global args
#     args = parser.parse_args()
#
#     # don't use defaults in argparser. Instead we default to a dict, override that with a profile
#     # and take this as default unless command line give
#     defaults = SUPPORTED_PROFILES["default"]
#
#     if args.profile:
#         profile = SUPPORTED_PROFILES[args.profile]
#         defaults.update(profile)
#     for k, v in defaults.items():
#         kc = k.replace("-", "_")
#         if getattr(args, kc) is None:
#             setattr(args, kc, v)
#     if args.inputs:
#         args.inputs = args.inputs.split(",")
#     if args.outputs:
#         args.outputs = args.outputs.split(",")
#
#     print(args)
#
#     return args


def get_backend(backend):
    if backend == "tensorflow":
        from backend_tf import BackendTensorflow
        backend = BackendTensorflow()
    elif backend == "onnxruntime":
        from backend_onnxruntime import BackendOnnxruntime
        backend = BackendOnnxruntime()
    elif backend == "null":
        from backend_null import BackendNull
        backend = BackendNull()
    elif backend == "pytorch":
        from backend_pytorch import BackendPytorch
        backend = BackendPytorch()
    elif backend == "pytorch-native":
        from backend_pytorch_native import BackendPytorchNative
        backend = BackendPytorchNative()      
    elif backend == "tflite":
        from backend_tflite import BackendTflite
        backend = BackendTflite()
    else:
        raise ValueError("unknown backend: " + backend)
    return backend
#
# def main_model():
#     main_model.args = get_args()

backend = BackendTensorflow()
def load_model():
    global backend
    # find backend
    backend = get_backend("tensorflow")

    # load model to backend
    t1 = time.time()
    model = backend.load("/model/resnet34_tf.22.1.pb", inputs=['image:0'], outputs=['detection_bboxes:0','detection_classes:0','detection_scores:0'])
    t2 = time.time()
    print("load model time= %f" % (t2 - t1))



    #
    # make one pass over the dataset to validate accuracy
    #
    # count = ds.get_item_count()

file_list = os.listdir("/data/coco-1200/val2017/")
item_count = len(file_list)

def inference_model():
    global backend
    global file_list
    global item_count
    #file_list = os.listdir("/data/coco-300/val2017/")
    #item_count = len(file_list)
    for _ in range(1):
        t1 = time.time()
        image = Image.open("/data/coco-1200/val2017/"+file_list[np.random.randint(0, item_count)])
        imag_np = np.array(image).reshape(3,1200,1200)
        image_new = imag_np[np.newaxis,:]
        t2 = time.time()
        print("image process time= %f" % (t2 - t1))
        print(image_new.shape)
        t1 = time.time()
        backend.predict({backend.inputs[0]: image_new})
        t2 = time.time()
        print("inference time= %f" % (t2 - t1))


if __name__ =="__main__":
    # get_args()
    load_model()
    inference_model()
