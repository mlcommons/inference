#!/bin/bash

if [ $# -lt 1 ]; then
    echo "usage: $0 tf|onnxruntime|pytorch|tflite|tvm-onnx|tvm-pytorch|tvm-tflite [resnet50|mobilenet|ssd-mobilenet|ssd-resnet34|retinanet] [cpu|gpu|tpu]"
    exit 1
fi
if [ "x$DATA_DIR" == "x" ]; then
    echo "DATA_DIR not set" && exit 1
fi
if [ "x$MODEL_DIR" == "x" ]; then
    echo "MODEL_DIR not set" && exit 1
fi

# defaults
backend=tf
model=resnet50
device="cpu"

for i in $* ; do
    case $i in
       tf|onnxruntime|tflite|pytorch|tvm-onnx|tvm-pytorch|tvm-tflite|ncnn) backend=$i; shift;;
       cpu|gpu|tpu|rocm) device=$i; shift;;
       gpu) device=gpu; shift;;
       resnet50|mobilenet|ssd-mobilenet|ssd-resnet34|ssd-resnet34-tf|retinanet) model=$i; shift;;
    esac
done

if [ $device == "cpu" ] ; then
    export CUDA_VISIBLE_DEVICES=""
fi

name="$model-$backend"
extra_args=""

#
# tensorflow
#
if [ $name == "resnet50-tf" ] ; then
    model_path="$MODEL_DIR/resnet50_v1.pb"
    profile=resnet50-tf
fi
if [ $name == "mobilenet-tf" ] ; then
    model_path="$MODEL_DIR/mobilenet_v1_1.0_224_frozen.pb"
    profile=mobilenet-tf
fi
if [ $name == "ssd-mobilenet-tf" ] ; then
    model_path="$MODEL_DIR/ssd_mobilenet_v1_coco_2018_01_28.pb"
    profile=ssd-mobilenet-tf
fi
if [ $name == "ssd-resnet34-tf" ] ; then
    model_path="$MODEL_DIR/resnet34_tf.22.1.pb"
    profile=ssd-resnet34-tf
fi

#
# onnxruntime
#
if [ $name == "resnet50-onnxruntime" ] ; then
    model_path="$MODEL_DIR/resnet50_v1.onnx"
    profile=resnet50-onnxruntime
fi
if [ $name == "mobilenet-onnxruntime" ] ; then
    model_path="$MODEL_DIR/mobilenet_v1_1.0_224.onnx"
    profile=mobilenet-onnxruntime
fi
if [ $name == "ssd-mobilenet-onnxruntime" ] ; then
    model_path="$MODEL_DIR/ssd_mobilenet_v1_coco_2018_01_28.onnx"
    profile=ssd-mobilenet-onnxruntime
fi
if [ $name == "ssd-resnet34-onnxruntime" ] ; then
    # use onnx model converted from pytorch
    model_path="$MODEL_DIR/resnet34-ssd1200.onnx"
    profile=ssd-resnet34-onnxruntime
fi
if [ $name == "ssd-resnet34-tf-onnxruntime" ] ; then
    # use onnx model converted from tensorflow
    model_path="$MODEL_DIR/ssd_resnet34_mAP_20.2.onnx"
    profile=ssd-resnet34-onnxruntime-tf
fi
if [ $name == "retinanet-onnxruntime" ] ; then
    model_path="$MODEL_DIR/resnext50_32x4d_fpn.onnx"
    profile=retinanet-onnxruntime
fi

#
# pytorch
#
if [ $name == "resnet50-pytorch" ] ; then
    model_path="$MODEL_DIR/resnet50-19c8e357.pth"
    profile=resnet50-pytorch
    extra_args="$extra_args --backend pytorch"
fi
if [ $name == "mobilenet-pytorch" ] ; then
    model_path="$MODEL_DIR/mobilenet_v1_1.0_224.onnx"
    profile=mobilenet-onnxruntime
    extra_args="$extra_args --backend pytorch"
fi
if [ $name == "ssd-resnet34-pytorch" ] ; then
    model_path="$MODEL_DIR/resnet34-ssd1200.pytorch"
    profile=ssd-resnet34-pytorch
fi
if [ $name == "retinanet-pytorch" ] ; then
    model_path="$MODEL_DIR/resnext50_32x4d_fpn.pth"
    profile=retinanet-pytorch
fi


#
# tflite
#
if [ "$name" = "resnet50-tflite" ]; then
  if [ "$device" = "tpu" ]; then
      model_path="$MODEL_DIR/resnet50_quant_full_mlperf_edgetpu.tflite"
      profile="resnet50-tflite"
      extra_args="$extra_args --backend tflite --device tpu"
  else
      model_path="$MODEL_DIR/resnet50_v1.tflite"
      profile="resnet50-tf"
      extra_args="$extra_args --backend tflite"
  fi
fi

if [ "$name" = "mobilenet-tflite" ]; then
    model_path="$MODEL_DIR/mobilenet_v1_1.0_224.tflite"
    profile="mobilenet-tf"
    extra_args="$extra_args --backend tflite"
fi

#
# TVM with ONNX models
#
if [ $name == "resnet50-tvm-onnx" ] ; then
    model_path="$MODEL_DIR/resnet50_v1.onnx"
    profile=resnet50-onnxruntime
    extra_args="$extra_args --backend tvm"
fi

#
# TVM with PyTorch models
#
if [ $name == "resnet50-tvm-pytorch" ] ; then
    model_path="$MODEL_DIR/resnet50_INT8bit_quantized.pt"
    profile=resnet50-pytorch
    extra_args="$extra_args --backend tvm"
fi

#
# TVM with TFLite models
#
if [ $name == "resnet50-tvm-tflite" ] ; then
    model_path="$MODEL_DIR/resnet50_v1.tflite"
    profile=resnet50-tf
    extra_args="$extra_args --backend tvm"
fi

#
# ncnn
#
if [ $name == "resnet50-ncnn" ] ; then
    model_path="$MODEL_DIR/resnet50_v1"
    profile=resnet50-ncnn
fi
name="$backend-$device/$model"
EXTRA_OPS="$extra_args $EXTRA_OPS"
