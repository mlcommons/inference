#!/bin/bash

# convert ssd_mobilenet_v1_coco to onnx

python -c "import tf2onnx" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "We use tensorflow-onnx to convert tensorflow to onnx."
    echo "See https://github.com/onnx/tensorflow-onnx for details."    
    echo "Install with:"
    echo "pip install tf2onnx"
    echo "or"
    echo "pip install https://github.com/onnx/tensorflow-onnx"
    exit 1
fi

model=ssd_mobilenet_v1_coco_2018_01_28
tfmodel="$model.pb"
onnxmodel="$model.onnx"
python -m tf2onnx.convert --input $tfmodel --output $onnxmodel \
    --fold_const --opset 10 --verbose \
    --inputs image_tensor:0 \
    --outputs num_detections:0,detection_boxes:0,detection_scores:0,detection_classes:0

#    --inputs-as-nchw image_tensor:0 
