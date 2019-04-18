#!/bin/bash

# convert resnet50 to onnx

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

model=resnet50_v1
tfmodel="$model.pb"
onnxmodel="$model.onnx"
url=https://zenodo.org/record/2535873/files/$tfmodel

if [ ! -r $local ]; then
    wget -o $local -q  $url
fi
python -m tf2onnx.convert --input $tfmodel --output $onnxmodel \
    --fold_const --opset 8 --verbose \
    --inputs-as-nchw input_tensor:0 \
    --inputs input_tensor:0 \
    --outputs ArgMax:0,softmax_tensor:0 
