# SSD-MobileNet-v1 in PyTorch

This repo implements SSD-MobileNet-v1 in inference mode in PyTorch.
It uses the pre-trained weight files from [the TensorFlow Object Detection ModelZoo](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz).

## Installation instructions

### Requirements
- PyTorch 1.0 or later
- torchvision
- cocoapi
- TensorFlow 1.12 or later (for loading the weights)

### Step-by-step installation with conda

```bash
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do

conda create --name pytorch_ssd_mobilenet
conda activate pytorch_ssd_mobilenet

# this installs the right pip and dependencies for the fresh python
conda install ipython

# coco api dependencies
pip install cython matplotlib

# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for CUDA 9.0 but you can pick whichever you prefer
conda install -c pytorch pytorch torchvision cudatoolkit=9.0

# install pycocotools
cd github
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install TF for decoding the weights
pip install tensorflow
```

## Running it

First, download the pre-trained weights from the TensorFlow ModelZoo
```bash
curl http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz -o ssd_mobilenet_v1_coco_2018_01_28.tar.gz

tar -zxvf ssd_mobilenet_v1_coco_2018_01_28.tar.gz
```

Now that you have dowloaded the weights, you can perform inference it via
```bash
python test_on_coco.py \
    --dataset-root /path/to/coco/imgs \
    --ann-file /path/to/coco/ann_file \
    --imgs-to-evaluate 50 \
    --weights-file ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb
```
which should give as a result
```
2019-04-11 13:47:58.583262: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2019-04-11 13:47:58.613927: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2199815000 Hz
2019-04-11 13:47:58.617907: I tensorflow/compiler/xla/service/service.cc:150] XLA service 0x5638b1a14b60 executing computations on platform Host. Devices:
2019-04-11 13:47:58.617927: I tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device (0): <undefined>, <undefined>
loading annotations into memory...
Done (t=0.51s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.13s).
Accumulating evaluation results...
DONE (t=0.24s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.321
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.440
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.366
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.062
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.201
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.693
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.303
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.330
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.330
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.064
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.202
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.712
Summary:
-------------------------------
Setup time 12.216338157653809s
All images loaded in 0.5750606060028076s
All images detected in 1.1177282333374023s
Average detection time: 0.022810780272191883s
mAP: 0.32111795341882854
Recall: 0.3033025069685554
-------------------------------
```

Here are the results when evaluating on the full 5000 images of COCO val2017
```
2019-04-11 13:41:24.266105: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2019-04-11 13:41:24.297876: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2199815000 Hz
2019-04-11 13:41:24.302272: I tensorflow/compiler/xla/service/service.cc:150] XLA service 0x564bcd1f21a0 executing computations on platform Host. Devices:
2019-04-11 13:41:24.302292: I tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device (0): <undefined>, <undefined>
loading annotations into memory...
Done (t=0.50s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.23s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=10.92s).
Accumulating evaluation results...
DONE (t=1.82s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.231
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.351
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.254
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.018
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.167
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.528
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.209
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.263
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.263
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.022
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.191
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.602
Summary:
-------------------------------
Setup time 11.774341344833374s
All images loaded in 49.872482776641846s
All images detected in 92.88805389404297s
Average detection time: 0.018581327044217437s
mAP: 0.2312394616636774
Recall: 0.20853911034074307
-------------------------------
```
