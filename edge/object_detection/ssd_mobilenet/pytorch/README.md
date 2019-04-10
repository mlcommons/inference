# SSD-MobileNet-v1 in PyTorch

This repo implements SSD-MobileNet-v1 in inference mode in PyTorch.
It uses the pre-trained weight files from [the TensorFlow Object Detection ModelZoo](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz).

## Installation instructions

### Requirements
- PyTorch 1.0 or later
- torchvision
- cocoapi
- TensorFlow 1.12 (for loading the weights)

### Step-by-step installation with conda

```bash
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do

conda create --name pytorch_ssd_mobilenet
conda activate pytorch_ssd_mobilenet

# this installs the right pip and dependencies for the fresh python
conda install ipython

# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for CUDA 9.0
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
2019-03-20 11:34:36.297500: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2019-03-20 11:34:36.325873: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2199815000 Hz
2019-03-20 11:34:36.330036: I tensorflow/compiler/xla/service/service.cc:150] XLA service 0x5590161a41b0 executing computations on platform Host. Devices:
2019-03-20 11:34:36.330057: I tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device (0): <undefined>, <undefined>
loading annotations into memory...
Done (t=0.50s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.18s).
Accumulating evaluation results...
DONE (t=0.18s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.316
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.444
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.357
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.062
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.177
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.691
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.294
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.324
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.324
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.062
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.178
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.709
Summary:
-------------------------------
Setup time 7.618103981018066s
All images loaded in 0.5562152862548828s
All images detected in 1.1915736198425293s
Average detection time: 0.024317828976378148s
mAP: 0.3159874092405704
Recall: 0.2940712479080072
-------------------------------
```

Here are the results when evaluating on the full 5000 images of COCO val2017
```
2019-03-20 11:26:00.922253: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2019-03-20 11:26:00.949812: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2199815000 Hz
2019-03-20 11:26:00.954376: I tensorflow/compiler/xla/service/service.cc:150] XLA service 0x55ad827e71b0 executing computations on platform Host. Devices:
2019-03-20 11:26:00.954409: I tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device (0): <undefined>, <undefined>
loading annotations into memory...
Done (t=0.49s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.21s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=10.37s).
Accumulating evaluation results...
DONE (t=1.93s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.205
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.317
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.226
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.015
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.144
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.483
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.192
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.240
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.240
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.018
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.160
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.569
Summary:
-------------------------------
Setup time 8.944006204605103s
All images loaded in 49.411635398864746s
All images detected in 90.85995388031006s
Average detection time: 0.01817562590124226s
mAP: 0.20468113436884472
Recall: 0.1923766276941554
-------------------------------
```
