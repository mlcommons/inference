# 1. Problem
High pixel Object detection. this repo is modified from https://github.com/HiKapok/SSD.TensorFlow.
and use 'py_cls_pred' and 'py_location_pred' nodes to align loc and cls conv output of pytroch version. 

# 2. Directions

### Steps to configure machine
```
1. install python3.6
2. Install pycocotools
   pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAP
3. Install IPython
4. Install tensorflow1.9+
```

### Steps to download data
```
cd dataset/
bash download_dataset.sh
```

### Steps to run benchmark.

## Steps to training

### prepare dataset
```
1. cd dataset
2. more details and steps can be found in README.md
3. generate tfrecords format data.
```
### train
python train_ssd_large.py 
 
### evaluation(support batchsize)
python eval_ssd_large.py

### frozen pb
```
1. python export_graph.py (config data_fortmat in eval_ssd_large.py to specify NHWC or NCHW format pb)
2. bash run_freeze_graph.sh
 ```
### ckpt and pb links
ckpt and pb links at:  https://zenodo.org/record/3712664#.XnB1D8eP63A

### Hyperparameter settings

Hyperparameters are recorded in the `train_*.py` and `eval_*.py` files for each configure.

# 3. Dataset/Environment
### Publiction/Attribution.
Microsoft COCO: COmmon Objects in Context. 2017.

### Training and test data separation
Train on 2017 COCO train data set, compute mAP on 2017 COCO val data set.

# 4. Model.
### Publication/Attribution
Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg. SSD: Single Shot MultiBox Detector. In the Proceedings of the European Conference on Computer Vision (ECCV), 2016.

Backbone is ResNet34 pretrained on ILSVRC. Modifications to the backbone networks: remove conv_5x residual blocks, change the first 3x3 convolution of the conv_4x block from stride 2 to stride 1 (this increases the resolution of the feature map to which detector heads are attached), attach all 6 detector heads to the output of the last conv_4x residual block.
# 5. Quality.
### Quality metric
Metric is COCO box mAP (averaged over IoU of 0.5:0.95), computed over 2017 COCO val data.

### Quality target
mAP of 0.225

### Evaluation frequency

### Evaluation thoroughness
All the images in COCO 2017 val data set.
