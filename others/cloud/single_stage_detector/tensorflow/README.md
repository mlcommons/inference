
# 1. Problem
This tensorflow model is trained with tensorflow with resnet-34 backbone.

# 2. Directions

### Environment install
The code is tested on python 3.6.5, tensorflow 1.12.0.
```
# python 3
# tensorflow 1.12.0
# cuda 9.0
# Other python packages
pip install -r requirements.txt
```

### Steps to download data and model
```
cd inference/cloud/single_stage_detector/tensorflow
bash download_dataset.sh
bash download_model.sh
```

### Steps to run benchmark.
```
python eval_tf.py 
```
### pb to tflite tools(support NHWC format pb model)
```
python convert_pb2litte.py -p path_to_pb  -t name_of_tflite
```

# 3. Dataset/Environment
### Publiction/Attribution.
Microsoft COCO: COmmon Objects in Context. 2017.

### Training and test data separation
Model was trained on 2017 COCO train data set - using tensorflow resnet34 ssd train codes, compute mAP on 2017 COCO val data set.

# 4. Model.
### Publication/Attribution
Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg. SSD: Single Shot MultiBox Detector. In the Proceedings of the European Conference on Computer Vision (ECCV), 2016.

Backbone is ResNet34 pretrained on ILSVRC 2012 (from torchvision). Modifications to the backbone networks: remove conv_5x residual blocks, change the first 3x3 convolution of the conv_4x block from stride 2 to stride1 (this increases the resolution of the feature map to which detector heads are attached), attach all 6 detector heads to the output of the last conv_4x residual block. Thus detections are attached to 38x38, 19x19, 10x10, 5x5, 3x3, and 1x1 feature maps. Convolutions in the detector layers are followed by batch normalization layers.

# 5. Quality.
### Quality metric
Metric is COCO box mAP (averaged over IoU of 0.5:0.95), computed over 2017 COCO val data.

### Quality target
mAP of 0.225

### Evaluation frequency

### Evaluation thoroughness
All the images in COCO 2017 val data set.
