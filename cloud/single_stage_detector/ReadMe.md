
# 1. Problem
Object detection. This task was modified from mmlperf/training  [repository](https://github.com/mlperf/training/tree/master/single_stage_detector) as to run on larger images (currently 1200x1200), please check the folder for additional details.

# 2. Directions

### Build the docker image for the single stage detection task
```
# Build from Dockerfile
cd inference/single_stage_detector/
docker build -t inference/single_stage_detector .
```

### Steps to download data and model
```
cd inference/single_stage_detector/
source download_dataset.sh
source download_model.sh
```

### Steps to run benchmark.
Using docker image:
```
docker run -v "$(pwd)"/coco:/mlperf/coco  -v "$(pwd)"/pretrained:/mlperf/pretrained -t -i --rm --ipc=host inference/single_stage_detector ./run_and_time.sh
```

# 3. Dataset/Environment
### Publiction/Attribution.
Microsoft COCO: COmmon Objects in Context. 2017.

### Training and test data separation
Mode was trained on 2017 COCO train data set - using mlperf/training/single_stage_detector repo , compute mAP on 2017 COCO val data set.

# 4. Model.
### Publication/Attribution
Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg. SSD: Single Shot MultiBox Detector. In the Proceedings of the European Conference on Computer Vision (ECCV), 2016.

Backbone is ResNet34 pretrained on ILSVRC 2012 (from torchvision). Modifications to the backbone networks: remove conv_5x residual blocks, change the first 3x3 convolution of the conv_4x block from stride 2 to stride1 (this increases the resolution of the feature map to which detector heads are attached), attach all 6 detector heads to the output of the last conv_4x residual block. Thus detections are attached to 38x38, 19x19, 10x10, 5x5, 3x3, and 1x1 feature maps. Convolutions in the detector layers are followed by batch normalization layers.

# 5. Quality.
### Quality metric
Metric is COCO box mAP (averaged over IoU of 0.5:0.95), computed over 2017 COCO val data.

### Quality target
mAP of 0.224

### Evaluation frequency

### Evaluation thoroughness
All the images in COCO 2017 val data set.
