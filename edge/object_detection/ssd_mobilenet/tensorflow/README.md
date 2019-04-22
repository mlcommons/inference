## 1. Problem
This is a tensorflow implementation of SSD-Mobilenetv1 for Object Detection on COCO 2017 test dataset. This code can be executed without any harness.

## 2. Direction
##### Steps to configure the machine
1. Checkout the MLPerf repository
    ```bash
    git clone https://github.intel.com/Intel-MLPerf/inference.git
    ```
2. Install Dependencies
GPU
    ```
    pip install tensorflow-gpu
    ```
    CPU
    ```
    pip install tensorflow
    ```

3. Install dependencies:
    ```
    sudo apt-get install python-tk
    pip install numpy
    sudo apt-get install protobuf-compiler python-pil python-lxml
    pip install --user contextlib2
    pip install pycocotools
    ```
    ##### Steps to download data
- Download COCO 2017 validation dataset by running
    ```
    ./download_dataset.sh
    ```
- Testing images would be downloaded in the folder coco/val2017, 5000 images there. Annotations would also be downloaded with this script.
    ##### Steps to download model
- Download the pre-trained model into the ssd_model folder
    ```
    ./download_model.sh
    ```
- The pre-trained model is from COCO 2017 dataset.
    ##### Steps to run benchmark

    Run the object_detect.py script
    ```
    cd inference/single_stage_detector
    ```
    Note:
    Before running workloads retreive previous results from Output folder.
   - If Output folder is empty please proceed with the next steps directly.
   - If Output folder contains files MlPerf_log_mobilenet_ssd.txt. Then these are results from previous run.
    Save them in another location or delete them. If not then the new results would be appended to these same log file.
- Start evaluation with specific batch size. Run size specifies the number of images to be run on. These form a subset of the COCO 2017 validation set. Default run_size is 5000.
    ```bash
    python object_detect.py --mode=performance --batch_size=4 --run_size=20
    ```
    For default run for MobileNet-SSD which runs the workload for batches 1,2,4,8,16,32,64 and 128 for entire run size of 5000 images and then performs validation run on 5000 images.
  ```
     ./work_mobilenetssd.sh
     ```
    The output log would be availabe after run in Output folder. Log is in a file called MlPerf_log_mobilenet_v1_ssd.txt. Generated log format is shown in example_log.txt and would vary based on run configuration.
## 3. Dataset/Environment
##### Publiction/Attribution.
Microsoft COCO: COmmon Objects in Context. 2017.
The pre-trained model is trained on 2017 COCO train data set. Ccompute mAP on 2017 COCO val data set.
## 4. Model.
##### Publiction/Attribution.

Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg. SSD: Single Shot MultiBox Detector. In the Proceedings of the European Conference on Computer Vision (ECCV), 2016.

Backbone is Mobilenetv1 pretrained model on COCO 2017 dataset.
## 5. Quality.
##### Quality metric
Metric is COCO box mAP (averaged over IoU of 0.5:0.95), computed over 2017 COCO val data.
##### Quality target
mAP of 0.204
##### Evaluation thoroughness
All the images in COCO 2017 val data set.

This code has been tested with Tensorflow 1.13.1.

Author: Srujana Gattupalli

---
No license (express or implied, by estoppel or otherwise) to any intellectual property rights is granted by this document. Any third party materials, including but not limited to pictures, images, video clips, documents, or any other data or datasets, may be subject to third party copyright. You must separately obtain permission from the copyright owner for your usage, and comply with any copyright terms and conditions. We disclaim any liability due to your usage. We do not control or audit third party data. You should review the content, consult other sources, and confirm whether referenced data are accurate.
 
If you own the copyright to any of the materials, please kindly contact us if you would like the reference be removed.



