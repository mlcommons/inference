# MLPerf™ Inference Benchmark for Automotive 3D Object Detection

This is the reference implementation for the MLPerf automotive 3D detection benchmark. The reference uses Pytorch as a backend. Additionally we provide an implementation using ONNX.

| model | accuracy | dataset | model source | precision | notes |
| ---- | ---- | ---- | ---- | ---- | ---- |
| PointPainting | 0.5425 mAP | Waymo Open Dataset | https://github.com/rod409/pp | fp32 | Single-Stream 99.9 percentile |

## Automated command to run the benchmark via MLCFlow

Please see the [new docs site](https://docs.mlcommons.org/inference/benchmarks/automotive/3d_object_detection/pointpainting/) for an automated way to run this benchmark across different available implementations and do an end-to-end submission with or without docker.

You can also do `pip install mlc-scripts` and then use `mlcr` commands for downloading the model and datasets using the commands given in the later sections.

### Download model through MLCFlow Automation

> [!Note]
> By default, the waymo dataset is downloaded from the mlcommons official drive. One has to accept the [MLCommons Waymo Open Dataset EULA](https://waymo.mlcommons.org/) to access the dataset files.

```
mlcr get,ml-model,pointpainting,_r2-downloader,_mlc --outdirname=<path_to_download> -j
```

### Download dataset through MLCFlow Automation

> [!Note]
> By default, the waymo dataset is downloaded from the mlcommons official drive. One has to accept the [MLCommons Waymo Open Dataset EULA](https://waymo.mlcommons.org/) to access the dataset files.

**Includes validation and calibration dataset**
```
mlcr get,dataset,waymo,_r2-downloader,_mlc --outdirname=<path_to_download> -j
```

**Includes only calibration dataset**

```
mlcr get,dataset,waymo,calibration,_r2-downloader,_mlc --outdirname=<path_to_download> -j
```

## Downloading the dataset and model checkpoints
MLCommons hosts the dataset and model checkpoints for download **exclusively by MLCommons Members**. You must first agree to the [confidentiality notice](https://waymo.mlcommons.org) using your organizational email address, then you will receive a link to a directory containing Rclone download instructions. _If you cannot access the form but you are part of a MLCommons Member organization, submit the [MLCommons subscription form](https://mlcommons.org/community/subscribe/) with your organizational email address and [associate a Google account](https://accounts.google.com/SignUpWithoutGmail) with your organizational email address._

Once you download the dataset and model checkpoints, you will need to place the kitti_format folder under a directory named waymo. There are four total checkpoints 2 for pytorch and 2 for onnx.

After downloading, the structure of the data should look like below:

```bash
├── waymo
│   ├── best_deeplabv3plus_resnet50_waymo_os16.pth
│   ├── deeplabv3+.onnx
│   ├── kitti_format
│   │   ├── ImageSets
│   │   ├── painted_waymo_infos_test.pkl
│   │   ├── painted_waymo_infos_train.pkl
│   │   ├── painted_waymo_infos_trainval.pkl
│   │   ├── painted_waymo_infos_val.pkl
│   │   ├── testing
│   │   ├── testing_3d_camera_only_detection
│   │   ├── training
│   │   ├── waymo_infos_test.pkl
│   │   ├── waymo_infos_train.pkl
│   │   ├── waymo_infos_trainval.pkl
│   │   └── waymo_infos_val.pkl
│   ├── pp_ep36.pth
│   ├── pp.onnx
```
Within the training folder is the validation data used for the dataset. Extract all the compressed files in the training folder.

```
cd <your path to waymo>/waymo/kitti_format/training
for f in *.tar.gz; do tar -xzvf "$f"; done
```

## Running with docker
Build the container and mount the inference repo and Waymo dataset directory.
Build the container and mount the inference repo and Waymo dataset directory.
```
docker build -t auto_inference -f dockerfile.gpu .

docker run --gpus=all -it -v <directory to inference repo>/inference/:/inference -v <path to waymo dataset>/waymo:/waymo --rm auto_inference
```
### Run performance mode with Pytorch
```
cd /inference/automotive/3d-object-detection
python main.py --dataset waymo --dataset-path /waymo/kitti_format/ --lidar-path <checkpoint_path>/pp_ep36.pth --segmentor-path <checkpoint_path>/best_deeplabv3plus_resnet50_waymo_os16.pth
```

### Run performance mode with ONNX
```
python main.py --dataset waymo --dataset-path /waymo/kitti_format/ --lidar-path <checkpoint_path>/pp.onnx --segmentor-path <checkpoint_path>/deeplabv3+.onnx --mlperf_conf /inference/mlperf.conf --backend onnx
```
> [!Note]
> The minimum number of queries needed to be run is 6636. To get the best latency, users are encouraged to run a larger number of samples.

### Accuracy run

Add the accuracy flag
```
cd /inference/automotive/3d-object-detection
python main.py --dataset waymo --dataset-path /waymo/kitti_format/ --lidar-path <checkpoint_path>/pp_ep36.pth --segmentor-path <checkpoint_path>/best_deeplabv3plus_resnet50_waymo_os16.pth --accuracy
```
### Evaluate the accuracy through MLCFlow Automation
```bash
mlcr process,mlperf,accuracy,_waymo --result_dir=<Path to directory where files are generated after the benchmark run>
```

Please click [here](https://github.com/mlcommons/inference/blob/master/automotive/3d-object-detection/accuracy_waymo.py) to view the Python script for evaluating accuracy for the Waymo dataset.

### Run the accuracy checker on the accuracy run log
```
python accuracy_waymo.py --mlperf-accuracy-file <path to accuracy file>/mlperf_log_accuracy.json --waymo-dir /waymo/kitti_format/
```

## Automated command for submission generation via MLCFlow

Please see the [new docs site](https://docs.mlcommons.org/inference/submission/) for an automated way to generate submission through MLCFlow. 
