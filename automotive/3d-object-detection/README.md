# MLPerf™ Inference Benchmark for Graph Neural Network

This is the reference implementation for the MLPerf automotive 3D detection benchmark. The reference uses Pytorch as a backend. Additionally we provide an implementation using ONNX.

## Downloading the dataset and model checkpoints
Contact MLCommons support for accessing the Waymo Open Dataset along with the model checkpoints for the reference implementation. You will need to accept a license agreement and will be given directions to download the data with rclone. You will need to place the kitti_format folder under a directory named waymo. There are four total checkpoints 2 for pytorch and 2 for onnx.

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

### Accuracy run

Add the accuracy flag
```
cd /inference/automotive/3d-object-detection
python main.py --dataset waymo --dataset-path /waymo/kitti_format/ --lidar-path <checkpoint_path>/pp_ep36.pth --segmentor-path <checkpoint_path>/best_deeplabv3plus_resnet50_waymo_os16.pth --accuracy
```

### Run the accuracy checker on the accuracy run log
```
python accuracy_waymo.py --mlperf-accuracy-file <path to accuracy file>/mlperf_log_accuracy.json --waymo-dir /waymo/kitti_format/
```
