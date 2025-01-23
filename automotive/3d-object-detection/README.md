## Reference implementation fo automotive 3D detection benchmark

## Dataset and model checkpoints
Contact MLCommons support for accessing the Waymo Open Dataset along with the model checkpoints for the reference implementation. You will need to accept a license agreement and will be given directions to download the data. You will need to place the kitti_format folder under a directory named waymo. There are four total checkpoints 2 for pytorch and 2 for onnx.

## Running with docker
Build the container and mount the inference repo and Waymo dataset directory.
```
docker build -t auto_inference -f dockerfile.gpu .

docker run --gpus=all -it -v <directory to inference repo>/inference/:/inference -v <path to waymo dataset>/waymo:/waymo --rm auto_inference
```
### Run with GPU
```
cd /inference/automotive/3d-object-detection
python main.py --dataset waymo --dataset-path /waymo/kitti_format/ --lidar-path <checkpoint_path>/pp_ep36.pth --segmentor-path <checkpoint_path>/best_deeplabv3plus_resnet50_waymo_os16.pth --mlperf_conf /inference/mlperf.conf
```

### Run with CPU and ONNX
```
python main.py --dataset waymo --dataset-path /waymo/kitti_format/ --lidar-path <checkpoint_path>/pp.onnx --segmentor-path <checkpoint_path>/deeplabv3+.onnx --mlperf_conf /inference/mlperf.conf
```

### Run the accuracy checker
```
python accuracy_waymo.py --mlperf-accuracy-file <path to accuracy file>/mlperf_log_accuracy.json --waymo-dir /waymo/kitti_format/
```
