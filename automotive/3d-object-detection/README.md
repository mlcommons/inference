## Reference implementation fo automotive 3D detection benchmark

## TODO: Instructions for dataset download after it is uploaded somewhere appropriate

## TODO: Instructions for checkpoints downloads after it is uploaded somewhere appropriate

## Running with docker
```
docker build -t auto_inference -f dockerfile.gpu .

docker run --gpus=all -it -v <directory to inference repo>/inference/:/inference -v <directory to waymo dataset>/waymo:/waymo --rm auto_inference

cd /inference/automotive/3d-object-detection
python main.py --dataset waymo --dataset-path /waymo/kitti_format/ --lidar-path <checkpoint_path>/pp_ep36.pth --segmentor-path <checkpoint_path>/best_deeplabv3plus_resnet50_waymo_os16.pth --mlperf_conf /inference/mlperf.conf
