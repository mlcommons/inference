# YOLO README - a working design doc

## What does it take to get YOLO fully working?
- Perf comparison local to Ultralyrics
- Perf and accuracy to run with LoadGen
- Calculate and output accuracy with LoadGen run
- Compliance
- Model distribution script
- Dataset processing / distribution script

## Dataset processing
The full coco dataset has images that are not compliant with MLC legal rules. In order to run inference and accuracy on the YOLOv11 benchmark with the safe version of the dataset, please execute the following commands:  
`python filter_coco_safe_images.py` to create a new folder with just the images that comply with license agreements  
`python create_safe_annotations.py` to create the correct associated annotations file, needed for the mAP accuracy score calcuations.

## How to run yolo_loadgen.py
Examples usage:  
Perf run  
`python yolo_loadgen.py --dataset-path {DATASET_PATH} --model {MODEL_FILE} --scenario {Offline, SingleStream, MultiStream} --output {OUTPUT_RESULTS_DIR}`

Accuracy run  
`python yolo_loadgen.py --dataset-path {DATASET_PATH} --model {MODEL_FILE} --scenario {Offline, SingleStream, MultiStream} --accuracy --output {OUTPUT_RESULTS_DIR}`

Arguments:  
`--dataset-path` -> path to dataset images  
`--model` -> path to YOLO model  
`--device` -> device # leave as is  
`--scenario` -> ["Offline", "SingleStream", "MultiStream"]  
`--accuracy` -> run accuracy mode  
`--count` -> number of samples  
`--output` -> output directory  

Example output is under inference/vision/classification_and_detection/yolo_result_10232025/ for YOLOv11[N, S, M, L, X]

## How to get mAP accuracy results with yolo_ultra_map.py
`python yolo_ultra_map.py --option {1, 2} --model {MODEL_FILE} --images {DATASET PATH} --data {DATA YAML FILE PATH} --annotations {ANNOTATION JSON FILE PATH} --output_json {OUTPUT JSON FILE PATH}`  
`--option` -> 1 is for the in built YOLO method (does not work) and 2 is for the pycocotools approach that requires the predicitions.json as well as the annotations file.  
`--model` -> model to run test on  
`--images` -> path for the dataset of images  
`--data` -> yaml file that contains the path to the dir of images as well as the labels  
`--annotations` -> path to the annotations json file  
`--output_json` -> output file  
