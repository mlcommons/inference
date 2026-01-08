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

The dataset has been uploaded the MLC S3 bucket, instructions for how to pull the safe dataset can be found at this website: https://inference.mlcommons-storage.org/index.html. This is what your dataset path should look like:\
coco_safe/\
├── annotations/\
&nbsp;&nbsp;&nbsp;&nbsp;instances_val2017_safe.json\
└── val2017_safe/\
    &nbsp;&nbsp;&nbsp;&nbsp;1525 images\
coco_safe.checksums

## How to run yolo_loadgen.py
Examples usage:  
Perf run  
`python yolo_loadgen.py --model {MODEL_FILE} --dataset-path {DATASET_PATH} --annotation-file {ANNOTATIONS_JSON_FILE} --count {integer value sample count} --output {OUTPUT_PATH} --PerformanceOnly --scenario {Offline, SingleStream, MultiStream}`

Accuracy run  
`python yolo_loadgen.py --model {MODEL_FILE} --dataset-path {DATASET_PATH} --annotation-file {ANNOTATIONS_JSON_FILE} --count {integer value sample count} --output {OUTPUT_PATH} --AccuracyOnly --scenario {Offline, SingleStream, MultiStream}`

Arguments:  
`--model` -> path to YOLO model
`--dataset-path` -> path to dataset images
`--annotation-file` -> path to annotation json file  
`--count` -> number of samples
`--output` -> output directory
`--scenario` -> ["Offline", "SingleStream", "MultiStream"]  
`--PerformanceOnly` -> run performance run  # mutually exclusive with AccuracyOnly
`--AccuracyOnly` -> run accuracy mode


Example output is under inference/vision/classification_and_detection/yolo_result_10232025/ for YOLOv11[N, S, M, L, X]

## How to get mAP accuracy results with yolo_ultra_map.py
`python yolo_ultra_map.py --option {1, 2} --model {MODEL_FILE} --images {DATASET PATH} --data {DATA YAML FILE PATH} --annotations {ANNOTATION JSON FILE PATH} --output_json {OUTPUT JSON FILE PATH}`  
`--option` -> 1 is for the in built YOLO method (does not work) and 2 is for the pycocotools approach that requires the predicitions.json as well as the annotations file.  
`--model` -> model to run test on  
`--images` -> path for the dataset of images  
`--data` -> yaml file that contains the path to the dir of images as well as the labels  
`--annotations` -> path to the annotations json file  
`--output_json` -> output file  

## How to get mAP accuracy results with accuracy-coco.py looking at mlperf_log_accuracy.json
## RECOMMENDED APPROACH for accuracy results
`python /inference/vision/classification_and_detection/tools/accuracy-coco.py --mlperf-accuracy-file {PATH TO mlperf_log_accuracy.json} --coco-dir {DATASET_PATH} --verbose`

## Method in which accuracy will be computed using mlperf_log_accuracy.json
1. LoadGen runs → creates mlperf_log_accuracy.json (hex-encoded)
2. yolo_loadgen.py calls validate_accuracy_requirement()
3. Passes mlperf_log_accuracy.json to yolo_ultra_map.py
4. yolo_ultra_map.py:
   - Detects it's a MLPerf log
   - Decodes hex data → decoded_predictions.json
   - Evaluates with COCO tools
   - Validates against threshold
5. Returns PASS/FAIL



