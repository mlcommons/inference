# YOLO README - a working design doc

## Automated command to run the benchmark via MLCFlow

Please see the [new docs site(WIP)]() for an automated way to run this benchmark across different available implementations and do an end-to-end submission with or without docker.

You can also do `pip install mlc-scripts` and then use `mlcr` commands for downloading the model and datasets using the commands given in the later sections.

## What does it take to get YOLO fully working?
- Perf comparison local to Ultralyrics
- Perf and accuracy to run with LoadGen
- Calculate and output accuracy with LoadGen run
- Compliance
- Model distribution script
- Dataset processing / distribution script

## Getting started

### Python virtual environment and installing dependencies

Please ensure you have Python installed, if not please see: https://www.python.org/downloads/.To isolate your development environment and manage dependencies cleanly, use Python’s built-in `venv` module.

#### Steps:
1. **Create a virtual environment**
   ```bash
   python3 -m venv .venv
   ```
2. **Activate the virtual environment**
   macOS/Linux:
   ```bash
   source .venv/bin/activate
   ```
   Windows:
   ```bash
   .venv\Scripts\Activate
   ```
3. Install dependencies from this yolo working dir
   ```bash
   pip install -r requirements.txt
   ```
This is a quick start but for the official documentaion, please refer to: https://docs.python.org/3/library/venv.html


## Dataset processing
The full coco dataset has images that are not compliant with MLC legal rules. In order to run inference and accuracy on the YOLOv11 benchmark with the safe version of the dataset, please execute the following commands:
```bash
python filter_coco_safe_images.py
```
This will create a new folder with just the images that comply with license agreements  
```bash
python create_safe_annotations.py
```
This will create the correct associated annotations file, needed for the mAP accuracy score calcuations.

### Download filtered dataset through MLCFlow Automation

```
mlcr get,dataset,mlperf-inference,yolo-coco2017-filtered,_mlc,_r2-downloader --outdirname=<path_to_download> -j
```

### Download filtered dataset through native method

The dataset has been uploaded the MLC S3 bucket, instructions for how to pull the safe dataset can be found at this website: https://inference.mlcommons-storage.org/index.html. 

This is what your dataset path should look like:

coco_safe/\
├── val2017_safe/\
    &nbsp;&nbsp;&nbsp;&nbsp;1525 images\
└── annotations/\
&nbsp;&nbsp;&nbsp;&nbsp;instances_val2017.json\
coco_safe.checksums

## Model download

### Download model through MLCFlow Automation

```
mlcr get-ml-model-yolov11,_mlc,_r2-downloader --outdirname=<Download path> -j
```

### Download model through native method

Instructions for how to download the model can be found at this website: https://inference.mlcommons-storage.org/index.html

## How to run yolo_loadgen.py
Perf run  
```bash
python yolo_loadgen.py --model {MODEL_FILE} --dataset-path {DATASET_PATH} --annotation-file {ANNOTATIONS_JSON_FILE} --count {integer value sample count} --output {OUTPUT_PATH} --PerformanceOnly --scenario {Offline, SingleStream, MultiStream}
```

Accuracy run  
```bash
python yolo_loadgen.py --model {MODEL_FILE} --dataset-path {DATASET_PATH} --annotation-file {ANNOTATIONS_JSON_FILE} --count {integer value sample count} --output {OUTPUT_PATH} --AccuracyOnly --scenario {Offline, SingleStream, MultiStream}
```

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

## Evaluate accuracy

### Evaluate the accuracy using MLCFlow
You can also evaulate the accuracy from the generated accuracy log by using the following MLC command

```
mlcr run,accuracy,mlperf,_yolo-coco2017-filtered --result_dir=<Path to directory where files are generated after the benchmark run> 
```

### Native method: How to get mAP accuracy results with accuracy-coco.py looking at mlperf_log_accuracy.json
#### RECOMMENDED APPROACH for accuracy results
```bash
python /inference/vision/classification_and_detection/tools/accuracy-coco.py --mlperf-accuracy-file {PATH TO mlperf_log_accuracy.json} --coco-dir {DATASET_PATH} --verbose
```

Here is an example output:
```
(mlc) manpreet@node076:/mnt/data/yolo$ python /mnt/data/yolo/inference/vision/classification_and_detection/tools
/accuracy-coco.py --mlperf-accuracy-file /mnt/data/yolo/yolo_out_converted_jan_7/MultiStream/mlperf_log_accuracy
.json --coco-dir /mnt/data/yolo/coco/val2017_safe --verbose                                                     
loading annotations into memory...                                                                              
Done (t=0.09s)                                                                                                  
creating index...                                                                                               
index created!                                                                                                  
no results: http://images.cocodataset.org/val2017/000000560371.jpg [images.cocodataset.org], idx=539                                     
Loading and preparing results...                                                                                
DONE (t=0.51s)                                                                                                  
creating index...                                                                                               
index created!                                                                                                  
Running per image evaluation...                                                                                 
Evaluate annotation type *bbox*                                                                                 
DONE (t=6.72s).                                                                                                 
Accumulating evaluation results...                                                                              
DONE (t=1.15s).                                                                                                 
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.531                                 
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.697                                 
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.582                                 
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.364                                 
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.587                                 
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.715                                 
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.381                                 
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.632                                 
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.679                                 
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.489                                 
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.735                                 
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.838                                 
mAP=53.103%                                                                                                     
found 1544 results                                                                                              
found 1525 images                                                                                               
found 1 images with no results                                                                                  
ignored 19 dupes   

```
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



