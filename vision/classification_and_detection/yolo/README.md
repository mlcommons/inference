YOLO README - a working design doc

What does it take to get YOLO fully working?
- Perf comparison local to Ultralyrics
- Perf and accuracy to run with LoadGen
- Calculate and output accuracy with LoadGen run
- Compliance
- Model distribution script
- Dataset processing / distribution script


How to run yolo_loadgen.py
Examples usage:
Perf run
python yolo_loadgen.py --dataset-path {DATASET_PATH} --model {MODEL_FILE} --scenario {Offline, SingleStream, MultiStream} --output {OUTPUT_RESULTS_DIR}

Accuracy run
python yolo_loadgen.py --dataset-path {DATASET_PATH} --model {MODEL_FILE} --scenario {Offline, SingleStream, MultiStream} --accuracy --output {OUTPUT_RESULTS_DIR}

Arguments:
--dataset-path -> path to dataset images
--model -> path to YOLO model
--device -> device # leave as is 
--scenario -> ["Offline", "SingleStream", "MultiStream"]
--accuracy -> run accuracy mode
--count -> number of samples
--output -> output directory

Example output is under inference/vision/classification_and_detection/yolo_result_10232025/ for YOLOv11[N, S, M, L, X]
