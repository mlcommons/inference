---
hide:
  - toc
---

# Object Detection using Retinanet

## Dataset

The benchmark implementation run command will automatically download the validation and calibration datasets and do the necessary preprocessing. In case you want to download only the datasets, you can use the below commands.

=== "Filtered dataset"

    ### Get COCO-2017 filtered Dataset
    ```
    mlcr get,dataset,mlperf-inference,yolo-coco2017-filtered,_mlc,_r2-downloader -j
    ```

- `--outdirname=<PATH_TO_DOWNLOAD_COCO2017_FILTERED_DATASET>` could be provided to download the dataset to a specific location.


## Model
The benchmark implementation run command will automatically download the required model and do the necessary conversions. In case you want to only download the official model, you can use the below commands.

=== "Pytorch"

    ### Get the Official MLPerf YOLO v11 Model
    ```
    mlcr get-ml-model-yolov11,_mlc,_r2-downloader -j
    ```

- `--outdirname=<PATH_TO_DOWNLOAD_YOLO_MODEL>` could be provided to download the model to a specific location.
