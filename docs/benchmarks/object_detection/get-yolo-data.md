---
hide:
  - toc
---

# Classification and Detection using Yolov11

## Dataset

The benchmark implementation run command(TBD) will automatically download the validation and calibration datasets and do the necessary preprocessing. In case you want to download only the datasets, you can use the below commands.

=== "Filtered"

    The filtered dataset contains the coco dataset images which are compliant with the MLC legal rules.

    === "Validation"

        ### Get Validation Dataset
        ```
        mlcr get,dataset,mlperf-inference,yolo-coco2017-filtered,_mlc,_r2-downloader -j
        ```

- `--outdirname=<PATH_TO_DOWNLOAD_COCO2017_FILTERED_DATASET>` could be provided to download the dataset to a specific location.

## Model
The benchmark implementation run command will automatically download the required model and do the necessary conversions. In case you want to only download the official model, you can use the below commands.

Get the Official MLPerf Yolov11 Model

TBD

<!-- - `--outdirname=<PATH_TO_DOWNLOAD_RESNET50_MODEL>` could be provided to download the model to a specific location. -->