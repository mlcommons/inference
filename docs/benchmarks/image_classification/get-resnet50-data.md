---
hide:
  - toc
---

# Image Classification using ResNet50 

## Dataset

The benchmark implementation run command will automatically download the validation and calibration datasets and do the necessary preprocessing. In case you want to download only the datasets, you can use the below commands.

=== "Unprocessed"
    === "Validation"
        ResNet50 validation run uses the Imagenet 2012 validation dataset consisting of 50,000 images.

        ### Get Validation Dataset
        ```
        mlcr get,dataset,imagenet,validation,_full -j
        ```
    === "Calibration"
        ResNet50 calibration dataset consist of 500 images selected from the Imagenet 2012 validation dataset. There are 2 alternative options for the calibration dataset.

        ### Get Calibration Dataset Using Option 1
        ```
        mlcr get,dataset,imagenet,calibration,_mlperf.option1 -j
        ```
        ### Get Calibration Dataset Using Option 2
        ```
        mlcr get,dataset,imagenet,calibration,_mlperf.option2 -j
        ```
=== "Preprocessed"
    ### Get ResNet50 preprocessed dataset

    ```
    mlcr get,dataset,image-classification,imagenet,preprocessed,_pytorch,_full-j
    ```

- `--outdirname=<PATH_TO_DOWNLOAD_IMAGENET_DATASET>` could be provided to download the dataset to a specific location.

## Model
The benchmark implementation run command will automatically download the required model and do the necessary conversions. In case you want to only download the official model, you can use the below commands.

Get the Official MLPerf ResNet50 Model

=== "Tensorflow"

    ### Tensorflow
    ```
    mlcr get,ml-model,resnet50,_tensorflow -j
    ```
=== "Onnx"

    ### Onnx
    ```
    mlcr get,ml-model,resnet50,image-classification,_onnx -j
    ```

- `--outdirname=<PATH_TO_DOWNLOAD_RESNET50_MODEL>` could be provided to download the model to a specific location.