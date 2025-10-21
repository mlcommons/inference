---
hide:
  - toc
---

# Object Detection using Retinanet

## Dataset

The benchmark implementation run command will automatically download the validation and calibration datasets and do the necessary preprocessing. In case you want to download only the datasets, you can use the below commands.

=== "Unprocessed"

    === "Validation"
        Retinanet validation run uses the OpenImages v6 MLPerf validation dataset resized to 800x800 and consisting of 24,576 images.

        ### Get Validation Dataset
        ```
        mlcr get,dataset,openimages,original,_validation -j
        ```

    === "Calibration"
        Retinanet calibration dataset consist of 500 images selected from the OpenImages v6 dataset.

        ### Get OpenImages Calibration dataset
        ```
        mlcr get,dataset,openimages,original,_calibration -j
        ```

=== "Preprocessed"

    ### Get Preprocessed OpenImages dataset
    ```
    mlcr get,dataset,object-detection,open-images,openimages,preprocessed,_validation -j 
    ```

- `--outdirname=<PATH_TO_DOWNLOAD_OPENIMAGES_DATASET>` could be provided to download the dataset to a specific location.


## Model
The benchmark implementation run command will automatically download the required model and do the necessary conversions. In case you want to only download the official model, you can use the below commands.

Get the Official MLPerf Retinanet Model

=== "Pytorch"

    ### Pytorch
    ```
    mlcr get,ml-model,retinanet,_pytorch -j
    ```
=== "Onnx"

    ### Onnx
    ```
    mlcr get,ml-model,retinanet,_onnx -j
    ```

- `--outdirname=<PATH_TO_DOWNLOAD_RETINANET_MODEL>` could be provided to download the model to a specific location.
