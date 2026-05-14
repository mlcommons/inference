---
hide:
  - toc
---

# Medical Imaging using 3d-unet (KiTS 2019 kidney tumor segmentation task)

## Dataset

The benchmark implementation run command will automatically download the validation and calibration datasets and do the necessary preprocessing. In case you want to download only the datasets, you can use the below commands.


=== "Unprocessed Dataset"

    === "Validation"
        3d-unet validation run uses the KiTS19 dataset performing [KiTS 2019](https://kits19.grand-challenge.org/) kidney tumor segmentation task

        ### Get Validation Dataset
        ```
        mlcr get,dataset,kits19,_validation -j
        ```

    === "Calibration"

        ### Get Calibration Dataset
        ```
        mlcr get,dataset,kits19,_calibration -j
        ```

=== "Preprocessed Dataset"

    ### Get Preprocessed Validation Dataset
    ```
    mlcr get,dataset,kits19,preprocessed -j
    ```

- `--outdirname=<PATH_TO_DOWNLOAD_KITS19_DATASET>` could be provided to download the dataset to a specific location.


## Model
The benchmark implementation run command will automatically download the required model and do the necessary conversions. In case you want to only download the official model, you can use the below commands.

Get the Official MLPerf 3d-unet Model

=== "Pytorch"

    ### Pytorch
    ```
    mlcr get,ml-model,3d-unet,_pytorch -j
    ```
=== "Onnx"

    ### Onnx
    ```
    mlcr get,ml-model,3d-unet,_onnx -j
    ```
=== "Tensorflow"

    ### Tensorflow
    ```
    mlcr get,ml-model,3d-unet,_tensorflow -j
    ```

- `--outdirname=<PATH_TO_DOWNLOAD_3DUNET_MODEL>` could be provided to download the model to a specific location.
