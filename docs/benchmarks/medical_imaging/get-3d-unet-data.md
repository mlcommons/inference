# Medical Imaging using 3d-unet (KiTS 2019 kidney tumor segmentation task)

## Dataset

The benchmark implementation run command will automatically download the validation and calibration datasets and do the necessary preprocessing. In case you want to download only the datasets, you can use the below commands.

=== "Validation"
    3d-unet validation run uses the KiTS19 dataset performing [KiTS 2019](https://kits19.grand-challenge.org/) kidney tumor segmentation task

    ### Get Validation Dataset
    ```
    cm run script --tags=get,dataset,kits19,validation -j
    ```

## Model
The benchmark implementation run command will automatically download the required model and do the necessary conversions. In case you want to only download the official model, you can use the below commands.

Get the Official MLPerf 3d-unet Model

=== "Pytorch"

    ### Pytorch
    ```
    cm run script --tags=get,ml-model,3d-unet,_pytorch -j
    ```
=== "Onnx"

    ### Onnx
    ```
    cm run script --tags=get,ml-model,3d-unet,_onnx -j
    ```
=== "Tensorflow"

    ### Tensorflow
    ```
    cm run script --tags=get,ml-model,3d-unet,_tensorflow -j
    ```

