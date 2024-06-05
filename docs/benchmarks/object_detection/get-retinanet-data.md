# Object Detection using Retinanet

## Dataset

The benchmark implementation run command will automatically download the validation and calibration datasets and do the necessary preprocessing. In case you want to download only the datasets, you can use the below commands.

=== "Validation"
    Retinanet validation run uses the OpenImages v6 MLPerf validation dataset resized to 800x800 and consisting of 24,576 images.

    ### Get Validation Dataset
    ```
    cm run script --tags=get,dataset,openimages,_validation -j
    ```
=== "Calibration"
    Retinanet calibration dataset consist of 500 images selected from the OpenImages v6 dataset.

    ```
    cm run script --tags=get,dataset,openimages,_calibration -j
    ```

## Model
The benchmark implementation run command will automatically download the required model and do the necessary conversions. In case you want to only download the official model, you can use the below commands.

Get the Official MLPerf Retinanet Model

=== "Pytorch"

    ### Pytorch
    ```
    cm run script --tags=get,ml-model,retinanet,_pytorch -j
    ```
=== "Onnx"

    ### Onnx
    ```
    cm run script --tags=get,ml-model,retinanet,_onnx -j
    ```

