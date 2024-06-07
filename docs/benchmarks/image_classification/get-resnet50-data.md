# Image Classification using ResNet50 

## Dataset

The benchmark implementation run command will automatically download the validation and calibration datasets and do the necessary preprocessing. In case you want to download only the datasets, you can use the below commands.

=== "Validation"
    ResNet50 validation run uses the Imagenet 2012 validation dataset consisting of 50,000 images.

    ### Get Validation Dataset
    ```
    cm run script --tags=get,dataset,imagenet,validation -j
    ```
=== "Calibration"
    ResNet50 calibration dataset consist of 500 images selected from the Imagenet 2012 validation dataset. There are 2 alternative options for the calibration dataset.

    ### Get Calibration Dataset Using Option 1
    ```
    cm run script --tags=get,dataset,imagenet,calibration,_mlperf.option1 -j
    ```
    ### Get Calibration Dataset Using Option 2
    ```
    cm run script --tags=get,dataset,imagenet,calibration,_mlperf.option2 -j
    ```

## Model
The benchmark implementation run command will automatically download the required model and do the necessary conversions. In case you want to only download the official model, you can use the below commands.

Get the Official MLPerf ResNet50 Model

=== "Tensorflow"

    ### Tensorflow
    ```
    cm run script --tags=get,ml-model,resnet50,_tensorflow -j
    ```
=== "Onnx"

    ### Onnx
    ```
    cm run script --tags=get,ml-model,resnet50,_onnx -j
    ```

