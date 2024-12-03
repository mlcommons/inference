# Text to Image using Stable Diffusion

## Dataset

The benchmark implementation run command will automatically download the validation and calibration datasets and do the necessary preprocessing. In case you want to download only the datasets, you can use the below commands.

=== "Validation"
    Stable Diffusion validation run uses the Coco 2014 dataset.

    ### Get Validation Dataset
    ```
    cm run script --tags=get,dataset,coco2014,_validation -j
    ```

## Model
The benchmark implementation run command will automatically download the required model and do the necessary conversions. In case you want to only download the official model, you can use the below commands.

Get the Official MLPerf Stable Diffusion Model

=== "Pytorch"

    ### Pytorch
    ```
    cm run script --tags=get,ml-model,sdxl,_pytorch -j
    ```

