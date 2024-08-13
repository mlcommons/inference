## Dataset

The benchmark implementation run command will automatically download the preprocessed validation and calibration datasets. In case you want to download only the datasets, you can use the below commands.

=== "Validation"
    mixtral-8x7b validation run uses the combined dataset - Open ORCA, GSM8K and MBXP.

    ### Get Validation Dataset
    ```
    cm run script --tags=get,dataset-mixtral,openorca-mbxp-gsm8k-combined -j
    ```

## Model
The benchmark implementation run command will automatically download the required model and do the necessary conversions. In case you want to only download the official model, you can use the below commands.

Get the Official MLPerf MIXTRAL-8x7b Model

=== "Pytorch"

    ### Pytorch
    ```
    cm run script --tags=get,ml-model,mixtral -j
    ```