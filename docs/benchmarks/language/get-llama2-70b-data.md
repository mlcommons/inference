# Text Summarization using LLAMA2-70b

## Dataset

The benchmark implementation run command will automatically download the validation and calibration datasets and do the necessary preprocessing. In case you want to download only the datasets, you can use the below commands.

=== "Validation"
    LLAMA2-70b validation run uses the Open ORCA dataset.

    ### Get Validation Dataset
    ```
    cm run script --tags=get,dataset,openorca,validation -j
    ```

## Model
The benchmark implementation run command will automatically download the required model and do the necessary conversions. In case you want to only download the official model, you can use the below commands.

Get the Official MLPerf LLAMA2-70b Model

=== "Pytorch"

    ### Pytorch
    ```
    cm run script --tags=get,ml-model,llama2-70b,_pytorch -j
    ```

