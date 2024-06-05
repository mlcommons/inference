# Text Summarization using GPT-J

## Dataset

The benchmark implementation run command will automatically download the validation and calibration datasets and do the necessary preprocessing. In case you want to download only the datasets, you can use the below commands.

=== "Validation"
    GPT-J validation run uses the CNNDM dataset.

    ### Get Validation Dataset
    ```
    cm run script --tags=get,dataset,cnndm,validation -j
    ```

## Model
The benchmark implementation run command will automatically download the required model and do the necessary conversions. In case you want to only download the official model, you can use the below commands.

Get the Official MLPerf GPT-J Model

=== "Pytorch"

    ### Pytorch
    ```
    cm run script --tags=get,ml-model,gptj,_pytorch -j
    ```
