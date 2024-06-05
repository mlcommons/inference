# Recommendation using DLRM v2

## Dataset

The benchmark implementation run command will automatically download the validation and calibration datasets and do the necessary preprocessing. In case you want to download only the datasets, you can use the below commands.

=== "Validation"
    DLRM validation run uses the Criteo dataset (Day 23).

    ### Get Validation Dataset
    ```
    cm run script --tags=get,dataset,criteo,validation -j
    ```
## Model
The benchmark implementation run command will automatically download the required model and do the necessary conversions. In case you want to only download the official model, you can use the below commands.

Get the Official MLPerf DLRM v2 Model

=== "Pytorch"

    ### Pytorch
    ```
    cm run script --tags=get,ml-model,dlrm_v2,_pytorch -j
    ```

