---
hide:
  - toc
---

# Text Summarization using LLAMA3.1-405b

## Dataset

The benchmark implementation run command will automatically download the validation and calibration datasets and do the necessary preprocessing. In case you want to download only the datasets, you can use the below commands.

=== "Validation"

    ### Get Validation Dataset
    ```
    mlcr get,dataset,mlperf,inference,llama3,_validation --outdirname=<path to download> -j
    ```
    
=== "Calibration"

    ### Get Calibration Dataset
    ```
    mlcr get,dataset,mlperf,inference,llama3,_calibration --outdirname=<path to download> -j
    ```

## Model
The benchmark implementation run command will automatically download the required model and do the necessary conversions. In case you want to only download the official model, you can use the below commands.

Get the Official MLPerf LLAMA3.1-405b Model

=== "Pytorch"

    ### Pytorch
    ```
    mlcr get,ml-model,llama3 --outdirname=<path to download>  -j
    ```
  
!!! tip

    [Access Request Link](https://llama3-1.mlcommons.org/) for MLCommons members
