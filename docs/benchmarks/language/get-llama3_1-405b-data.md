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
    cm run script --tags=get,dataset,mlperf,inference,llama3,_validation -j
    ```
    
=== "Calibration"

    ### Get Calibration Dataset
    ```
    cm run script --tags=get,dataset,mlperf,inference,llama3,_calibration -j
    ```

## Model
The benchmark implementation run command will automatically download the required model and do the necessary conversions. In case you want to only download the official model, you can use the below commands.

Get the Official MLPerf LLAMA3.1-405b Model

=== "Pytorch"

    ### Pytorch
    ```
    cm run script --tags=get,ml-model,llama3 -j
    ```
  
!!! tip

    Downloading llama3_1-405B model from Hugging Face will prompt you to enter the Hugging Face username and password. Please note that the password required is the [**access token**](https://huggingface.co/settings/tokens) generated for your account. Additionally, ensure that your account has access to the [llama3-405B](https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct) model.

