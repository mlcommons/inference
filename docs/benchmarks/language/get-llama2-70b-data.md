---
hide:
  - toc
---

# Text Summarization using LLAMA2-70b

## Dataset

The benchmark implementation run command will automatically download the validation and calibration datasets and do the necessary preprocessing. In case you want to download only the datasets, you can use the below commands.

=== "Validation"
    LLAMA2-70b validation run uses the Open ORCA dataset.

    ### Get Validation Dataset
    ```
    mlcr get,dataset,openorca,validation -j
    ```

## Model
The benchmark implementation run command will automatically download the required model and do the necessary conversions. In case you want to only download the official model, you can use the below commands.

> **Note:** By default, the Llama2-70B is downloaded from the mlcommons official drive. One has to accept the [MLCommons Llama 2 License Confidentiality Notice](https://llama2.mlcommons.org/) to access the model files. 

Get the Official MLPerf LLAMA2-70b Model

=== "Pytorch"

    ### Pytorch
    ```
    mlcr get,ml-model,llama2-70b,_pytorch -j --outdirname=<My download path>
    ```

- Adding the `_hf` tag along with the run command wil shift the download source to HuggingFace instead of MLCOMMONS GDrive. Access to the HuggingFace model could be requested [here](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)
