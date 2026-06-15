---
hide:
  - toc
---

# Text Summarization using GPT-J

## Dataset

The benchmark implementation run command will automatically download the validation and calibration datasets and do the necessary preprocessing. In case you want to download only the datasets, you can use the below commands.

=== "Validation"
    GPT-J validation run uses the CNNDM dataset.

    ### Get Validation Dataset
    ```
    mlcr get,dataset,cnndm,_validation -j
    ```

=== "Calibration"
    GPT-J calibration dataset is extracted from the CNNDM dataset.

    ### Get Validation Dataset
    ```
    mlcr get,dataset,cnndm,_calibration -j
    ```

- `--outdirname=<PATH_TO_DOWNLOAD_CNNDM_DATASET>` could be provided to download the dataset to a specific location.

## Model
The benchmark implementation run command will automatically download the required model and do the necessary conversions. In case you want to only download the official model, you can use the below commands.

Get the Official MLPerf GPT-J Model

=== "Pytorch"

    ### Pytorch
    ```
    mlcr get,ml-model,gptj,_fp32,_pytorch,_r2-downloader -j
    ```

- `--outdirname=<PATH_TO_DOWNLOAD_GPTJ_MODEL>` could be provided to download the model to a specific location.