---
hide:
  - toc
---

## Dataset

The benchmark implementation run command will automatically download the preprocessed validation and calibration datasets. In case you want to download only the datasets, you can use the below commands.

=== "Validation"
    mixtral-8x7b validation run uses the combined dataset - Open ORCA, GSM8K and MBXP.

    ### Get Validation Dataset
    ```
    mlcr get,dataset-mixtral,openorca-mbxp-gsm8k-combined,_r2-downloader,_validation -j
    ```

=== "Calibration"
    
    ### Get Calibration Dataset
    ```
    mlcr get,dataset-mixtral,openorca-mbxp-gsm8k-combined,_r2-downloader,_calibration -j
    ```

- `--outdirname=<PATH_TO_DOWNLOAD_MIXTRAL_DATASET>` could be provided to download the dataset to a specific location.

## Model
The benchmark implementation run command will automatically download the required model and do the necessary conversions. In case you want to only download the official model, you can use the below commands.

Get the Official MLPerf MIXTRAL-8x7b Model

=== "Pytorch"

    ### Pytorch
    ```
    mlcr get,ml-model,mixtral,_r2-downloader,_mlc -j
    ```

- `--outdirname=<PATH_TO_DOWNLOAD_MIXTRAL_MODEL>` could be provided to download the model to a specific location.