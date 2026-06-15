---
hide:
  - toc
---

# Reasoning using DeepSeek R1

## Dataset

The benchmark implementation run command will automatically download the validation and calibration datasets and do the necessary preprocessing. In case you want to download only the datasets, you can use the below commands.

=== "Validation"


  ### Get Validation Dataset
  ```
  mlcr get,preprocessed,dataset,deepseek-r1,_validation,_mlc,_r2-downloader --outdirname=<path to download> -j
  ```

    ### Get Calibration Dataset
    ```
    mlcr get,preprocessed,dataset,deepseek-r1,_calibration,_mlc,_rclone --outdirname=<path to download> -j
    ```

## Model
The benchmark implementation run command will automatically download the required model and do the necessary conversions. In case you want to only download the official model, you can use the below commands.

=== "Pytorch"

    === "From MLCOMMONS Storage"

        ### Get the Official MLPerf DeekSeek-R1 model from MLCOMMONS Storage
        ```
        mlcr get,ml-model,deepseek-r1,_r2-downloader,_mlc,_dry-run -j
        ```
