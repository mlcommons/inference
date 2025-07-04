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
    mlcr get,preprocessed,dataset,deepseek-r1,_validation,_mlc,_rclone --outdirname=<path to download> -j
    ```

=== "Calibration"

    ### Get Calibration Dataset
    ```
    mlcr get,preprocessed,dataset,deepseek-r1,_calibration,_mlc,_rclone --outdirname=<path to download> -j
    ```