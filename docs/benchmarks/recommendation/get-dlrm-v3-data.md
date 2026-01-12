---
hide:
  - toc
---

# Recommendation using DLRM v3

## Dataset

The benchmark implementation run(TBD) command will automatically download the validation and calibration datasets and do the necessary preprocessing. In case you want to download only the datasets, you can use the below commands.

=== "Validation"

    ### Get Validation Dataset
    ```
    mlcr get,mlperf,inference,dataset,synthetic-streaming,_r2-downloader,_mlc -j
    ```

- `--outdirname=<PATH_TO_DOWNLOAD_DLRMV3_SYNTHETIC_DATASET>` could be provided to download the dataset to a specific location.

## Model
The benchmark implementation run command(TBD) will automatically download the required model and do the necessary conversions. In case you want to only download the official model, you can use the below commands.

Get the Official MLPerf DLRM v3 Model

=== "Pytorch"

    ### Pytorch
    ```
    mlcr get,ml-model,dlrm-v3,_mlc,_r2-downloader -j
    ```


- `--outdirname=<PATH_TO_DOWNLOAD_DLRM_V3_MODEL>` could be provided to download the model to a specific location.