---
hide:
  - toc
---

# Recommendation using DLRM v2

## Dataset

The benchmark implementation run command will automatically download the validation and calibration datasets and do the necessary preprocessing. In case you want to download only the datasets, you can use the below commands.

=== "Validation"
    DLRM validation run uses the Criteo dataset (Day 23).

    ### Get Validation Dataset
    ```
    mlcr get,preprocessed,dataset,criteo,_r2-downloader,_mlc -j
    ```

- `--outdirname=<PATH_TO_DOWNLOAD_CRITEO_DATASET>` could be provided to download the dataset to a specific location.

## Model
The benchmark implementation run command will automatically download the required model and do the necessary conversions. In case you want to only download the official model, you can use the below commands.

Get the Official MLPerf DLRM v2 Model

=== "Pytorch"

    ### Pytorch
    ```
    mlcr get,ml-model,dlrm,_pytorch,_fp32,_weight_sharded,_r2-downloader -j
    ```


- `--outdirname=<PATH_TO_DOWNLOAD_DLRM_V2_MODEL>` could be provided to download the model to a specific location.