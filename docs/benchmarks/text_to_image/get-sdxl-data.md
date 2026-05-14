---
hide:
  - toc
---

# Text to Image using Stable Diffusion

## Dataset

The benchmark implementation run command will automatically download the validation and calibration datasets and do the necessary preprocessing. In case you want to download only the datasets, you can use the below commands.

=== "Validation"
    Stable Diffusion validation run uses the Coco 2014 dataset.

    ### Get Validation Dataset
    ```
    mlcr get,dataset,coco2014,_validation -j
    ```

=== "Calibration"

    ### Get COCO2014 Calibration Dataset
    ```
    mlcr get,dataset,coco2014,_calibration -j
    ```

- `--outdirname=<PATH_TO_DOWNLOAD_COCO2014_DATASET>` could be provided to download the dataset to a specific location.

## Model
The benchmark implementation run command will automatically download the required model and do the necessary conversions. In case you want to only download the official model, you can use the below commands.

Get the Official MLPerf Stable Diffusion Model

=== "Pytorch"
    === "FP 16"
        ### Pytorch
        ```
        mlcr get,ml-model,sdxl,_pytorch,_fp16,_r2-downloader -j
        ```
    === "FP 32"
        ### Pytorch
        ```
        mlcr get,ml-model,sdxl,_pytorch,_fp32,_r2-downloader -j
        ```

- `--outdirname=<PATH_TO_DOWNLOAD_SDXL_MODEL>` could be provided to download the model to a specific location.
