---
hide:
  - toc
---

# 3-D Object Detection using PointPainting

## Dataset

> **Note:** By default, the waymo dataset is downloaded from the mlcommons official drive. One has to accept the [MLCommons Waymo Open Dataset EULA](https://waymo.mlcommons.org/) to access the dataset files. 

The benchmark implementation run command will automatically download the preprocessed dataset. In case you want to download only the datasets, you can use the below commands.

=== "Validation"

    ### Get Validation and Calibration Dataset
    ```
    mlcr get,dataset,waymo,_r2-downloader,_mlc -j
    ```
    
=== "Calibration"

    ### Get Calibration Dataset only
    ```
    mlcr get,dataset,waymo,calibration,_r2-downloader,_mlc -j
    ```

- `--outdirname=<PATH_TO_DOWNLOAD_WAYMO_DATASET>` could be provided to download the dataset to a specific location.

## Model
> **Note:** By default, the PointPainting is downloaded from the mlcommons official drive. One has to accept the [MLCommons Waymo Open Dataset EULA](https://waymo.mlcommons.org/) to access the model files. 

The benchmark implementation run command will automatically download the model. In case you want to download only the PointPainting model, you can use the below command.

```bash
mlcr get,ml-model,pointpainting,_r2-downloader,_mlc -j
```

- `--outdirname=<PATH_TO_DOWNLOAD_POINTPAINTING_MODEL>` could be provided to download the model files to a specific location.
