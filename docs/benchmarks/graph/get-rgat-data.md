---
hide:
  - toc
---

# Graph Neural Network using R-GAT 

## Dataset

The benchmark implementation run command will automatically download the validation and calibration datasets and do the necessary preprocessing. In case you want to download only the datasets, you can use the below commands.

=== "Validation"

    === "Full Dataset"
        R-GAT validation run uses the IGBH dataset consisting of 547,306,935 nodes and 5,812,005,639 edges.

        ### Get Full Dataset
        ```
        mlcr get,dataset,igbh,_full -j
        ```

    === "Debug Dataset"
        R-GAT debug run uses the IGBH debug dataset(tiny).

        ### Get Full Dataset
        ```
        mlcr get,dataset,igbh,_debug -j
        ```

=== "Calibration"
    The calibration dataset contains 5000 nodes from the training paper nodes of the IGBH dataset. IGBH `full` dataset would be downloaded for creating calibration dataset. 

    ### Get Calibration Dataset
    ```
    mlcr get,dataset,igbh,_full,_calibration -j
    ```

- `--outdirname=<PATH_TO_DOWNLOAD_IGBH_DATASET>` could be provided to download the dataset to a specific location.

## Model
The benchmark implementation run command will automatically download the required model and do the necessary conversions. In case you want to only download the official model, you can use the below commands.

Get the Official MLPerf R-GAT Model

=== "PyTorch"

    ### PyTorch
    ```
    mlcr get,ml-model,rgat,_r2-downloader,_mlcommons -j
    ```

- `--outdirname=<PATH_TO_DOWNLOAD_RGAT_MODEL>` could be provided to download the model to a specific location.