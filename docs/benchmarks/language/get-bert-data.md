---
hide:
  - toc
---

# Question Answering using Bert-Large

## Dataset

The benchmark implementation run command will automatically download the validation and calibration datasets and do the necessary preprocessing. In case you want to download only the datasets, you can use the below commands.

=== "Validation"
    BERT validation run uses the SQuAD v1.1 dataset.

    ### Get Validation Dataset
    ```
    mlcr get,dataset,squad,validation -j
    ```

=== "Calibration"

    === "Calibration Set 1"

        ### Get Calibration Dataset
        ```
        mlcr get,dataset,squad,_calib1 -j
        ```
    
    === "Calibration Set 2"

        ### Get Calibration Dataset
        ```
        mlcr get,dataset,squad,_calib2 -j
        ```

- `--outdirname=<PATH_TO_DOWNLOAD_SQUAD_DATASET>` could be provided to download the dataset to a specific location.

## Model
The benchmark implementation run command will automatically download the required model and do the necessary conversions. In case you want to only download the official model, you can use the below commands.

Get the Official MLPerf Bert-Large Model

=== "Pytorch"

    ### Pytorch
    ```
    mlcr get,ml-model,bert-large,_pytorch -j
    ```
=== "Onnx"

    ### Onnx
    ```
    mlcr get,ml-model,bert-large,_onnx -j
    ```
=== "Tensorflow"

    ### Tensorflow
    ```
    mlcr get,ml-model,bert-large,_tensorflow -j
    ```

- `--outdirname=<PATH_TO_DOWNLOAD_BERT_MODEL>` could be provided to download the model to a specific location.
