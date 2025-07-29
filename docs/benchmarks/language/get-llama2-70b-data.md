---
hide:
  - toc
---

# Text Summarization using LLAMA2-70b

## Dataset

The benchmark implementation run command will automatically download the validation and calibration datasets and do the necessary preprocessing. In case you want to download only the datasets, you can use the below commands.

=== "Preprocessed Dataset"

    === "Validation"
        LLAMA2-70b validation run uses the Open ORCA dataset.
    
        ### Get Preprocessed Validation Dataset
        ```
        mlcr get,dataset,preprocessed,openorca,_validation,_mlcommons -j
        ```

    === "Calibration"

        ### Get Preprocessed Calibration dataset
        ```
        mlcr get,dataset,preprocessed,openorca,_calibration -j
        ```

=== "Unprocessed Dataset"

    === "Validation"
        LLAMA2-70b validation run uses the Open ORCA dataset.

        ### Get Unprocessed Validation Dataset
        ```
        mlcr get,dataset,openorca,_validation -j
        ```

    === "Calibration"

        ### Get Unprocessed Validation Dataset
        ```
        mlcr get,dataset,openorca,_validation -j
        ```

- `--outdirname=<PATH_TO_DOWNLOAD_OPENORCA_DATASET>` could be provided to download the dataset to a specific location.

## Model
The benchmark implementation run command will automatically download the required model and do the necessary conversions. In case you want to only download the official model, you can use the below commands.

=== "Pytorch"

    === "From MLCOMMONS Google Drive"

        > **Note:**  One has to accept the [MLCommons Llama 2 License Confidentiality Notice](https://llama2.mlcommons.org/) to access the model files in MLCOMMONS Google Drive. 

        ### Get the Official MLPerf LLAMA2-70B model from MLCOMMONS Google Drive
        ```
        mlcr get,ml-model,llama2-70b,_rclone,_mlc,_70b -j
        ```

    === "From MLCOMMONS Cloudfare R2"

        > **Note:**  One has to accept the [MLCommons Llama 2 License Confidentiality Notice](https://llama2.mlcommons.org/) to access the model files in MLCOMMONS Google Drive. 

        ### Get the Official MLPerf LLAMA2-70B model from MLCOMMONS Cloudfare R2

        ```
        mlcr get,ml-model,llama2-70b,_mlc,_r2-downloader,_70b -j
        ```

    === "From Hugging Face repo"

        > **Note:** Access to the HuggingFace model could be requested [here](https://ai.meta.com/resources/models-and-libraries/llama-downloads/).

        ### Get model from HuggingFace repo
        ```
        mlcr get,ml-model,llama2-70b,_hf --hf_token=<huggingface access token> -j
        ```

- `--outdirname=<PATH_TO_DOWNLOAD_LLAMA2_70B_MODEL>` could be provided to download the model to a specific location.