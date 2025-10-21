---
hide:
  - toc
---

# Text Summarization using LLAMA3.1-8b

## Dataset

The benchmark implementation run command will automatically download the validation and calibration datasets and do the necessary preprocessing. In case you want to download only the datasets, you can use the below commands.

=== "Validation"
    
    === "Full dataset (Datacenter)"

        ### Get Validation Dataset
        ```
        mlcr get,dataset,cnndm,_validation,_datacenter,_llama3,_mlc,_r2-downloader --outdirname=<path to download> -j
        ```

    === "5000 samples (Edge)"

        ### Get Validation Dataset
        ```
        mlcr get,dataset,cnndm,_validation,_edge,_llama3,_mlc,_r2-downloader --outdirname=<path to download> -j
        ```

=== "Calibration"
        ```
        
    ### Get Calibration Dataset
    ```
    mlcr get,dataset,cnndm,_calibration,_llama3,_mlc,_r2-downloader --outdirname=<path to download> -j
    ```

- `--outdirname=<PATH_TO_DOWNLOAD_LLAMA3_405B_DATASET>` could be provided to download the dataset to a specific location.

## Model
The benchmark implementation run command will automatically download the required model and do the necessary conversions. In case you want to only download the official model, you can use the below commands.

=== "Pytorch"

    === "From MLCOMMONS Storage"

        > **Note:**  One has to accept the [MLCommons Llama 3.1 License Confidentiality Notice](http://llama3-1.mlcommons.org/) to access the model files in MLCOMMONS Storage. 

        ### Get the Official MLPerf LLAMA3.1-8B model from MLCOMMONS Cloudfare R2
        ```
        mlcr get,ml-model,llama3,_mlc,_8b,_r2-downloader --outdirname=<path to download> -j
        ```

    === "From Hugging Face repo"

        > **Note:** Access to the HuggingFace model could be requested [here](https://ai.meta.com/resources/models-and-libraries/llama-downloads/).

        ### Get model from HuggingFace repo
        ```
        mlcr get,ml-model,llama3,_hf,_meta-llama/Llama-3.1-8B-Instruct --hf_token=<huggingface access token> -j
        ```

- `--outdirname=<PATH_TO_DOWNLOAD_LLAMA3_8B_MODEL>` could be provided to download the model to a specific location.