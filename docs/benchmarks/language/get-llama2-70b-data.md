---
hide:
  - toc
---

# Text Summarization using LLAMA2-70b

## Dataset

The benchmark implementation run command will automatically download the validation and calibration datasets and do the necessary preprocessing. In case you want to download only the datasets, you can use the below commands.

=== "Nvidia Preprocessed Dataset"

    === "Validation"
        LLAMA2-70b validation run uses the Open ORCA dataset.
    
        ### Get Preprocessed Validation Dataset
        ```
        mlcr get,dataset,preprocessed,openorca,_validation,_mlcommons,_nvidia -j
        ```

    === "Calibration"

        ### Get Preprocessed Calibration dataset
        ```
        mlcr get,dataset,preprocessed,openorca,_calibration,_mlcommons,_nvidia -j
        ```

=== "MLCommons Preprocessed Dataset"

    === "Validation"
        LLAMA2-70b validation run uses the Open ORCA dataset.
    
        ### Get Preprocessed Validation Dataset
        ```
        mlcr get,dataset,preprocessed,openorca,_validation,_r2-downloader,_mlc -j
        ```

    === "Calibration"

        ### Get Preprocessed Calibration dataset
        ```
        mlcr get,dataset,preprocessed,openorca,_calibration,_r2-downloader,_mlc -j
        ```
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

=== "Full-Precision Models"

    === "Pytorch"

        === "From MLCOMMONS Storage"
    
            > **Note:**  One has to accept the [MLCommons Llama 2 License Confidentiality Notice](https://llama2.mlcommons.org/) to access the model files in MLCOMMONS Storage. 
    
            ### Get the Official MLPerf LLAMA2-70B model from MLCOMMONS Storage
            ```
            mlcr get,ml-model,llama2-70b,_pytorch,_r2-downloader,_70b,_mlc -j
            ```

        === "From Hugging Face repo"

            > **Note:** Access to the HuggingFace model could be requested [here](https://ai.meta.com/resources/models-and-libraries/llama-downloads/).

            ### Get model from HuggingFace repo
            ```
            mlcr get,ml-model,llama2-70b,_hf --hf_token=<huggingface access token> -j
            ```

=== "Quantized Models"

    === "Using TensorRT-LLM"

        === "Round v5.1"

            === "Quantize Locally"

                > **Note:**  One has to accept the [MLCommons Llama 2 License Confidentiality Notice](https://llama2.mlcommons.org/) to access the full precision model files in MLCOMMONS storage which are needed for quantization process.

                ```
                mlcr get,ml-model,llama2-70b,_nvidia,_fp8,_v5.1 -j
                ``` 

                - Use `--checkpoint=<Full Precision model path>` if model is already downloaded to a specific location.
        
        === "Round v5.0"

            === "Quantize Locally"

                > **Note:**  One has to accept the [MLCommons Llama 2 License Confidentiality Notice](https://llama2.mlcommons.org/) to access the full precision model files in MLCOMMONS storage which are needed for quantization process.

                ```
                mlcr get,ml-model,llama2-70b,_nvidia,_fp8,_v5.0 -j
                ``` 

                - Use `--checkpoint=<Full Precision model path>` if model is already downloaded to a specific location.

            === "Pre-Quantized Model from MLCOMMONS Storage"

                > **Note:**  One has to accept the [MLCommons Llama 2 License Confidentiality Notice](https://llama2.mlcommons.org/) to access the full precision model files and pre-quantized model files in MLCOMMONS storage.

                ```
                mlcr get,ml-model,llama2-70b,_nvidia,_fp8,_v5.0,_pre-quantized -j
                ``` 

                - Use `--checkpoint=<Full Precision model path>` if full precision model is already downloaded to a specific location.



- `--outdirname=<PATH_TO_DOWNLOAD_LLAMA2_70B_MODEL>` could be provided to download the model to a specific location.