---
hide:
  - toc
---

# Speech to Text using Whisper

## Dataset

The benchmark implementation run command will automatically download the validation and calibration datasets and do the necessary preprocessing. In case you want to download only the datasets, you can use the below commands.

=== "Validation"

    === "Preprocessed"

        ### Get Preprocessed Validation Dataset
        ```
        mlcr get,dataset,whisper,_preprocessed,_mlc,_r2-downloader --outdirname=<path to download> -j
        ```

    === "Unprocessed"

        ### Get Unprocessed Validation Dataset
        ```
        mlcr get,dataset,whisper,_unprocessed --outdirname=<path to download> -j
        ```

## Model
The benchmark implementation run command will automatically download the required model and do the necessary conversions if any. In case you want to only download the official model, you can use the below commands.

=== "Pytorch"

    === "From MLCOMMONS"

        ### Get the Official MLPerf Whisper model from MLCOMMONS Cloudflare R2
        ```
        mlcr get,ml-model,whisper,_r2-downloader,_mlc -j
        ```

- `--outdirname=<PATH_TO_DOWNLOAD_WHISPER_MODEL>` could be provided to download the model to a specific location.