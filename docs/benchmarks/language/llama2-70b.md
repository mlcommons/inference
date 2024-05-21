# Text Summarization using LLAMA2-70b

## Dataset

The benchmark implementation run command will automatically download the validation and calibration datasets and do the necessary preprocessing. In case you want to download only the datasets, you can use the below commands.

=== "Validation"
    LLAMA2-70b validation run uses the Open ORCA dataset.

    ### Get Validation Dataset
    ```
    cm run script --tags=get,dataset,openorca,validation -j
    ```

## Model
The benchmark implementation run command will automatically download the required model and do the necessary conversions. In case you want to only download the official model, you can use the below commands.

Get the Official MLPerf LLAMA2-70b Model

=== "Pytorch"

    ### Pytorch
    ```
    cm run script --tags=get,ml-model,llama2-70b,_pytorch -j
    ```

## Benchmark Implementations
=== "MLCommons-Python"
    ### MLPerf Reference Implementation in Python
    
    LLAMA2-70b-99
{{ mlperf_inference_implementation_readme (4, "llama2-70b-99", "reference") }}

    LLAMA2-70b-99.9
{{ mlperf_inference_implementation_readme (4, "llama2-70b-99.9", "reference") }}

=== "Nvidia"
    ### Nvidia MLPerf Implementation
    
    LLAMA2-70b-99
{{ mlperf_inference_implementation_readme (4, "llama2-70b-99", "nvidia") }}

    LLAMA2-70b-99.9
{{ mlperf_inference_implementation_readme (4, "llama2-70b-99.9", "nvidia") }}


=== "Qualcomm"
    ### Qualcomm AI100 MLPerf Implementation

    LLAMA2-70b-99
{{ mlperf_inference_implementation_readme (4, "llama2-70b-99", "qualcomm") }}

