The below tables lists the actively tests being carried out by MLCommons for the currently active MLPerf inference benchmarks. Please contact [Support](support@mlcommons.org) if you would like to add any new tests here. The tests are run using GitHub actions and the test results go to the below two repositories.
1. Short Runs: https://github.com/mlcommons/mlperf_inference_test_submissions_v5.0/
2. Full Runs: https://github.com/mlcommons/mlperf_inference_unofficial_submissions_v5.0/

These reporitories are per inference round.

[![MLPerf Inference Nvidia implementations](https://github.com/GATEOverflow/mlperf-automations/actions/workflows/test-nvidia-mlperf-inference-implementations.yml/badge.svg)](https://github.com/GATEOverflow/mlperf-automations/actions/workflows/test-nvidia-mlperf-inference-implementations.yml)

[![MLPerf inference ResNet50](https://github.com/mlcommons/mlperf-automations/actions/workflows/test-mlperf-inference-resnet50.yml/badge.svg)](https://github.com/mlcommons/mlperf-automations/actions/workflows/test-mlperf-inference-resnet50.yml)

[![MLPerf inference retinanet](https://github.com/mlcommons/mlperf-automations/actions/workflows/test-mlperf-inference-retinanet.yml/badge.svg)](https://github.com/mlcommons/mlperf-automations/actions/workflows/test-mlperf-inference-retinanet.yml)

[![MLPerf inference bert (deepsparse, tf, onnxruntime, pytorch)](https://github.com/mlcommons/mlperf-automations/actions/workflows/test-mlperf-inference-bert-deepsparse-tf-onnxruntime-pytorch.yml/badge.svg)](https://github.com/mlcommons/mlperf-automations/actions/workflows/test-mlperf-inference-bert-deepsparse-tf-onnxruntime-pytorch.yml)

[![MLPerf inference R-GAT](https://github.com/mlcommons/mlperf-automations/actions/workflows/test-mlperf-inference-rgat.yml/badge.svg)](https://github.com/mlcommons/mlperf-automations/actions/workflows/test-mlperf-inference-rgat.yml)



## Reference Implementation
| Benchmark            | CPU/GPU | Test Type | 
|----------------------|---------|-----------|
| resnet50            |   CPU | Short   |     
| retinanet           |   CPU | Short        |
| 3d-unet             |    -  |   TBD        |
| bert               |   CPU  |  Short       |
| gptj               |   Both | Short        |               
| llama2-70b         |   CPU  |  Short       |
| mixtral-8x7b       |   CPU  |  Short       |
| llama3.1-405b      |   -    |      -       |
| rgat               |   CPU      |   Short  |
| pointpainting      |   GPU      |    -     |
| stable-diffusion-xl |  GPU      |   Short  |
| dlrm_v2            | CPU        | Short    |

## Nvidia Implementation
| Benchmark            |  Test Type | 
|----------------------|-----------|
| resnet50            | Full       |     
| retinanet           | Full       |
| 3d-unet             | Full       |
| bert               |  Full       |
| gptj               |  Full       |               
| llama2-70b         |     -       |
| mixtral-8x7b       |     -       |
| llama3.1-405b      |     -      |
| rgat               |   NA       |
| pointpainting      |   NA       |
| stable-diffusion-xl | Full      |
| dlrm_v2            |  TBD       |
