# FuriosaAI Internal Evaluation(MLPerf v4.1 candidate)
## Prerequisites
* Set up AWS credentials, following instructions on "[기타 - DVC & AWS S3 설정](https://www.notion.so/furiosa/DVC-AWS-S3-89c2ee0ce6564dc1bb6ba134e6e86381)".
* Install Conda: https://docs.anaconda.com/free/anaconda/install/index.html


## Installation
```
git clone --branch v4.1-internal https://github.com/furiosa-ai/inference.git
cd inference

# (optional, for mlperf loadgen) if GCC Compiler is not installed on Ubuntu,
apt-get update && apt-get install build-essential -y

# (optional, for Stable Diffusion only) it requires cv2 related devian packages to run Stable Diffusion
DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install libgl1 libglib2.0-0 -y
```

## How to run end-to-end evaluation
Ene-to-end(E2E) evaluation is the process of downloading models and dataset, building a Python environment, and performing model accuracy evaluation. E2E scripts are developed based on [f9a643c](https://github.com/mlcommons/inference/commit/f9a643c0a0e920588da1b51a1d822e1071a9dbec). 

To run E2E evaluation:

```
make [model_name]
```

or equivalently,

```
bash scripts/build_[model_name]_env.sh
bash scripts/eval_[model_name].sh
```

- `model_name` includes [resnet, retinanet, 3d-unet, bert, gpt-j, rnnt, llama2, stablediffusion, all]
- For example, to run E2E ResNet evaluation
    ```
    make resnet
    ```

    or

    ```
    # build conda environment and download dataset
    bash scripts/build_resnet_env.sh
    
    # run evaluation on pre-built conda environment
    bash scripts/eval_resnet.sh
    ```

### Configurability
Some parameters are configurable, for example,
- llama2-70b
    
    The command `make llama2` is equivalent to
    ```
    export SCENARIO=Offline # SCENARIO is one of [Offline, Server]
    export N_COUNT=24576   # N_COUNT is a number between [1, 24576]
    export DATA_TYPE=float32    # DATA_TYPE is one of [float32, float16, bfloat16]
    export DEVICE=cuda:0    # DEVICE is one of [cpu, cuda:0]
    make llama2
    ```
    Each environment variable above has the value as default, which can be changed to another.

Likewise,

- stable-diffusion-xl-base

    ```
    export SCENARIO=Offline # SCENARIO is one of [Offline, SingleStream, MultiStream, Server]
    export N_COUNT=5000   # N_COUNT is a number between [1, 5000]
    export DATA_TYPE=fp32    # DATA_TYPE is one of [fp32, fp16, bf16]
    export DEVICE=cuda    # DEVICE is one of [cpu, cuda]
    make stablediffusion
    ```


## Evaluation results

### Summary

In this section, we present our evaluation result(our result) and [mlperf reference accuracy(reference accuracy)](https://github.com/mlcommons/inference/blob/e39003a9c4c89a2215db0ca57ad7a57b16f9a785/tools/submission/submission_checker.py#L1116-L1129) in a comparative manner. It is expected that this will allow us to clearly see the goals we need to achieve and our current status. 

Note that all models are in Pytorch framework with float32 data type, and all experiments were conducted in Offline scenario.

#### language

- llama2-70b

    |     benchmark     | our result | reference accuracy |
    |:-----------------:|:----------:|:---------------:|
    | ROUGE1            | 44.4312 (100.00%*)   | [44.4312](https://github.com/mlcommons/inference/blob/e39003a9c4c89a2215db0ca57ad7a57b16f9a785/tools/submission/submission_checker.py#L1128C44-L1128C51)         |
    | ROUGE2            | 22.0352 (100.00%)   | [22.0352](https://github.com/mlcommons/inference/blob/e39003a9c4c89a2215db0ca57ad7a57b16f9a785/tools/submission/submission_checker.py#L1128C71-L1128C78)         |
    | ROUGEL            | 28.6162 (100.00%)   | [28.6162](https://github.com/mlcommons/inference/blob/e39003a9c4c89a2215db0ca57ad7a57b16f9a785/tools/submission/submission_checker.py#L1128C98-L1128C105)         |
    | TOKENS_PER_SAMPLE | 294.4** (99.85%)    | [294.45](https://github.com/mlcommons/inference/blob/e39003a9c4c89a2215db0ca57ad7a57b16f9a785/tools/submission/submission_checker.py#L1128C136-L1128C142)          |

    \* `round(our result / reference accuracy, 2)`

    \** `294.4`, our result, is [rounded at the second decimal place](https://github.com/mlcommons/inference/blob/e39003a9c4c89a2215db0ca57ad7a57b16f9a785/language/llama2-70b/evaluate-accuracy.py#L103), whereas `294.4462890625` is actual. It is estimated that the reference accuracy is rounded to the 3rd decimal place.


- gpt-j

    |         | our result | reference accuracy |
    |:-------:|:----------------------:|:---------------:|
    | ROUGE1  | 42.9865 (100.00%)         | [42.9865](https://github.com/mlcommons/inference/blob/e39003a9c4c89a2215db0ca57ad7a57b16f9a785/tools/submission/submission_checker.py#L1126C38-L1126C45)         |
    | ROUGE2  | 20.1235 (100.00%)         | [20.1235](https://github.com/mlcommons/inference/blob/e39003a9c4c89a2215db0ca57ad7a57b16f9a785/tools/submission/submission_checker.py#L1126C65-L1126C72)         |
    | ROUGEL  | 29.9881 (100.00%)               | [29.9881](https://github.com/mlcommons/inference/blob/e39003a9c4c89a2215db0ca57ad7a57b16f9a785/tools/submission/submission_checker.py#L1126C92-L1126C99)         |
    | GEN_LEN | 4016878 (100.00%)               | [4016878](https://github.com/mlcommons/inference/blob/e39003a9c4c89a2215db0ca57ad7a57b16f9a785/tools/submission/submission_checker.py#L1126C120-L1126C127)         |

- bert

    |    |        our result        | reference accuracy |
    |:--:|:------------------------:|:---------------:|
    | F1 | 90.87487229720105 (100.00%) | [90.874](https://github.com/mlcommons/inference/blob/e39003a9c4c89a2215db0ca57ad7a57b16f9a785/tools/submission/submission_checker.py#L1119C31-L1119C37)          |

#### vision

- resnet

    |        |     our result    | reference accuracy | accuracy(pytorch backend) |
    |:------:|:-----------------:|:---------------------------------------:|:---------------------------:|
    | acc(%) | 76.144 (99.59% / 100.17%)* | [76.46](https://github.com/mlcommons/inference/blob/e39003a9c4c89a2215db0ca57ad7a57b16f9a785/tools/submission/submission_checker.py#L1116C31-L1116C36)                                   | [76.014](https://github.com/mlcommons/inference/blob/e39003a9c4c89a2215db0ca57ad7a57b16f9a785/vision/classification_and_detection/README.md?plain=1#L18C29-L18C36)                      |
    
    \* There are differences between the types of model framework. That is, our model used for our result is Pytorch backended, and the one used for reference accuracy is Tensorflow. The reference accuracy of Pytorch backended model can be checked in the accuracy(pytorch backend) item.

- retinanet

    |        | our result | reference accuracy |
    |:------:|:----------:|:---------------:|
    | mAP(%) | 37.552 (100.01%)    | [37.55](https://github.com/mlcommons/inference/blob/e39003a9c4c89a2215db0ca57ad7a57b16f9a785/tools/submission/submission_checker.py#L1117C34-L1117C39)           |

- 3d-unet

    |      | our result | reference accuracy |
    |:----:|:----------:|:---------------:|
    | DICE | 0.86173 (100.00%)    | [0.86170](https://github.com/mlcommons/inference/blob/e39003a9c4c89a2215db0ca57ad7a57b16f9a785/tools/submission/submission_checker.py#L1124C38-L1124C45)         |

#### speech recognition

- rnnt

    |            |        our result       | reference accuracy |
    |:----------:|:-----------------------:|:---------------:|
    | WER(%)     | 7.459018241673995      | [7.452](https://github.com/mlcommons/inference/blob/e39003a9c4c89a2215db0ca57ad7a57b16f9a785/tools/submission/submission_checker.py#L1118C36-L1118C41)           |
    | 100-WER(%) | 92.5409817583 (99.99%*) | [92.548](https://github.com/mlcommons/inference/blob/e39003a9c4c89a2215db0ca57ad7a57b16f9a785/tools/submission/submission_checker.py#L1118C29-L1118C42)          |

    \* Reason unknown.

#### recommendation

- dlrm-v2

    |     | our result | reference accuracy |
    |:---:|:----------:|:---------------:|
    | AUC | TBA*       | [80.31](https://github.com/mlcommons/inference/blob/e39003a9c4c89a2215db0ca57ad7a57b16f9a785/tools/submission/submission_checker.py#L1122C37-L1122C42)           |

    \* 8 GPUs are needed for evaluation.

#### text to image

- stable-diffusion-xl-base

    |            |     our result     | reference accuracy |
    |:----------:|:------------------:|:---------------:|
    | CLIP_SCORE | 31.74663716465235 (100.19%)  | [31.68631873](https://github.com/mlcommons/inference/blob/e39003a9c4c89a2215db0ca57ad7a57b16f9a785/tools/submission/submission_checker.py#L1129C51-L1129C62)     |
    | FID_SCORE  | 23.431448173651063 (101.83%) | [23.01085758](https://github.com/mlcommons/inference/blob/e39003a9c4c89a2215db0ca57ad7a57b16f9a785/tools/submission/submission_checker.py#L1129C77-L1129C88)     |

    \* The gap between our result and reference accuracy is quite large. There is a possibility that the published weight and the one used in the experiment are different. It is neccesary to update evaluation result later.


### v3.1
- Default settings:
    - scenario: Offline
    - model framework: pytorch
    - data type: f32
- Device info:
    - GPU: 1 NVIDIA A100-SXM4-80GB
    - CPU: Intel(R) Xeon(R) Platinum 8358 CPU

| model name | our result    | mlperf result                                                                                                               | input shape*            | dataset                                                                                                                               |
|------------|--------------------|----------------------------------------------------------------------------------------------------------------------------------|------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| resnet     | 76.144%(top1 Acc.) | [76.014%(top1 Acc.)](https://github.com/mlcommons/inference/blob/v3.1/vision/classification_and_detection/README.md?plain=1#L18) | 1x3x224x224(NxCxHxW)       | [Imagenet2012 validation](https://github.com/mlcommons/inference/blob/v3.1/vision/classification_and_detection/README.md?plain=1#L86) (num_data: 50,000) |
| retinanet  | 0.3755(mAP)        | [0.3755(mAP)](https://github.com/mlcommons/inference/blob/v3.1/vision/classification_and_detection/README.md?plain=1#L21)        | 1x3x800x800(NxCxHxW)       | [MLPerf Openimages](https://github.com/mlcommons/inference/blob/v3.1/vision/classification_and_detection/README.md?plain=1#L87) (num_data: 24,781)      |
| 3d-unet    | 0.86173(Dice)      | [0.86170(Dice)](https://github.com/mlcommons/inference/blob/v3.1/vision/medical_imaging/3d-unet-kits19/README.md?plain=1#L23)    | 1x1x128x128x128(NxCxDxHxW) | [eval set of KiTS 2019](https://github.com/mlcommons/inference/blob/v3.1/vision/medical_imaging/3d-unet-kits19/README.md?plain=1#L23) (num_data: 2,761) |
| bert       | 90.874%(F1)        | [90.874%(F1)](https://github.com/mlcommons/inference/blob/v3.1/language/bert/README.md?plain=1#L19)                              | 1x384(NxS)                 | [SQuAD v1.1 validation set](https://github.com/mlcommons/inference/blob/v3.1/language/bert/README.md?plain=1#L19) (num_data: 10,833)                     |
| gpt-j      | 42.9865(Rouge1)    | [42.9865(Rouge1)](https://github.com/mlcommons/inference/blob/v3.1/language/gpt-j/README.md?plain=1#L91)                         | 1x1919(NxS)                | [CNN-Daily Mail](https://github.com/mlcommons/inference/blob/v3.1/language/gpt-j/README.md?plain=1#L54) (num_data: 13,368)                              |
| rnnt       | 7.45901%(WER)      | [74.45225%(WER)](https://github.com/mlcommons/inference/blob/v3.1/speech_recognition/rnnt/README.md?plain=1#L116)                | 500x1x240(SxNxF)       | [OpenSLR LibriSpeech Corpus](https://github.com/mlcommons/inference/blob/v3.1/speech_recognition/rnnt/README.md?plain=1#L24) (num_data: 2,513)         |
| dlrm-v2    | TBA                | [80.31%(AUC)](https://github.com/mlcommons/inference/blob/v3.1/recommendation/dlrm_v2/pytorch/README.md?plain=1#L12)             | TBA                    | [Criteo Terabyte (day 23)](https://github.com/mlcommons/inference/blob/v3.1/recommendation/dlrm_v2/pytorch/README.md?plain=1#L92) (num_data: TBA)     |

\* Shape of preprocessed(transformed/tokenized) input. Notations:
- N: Batch size
- C: input Channel dimension
- H: Height dimension
- W: Width dimension
- D: Depth dimension
- S: max Sequence length
- F: input Feature dimension

To get verified evaluation log:

```
# (optional) if not installed,
pip install dvc[s3]

make log_[model_name]
```

- `model_name` includes [resnet, retinanet, 3d-unet, bert, gpt-j, rnnt, all]
- For example, with
    ```
    make log_resnet
    ```
    the evaluation log of ResNet will be pulled to `logs/internal/resnet`.


### v4.0
- Default settings:
    - scenario: Offline
    - model framework: pytorch

- LLaMA2-70b

    | data type | our result | mlperf result                                                                                                      | elapsed time | device |
    |-----------|-----------------|--------------------------------------------------------------------------------------------------------------------|--------------|--------|
    | float32   | 44.4312(Rouge1) | [44.4312(Rouge1)](https://github.com/mlcommons/inference/blob/master/tools/submission/submission_checker.py#L1128) | 6.6 days     | 6 H100 PCIe |
    | float16   | 44.4362(Rouge1) |                                                          -                                                         | 3.5 days     | 3 A100-SXM4-80GB |
    | bfloat16  | 44.4625(Rouge1) |                                                          -                                                         | 3.6 days     | 3 A100-SXM4-80GB |

- Stable Diffusion XL base

    | data type | our result     | mlperf result                                                                                                          | elapsed time | device    |
    |-----------|---------------------|------------------------------------------------------------------------------------------------------------------------|--------------|-----------|
    | float32   | 31.7466(clip_score) | [31.6863(clip_score)](https://github.com/mlcommons/inference/blob/master/tools/submission/submission_checker.py#L1129) | 24 hours     | 1 GeForce RTX 3090 |
    | float16   | 31.7558(clip_score) |                                                            -                                                           | 7.5 hours    | 1 GeForce RTX 3090 |
    | bfloat16  | 31.7380(clip_score) |                                                            -                                                           | 8.3 hours    | 1 GeForce RTX 3090 |


To get verified evaluation log:

```
# (optional) if not installed,
pip install dvc[s3]

make log_[model_name]
```

- model_name includes [llama2, stablediffusion, all]
- For example, with
    ```
    make log_llama2
    ```
    the evaluation logs of LLaMA2-70b will be pulled to logs/internal/llama2-70b.


# MLPerf™ Inference Benchmark Suite
MLPerf Inference is a benchmark suite for measuring how fast systems can run models in a variety of deployment scenarios. 

Please see the [MLPerf Inference benchmark paper](https://arxiv.org/abs/1911.02549) for a detailed description of the benchmarks along with the motivation and guiding principles behind the benchmark suite. If you use any part of this benchmark (e.g., reference implementations, submissions, etc.), please cite the following:

```
@misc{reddi2019mlperf,
    title={MLPerf Inference Benchmark},
    author={Vijay Janapa Reddi and Christine Cheng and David Kanter and Peter Mattson and Guenther Schmuelling and Carole-Jean Wu and Brian Anderson and Maximilien Breughe and Mark Charlebois and William Chou and Ramesh Chukka and Cody Coleman and Sam Davis and Pan Deng and Greg Diamos and Jared Duke and Dave Fick and J. Scott Gardner and Itay Hubara and Sachin Idgunji and Thomas B. Jablin and Jeff Jiao and Tom St. John and Pankaj Kanwar and David Lee and Jeffery Liao and Anton Lokhmotov and Francisco Massa and Peng Meng and Paulius Micikevicius and Colin Osborne and Gennady Pekhimenko and Arun Tejusve Raghunath Rajan and Dilip Sequeira and Ashish Sirasao and Fei Sun and Hanlin Tang and Michael Thomson and Frank Wei and Ephrem Wu and Lingjie Xu and Koichi Yamada and Bing Yu and George Yuan and Aaron Zhong and Peizhao Zhang and Yuchen Zhou},
    year={2019},
    eprint={1911.02549},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
Please see [here](https://docs.mlcommons.org/inference/benchmarks/) for the MLPerf inference documentation website which includes automated commands to run MLPerf inference benchmarks using different implementations.

## MLPerf Inference v4.1 (submission deadline July 26, 2024)

For submissions, please use the master branch and any commit since the [4.1 seed release](https://github.com/mlcommons/inference/pull/1736/files) although it is best to use the latest commit. v4.1 tag will be created from the master branch after the result publication.

For power submissions please use [SPEC PTD 1.10](https://github.com/mlcommons/power/tree/main/inference_v1.0) (needs special access) and any commit of the power-dev repository after the [code-freeze](https://github.com/mlcommons/power-dev/pull/325)

| model | reference app | framework | dataset | category
| ---- | ---- | ---- | ---- | ---- |
| resnet50-v1.5 | [vision/classification_and_detection](https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection) | tensorflow, onnx, tvm, ncnn | imagenet2012 | edge,datacenter |
| retinanet 800x800 | [vision/classification_and_detection](https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection) | pytorch, onnx | openimages resized to 800x800| edge,datacenter |
| bert | [language/bert](https://github.com/mlcommons/inference/tree/master/language/bert) | tensorflow, pytorch, onnx | squad-1.1 | edge,datacenter |
| dlrm-v2 | [recommendation/dlrm_v2](https://github.com/mlcommons/inference/tree/master/recommendation/dlrm_v2/pytorch) | pytorch | Multihot Criteo Terabyte | datacenter |
| 3d-unet | [vision/medical_imaging/3d-unet-kits19](https://github.com/mlcommons/inference/tree/master/vision/medical_imaging/3d-unet-kits19) | pytorch, tensorflow, onnx | KiTS19 | edge,datacenter |
| gpt-j | [language/gpt-j](https://github.com/mlcommons/inference/tree/master/language/gpt-j)| pytorch | CNN-Daily Mail | edge,datacenter |
| stable-diffusion-xl | [text_to_image](https://github.com/mlcommons/inference/tree/master/text_to_image) | pytorch | COCO 2014| edge,datacenter |
| llama2-70b | [language/llama2-70b](https://github.com/mlcommons/inference/tree/master/language/llama2-70b) | pytorch | OpenOrca | datacenter |
| mixtral-8x7b | [language/mixtral-8x7b](https://github.com/mlcommons/inference/tree/master/language/mixtral-8x7b) | pytorch | OpenOrca, MBXP, GSM8K | datacenter |

* Framework here is given for the reference implementation. Submitters are free to use their own frameworks to run the benchmark.

## MLPerf Inference v4.0 (submission February 23, 2024)

There is an extra one-week extension allowed only for the llama2-70b submissions. For submissions, please use the master branch and any commit since the [4.0 seed release](https://github.com/mlcommons/inference/commit/8e36925bd36a503e39fcbbc488e9e46126f079ed) although it is best to use the latest commit. v4.0 tag will be created from the master branch after the result publication.

For power submissions please use [SPEC PTD 1.10](https://github.com/mlcommons/power/tree/main/inference_v1.0) (needs special access) and any commit of the power-dev repository after the [code-freeze](https://github.com/mlcommons/power-dev/commit/4e026f43481f46ad57d2464d28924018444b0428)

| model | reference app | framework | dataset | category
| ---- | ---- | ---- | ---- | ---- |
| resnet50-v1.5 | [vision/classification_and_detection](https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection) | tensorflow, onnx, tvm, ncnn | imagenet2012 | edge,datacenter |
| retinanet 800x800 | [vision/classification_and_detection](https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection) | pytorch, onnx | openimages resized to 800x800| edge,datacenter |
| bert | [language/bert](https://github.com/mlcommons/inference/tree/master/language/bert) | tensorflow, pytorch, onnx | squad-1.1 | edge,datacenter |
| dlrm-v2 | [recommendation/dlrm_v2](https://github.com/mlcommons/inference/tree/master/recommendation/dlrm_v2/pytorch) | pytorch | Multihot Criteo Terabyte | datacenter |
| 3d-unet | [vision/medical_imaging/3d-unet-kits19](https://github.com/mlcommons/inference/tree/master/vision/medical_imaging/3d-unet-kits19) | pytorch, tensorflow, onnx | KiTS19 | edge,datacenter |
| rnnt | [speech_recognition/rnnt](https://github.com/mlcommons/inference/tree/master/speech_recognition/rnnt) | pytorch | OpenSLR LibriSpeech Corpus | edge,datacenter |
| gpt-j | [language/gpt-j](https://github.com/mlcommons/inference/tree/master/language/gpt-j)| pytorch | CNN-Daily Mail | edge,datacenter |
| stable-diffusion-xl | [text_to_image](https://github.com/mlcommons/inference/tree/master/text_to_image) | pytorch | COCO 2014| edge,datacenter |
| llama2-70b | [language/llama2-70b](https://github.com/mlcommons/inference/tree/master/language/llama2-70b) | pytorch | OpenOrca | datacenter |

* Framework here is given for the reference implementation. Submitters are free to use their own frameworks to run the benchmark.

## MLPerf Inference v3.1 (submission August 18, 2023)
Please use [v3.1 tag](https://github.com/mlcommons/inference/releases/tag/v3.1) (```git checkout v3.1```) if you would like to reproduce the v3.1 results. 

For reproducing power submissions please use the `master` branch of the [MLCommons power-dev](https://github.com/mlcommons/power-dev) repository and checkout to [e9e16b1299ef61a2a5d8b9abf5d759309293c440](https://github.com/mlcommons/power-dev/tree/e9e16b1299ef61a2a5d8b9abf5d759309293c440). 

You can see the individual README files in the benchmark task folders for more details regarding the benchmarks. For reproducing the submitted results please see the README files under the respective submitter folders in the [inference v3.1 results repository](https://github.com/mlcommons/inference_results_v3.1).

| model | reference app | framework | dataset | category
| ---- | ---- | ---- | ---- | ---- |
| resnet50-v1.5 | [vision/classification_and_detection](https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection) | tensorflow, onnx, tvm, ncnn | imagenet2012 | edge,datacenter |
| retinanet 800x800 | [vision/classification_and_detection](https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection) | pytorch, onnx | openimages resized to 800x800| edge,datacenter |
| bert | [language/bert](https://github.com/mlcommons/inference/tree/master/language/bert) | tensorflow, pytorch, onnx | squad-1.1 | edge,datacenter |
| dlrm-v2 | [recommendation/dlrm_v2](https://github.com/mlcommons/inference/tree/master/recommendation/dlrm_v2/pytorch) | pytorch | Multihot Criteo Terabyte | datacenter |
| 3d-unet | [vision/medical_imaging/3d-unet-kits19](https://github.com/mlcommons/inference/tree/master/vision/medical_imaging/3d-unet-kits19) | pytorch, tensorflow, onnx | KiTS19 | edge,datacenter |
| rnnt | [speech_recognition/rnnt](https://github.com/mlcommons/inference/tree/master/speech_recognition/rnnt) | pytorch | OpenSLR LibriSpeech Corpus | edge,datacenter |
| gpt-j | [language/gpt-j](https://github.com/mlcommons/inference/tree/master/language/gpt-j)| pytorch | CNN-Daily Mail | edge,datacenter |


## MLPerf Inference v3.0 (submission 03/03/2023)
Please use the v3.0 tag (```git checkout v3.0```) if you would like to reproduce v3.0 results.

You can see the individual Readme files in the reference app for more details.

| model | reference app | framework | dataset | category
| ---- | ---- | ---- | ---- | ---- |
| resnet50-v1.5 | [vision/classification_and_detection](https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection) | tensorflow, onnx, tvm | imagenet2012 | edge,datacenter |
| retinanet 800x800 | [vision/classification_and_detection](https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection) | pytorch, onnx | openimages resized to 800x800| edge,datacenter |
| bert | [language/bert](https://github.com/mlcommons/inference/tree/master/language/bert) | tensorflow, pytorch, onnx | squad-1.1 | edge,datacenter |
| dlrm | [recommendation/dlrm](https://github.com/mlcommons/inference/tree/master/recommendation/dlrm/pytorch) | pytorch, tensorflow | Criteo Terabyte | datacenter |
| 3d-unet | [vision/medical_imaging/3d-unet-kits19](https://github.com/mlcommons/inference/tree/master/vision/medical_imaging/3d-unet-kits19) | pytorch, tensorflow, onnx | KiTS19 | edge,datacenter |
| rnnt | [speech_recognition/rnnt](https://github.com/mlcommons/inference/tree/master/speech_recognition/rnnt) | pytorch | OpenSLR LibriSpeech Corpus | edge,datacenter |


## MLPerf Inference v2.1 (submission 08/05/2022)
Use the r2.1 branch (```git checkout r2.1```) if you want to submit or reproduce v2.1 results.

See the individual Readme files in the reference app for details.

| model | reference app | framework | dataset | category
| ---- | ---- | ---- | ---- | ---- |
| resnet50-v1.5 | [vision/classification_and_detection](https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection) | tensorflow, onnx | imagenet2012 | edge,datacenter |
| retinanet 800x800 | [vision/classification_and_detection](https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection) | pytorch, onnx | openimages resized to 800x800| edge,datacenter |
| bert | [language/bert](https://github.com/mlcommons/inference/tree/master/language/bert) | tensorflow, pytorch, onnx | squad-1.1 | edge,datacenter |
| dlrm | [recommendation/dlrm](https://github.com/mlcommons/inference/tree/master/recommendation/dlrm/pytorch) | pytorch, tensorflow | Criteo Terabyte | datacenter |
| 3d-unet | [vision/medical_imaging/3d-unet-kits19](https://github.com/mlcommons/inference/tree/master/vision/medical_imaging/3d-unet-kits19) | pytorch, tensorflow, onnx | KiTS19 | edge,datacenter |
| rnnt | [speech_recognition/rnnt](https://github.com/mlcommons/inference/tree/master/speech_recognition/rnnt) | pytorch | OpenSLR LibriSpeech Corpus | edge,datacenter |


## MLPerf Inference v2.0 (submission 02/25/2022)
Use the r2.0 branch (```git checkout r2.0```) if you want to submit or reproduce v2.0 results.

See the individual Readme files in the reference app for details.

| model | reference app | framework | dataset | category |
| ---- | ---- | ---- | ---- | ---- |
| resnet50-v1.5 | [vision/classification_and_detection](https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection) | tensorflow, onnx | imagenet2012 | edge,datacenter |
| ssd-mobilenet 300x300 | [vision/classification_and_detection](https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection) | tensorflow, pytorch, onnx| coco resized to 300x300 | edge |
| ssd-resnet34 1200x1200 | [vision/classification_and_detection](https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection) | tensorflow, pytorch, onnx | coco resized to 1200x1200| edge,datacenter |
| bert | [language/bert](https://github.com/mlcommons/inference/tree/master/language/bert) | tensorflow, pytorch, onnx | squad-1.1 | edge,datacenter |
| dlrm | [recommendation/dlrm](https://github.com/mlcommons/inference/tree/master/recommendation/dlrm/pytorch) | pytorch, tensorflow | Criteo Terabyte | datacenter |
| 3d-unet | [vision/medical_imaging/3d-unet-kits19](https://github.com/mlcommons/inference/tree/master/vision/medical_imaging/3d-unet-kits19) | pytorch, tensorflow, onnx | KiTS19 | edge,datacenter |
| rnnt | [speech_recognition/rnnt](https://github.com/mlcommons/inference/tree/master/speech_recognition/rnnt) | pytorch | OpenSLR LibriSpeech Corpus | edge,datacenter |


## MLPerf Inference v1.1 (submission 08/13/2021)
Use the r1.1 branch (```git checkout r1.1```) if you want to submit or reproduce v1.1 results.

See the individual Readme files in the reference app for details.

| model | reference app | framework | dataset | category |
| ---- | ---- | ---- | ---- | ---- |
| resnet50-v1.5 | [vision/classification_and_detection](https://github.com/mlcommons/inference/tree/r1.1/vision/classification_and_detection) | tensorflow, onnx | imagenet2012 | edge,datacenter |
| ssd-mobilenet 300x300 | [vision/classification_and_detection](https://github.com/mlcommons/inference/tree/r1.1/vision/classification_and_detection) | tensorflow, pytorch, onnx| coco resized to 300x300 | edge |
| ssd-resnet34 1200x1200 | [vision/classification_and_detection](https://github.com/mlcommons/inference/tree/r1.1/vision/classification_and_detection) | tensorflow, pytorch, onnx | coco resized to 1200x1200| edge,datacenter |
| bert | [language/bert](https://github.com/mlcommons/inference/tree/r1.1/language/bert) | tensorflow, pytorch, onnx | squad-1.1 | edge,datacenter |
| dlrm | [recommendation/dlrm](https://github.com/mlcommons/inference/tree/r1.1/recommendation/dlrm/pytorch) | pytorch, tensorflow | Criteo Terabyte | datacenter |
| 3d-unet | [vision/medical_imaging/3d-unet](https://github.com/mlcommons/inference/tree/r1.1/vision/medical_imaging/3d-unet) | pytorch, tensorflow(?), onnx(?) | BraTS 2019 | edge,datacenter |
| rnnt | [speech_recognition/rnnt](https://github.com/mlcommons/inference/tree/r1.1/speech_recognition/rnnt) | pytorch | OpenSLR LibriSpeech Corpus | edge,datacenter |

## MLPerf Inference v1.0 (submission 03/19/2021)
Use the r1.0 branch (```git checkout r1.0```) if you want to submit or reproduce v1.0 results.

See the individual Readme files in the reference app for details.

| model | reference app | framework | dataset | category |
| ---- | ---- | ---- | ---- | ---- |
| resnet50-v1.5 | [vision/classification_and_detection](https://github.com/mlcommons/inference/tree/r1.0/vision/classification_and_detection) | tensorflow, onnx | imagenet2012 | edge,datacenter |
| ssd-mobilenet 300x300 | [vision/classification_and_detection](https://github.com/mlcommons/inference/tree/r1.0/vision/classification_and_detection) | tensorflow, pytorch, onnx| coco resized to 300x300 | edge |
| ssd-resnet34 1200x1200 | [vision/classification_and_detection](https://github.com/mlcommons/inference/tree/r1.0/vision/classification_and_detection) | tensorflow, pytorch, onnx | coco resized to 1200x1200| edge,datacenter |
| bert | [language/bert](https://github.com/mlcommons/inference/tree/r1.0/language/bert) | tensorflow, pytorch, onnx | squad-1.1 | edge,datacenter |
| dlrm | [recommendation/dlrm](https://github.com/mlcommons/inference/tree/r1.0/recommendation/dlrm/pytorch) | pytorch, tensorflow(?) | Criteo Terabyte | datacenter |
| 3d-unet | [vision/medical_imaging/3d-unet](https://github.com/mlcommons/inference/tree/r1.0/vision/medical_imaging/3d-unet) | pytorch, tensorflow(?), onnx(?) | BraTS 2019 | edge,datacenter |
| rnnt | [speech_recognition/rnnt](https://github.com/mlcommons/inference/tree/r1.0/speech_recognition/rnnt) | pytorch | OpenSLR LibriSpeech Corpus | edge,datacenter |


## MLPerf Inference v0.7 (submission 9/18/2020)
Use the r0.7 branch (```git checkout r0.7```) if you want to submit or reproduce v0.7 results.

See the individual Readme files in the reference app for details.

| model | reference app | framework | dataset |
| ---- | ---- | ---- | ---- |
| resnet50-v1.5 | [vision/classification_and_detection](https://github.com/mlcommons/inference/tree/r0.7/vision/classification_and_detection) | tensorflow, pytorch, onnx | imagenet2012 |
| ssd-mobilenet 300x300 | [vision/classification_and_detection](https://github.com/mlcommons/inference/tree/r0.7/vision/classification_and_detection) | tensorflow, pytorch, onnx| coco resized to 300x300 | 
| ssd-resnet34 1200x1200 | [vision/classification_and_detection](https://github.com/mlcommons/inference/tree/r0.7/vision/classification_and_detection) | tensorflow, pytorch, onnx | coco resized to 1200x1200|
| bert | [language/bert](https://github.com/mlcommons/inference/tree/r0.7/language/bert) | tensorflow, pytorch, onnx | squad-1.1 |
| dlrm | [recommendation/dlrm](https://github.com/mlcommons/inference/tree/r0.7/recommendation/dlrm/pytorch) | pytorch, tensorflow(?), onnx(?) | Criteo Terabyte |
| 3d-unet | [vision/medical_imaging/3d-unet](https://github.com/mlcommons/inference/tree/r0.7/vision/medical_imaging/3d-unet) | pytorch, tensorflow(?), onnx(?) | BraTS 2019 |
| rnnt | [speech_recognition/rnnt](https://github.com/mlcommons/inference/tree/r0.7/speech_recognition/rnnt) | pytorch | OpenSLR LibriSpeech Corpus |

## MLPerf Inference v0.5
Use the r0.5 branch (```git checkout r0.5```) if you want to reproduce v0.5 results.

See the individual Readme files in the reference app for details.

| model | reference app | framework | dataset |
| ---- | ---- | ---- | ---- |
| resnet50-v1.5 | [v0.5/classification_and_detection](https://github.com/mlcommons/inference/tree/r0.5/v0.5/classification_and_detection) | tensorflow, pytorch, onnx | imagenet2012 |
| mobilenet-v1 | [v0.5/classification_and_detection](https://github.com/mlcommons/inference/tree/r0.5/v0.5/classification_and_detection) |tensorflow, pytorch, onnx | imagenet2012 |
| ssd-mobilenet 300x300 | [v0.5/classification_and_detection](https://github.com/mlcommons/inference/tree/r0.5/v0.5/classification_and_detection) |tensorflow, pytorch, onnx | coco resized to 300x300 |
| ssd-resnet34 1200x1200 | [v0.5/classification_and_detection](https://github.com/mlcommons/inference/tree/r0.5/v0.5/classification_and_detection) | tensorflow, pytorch, onnx | coco resized to 1200x1200 |
| gnmt | [v0.5/translation/gnmt/](https://github.com/mlcommons/inference/tree/r0.5/v0.5/translation/gnmt/tensorflow) | tensorflow, pytorch | See Readme |
