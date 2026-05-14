# MLPerf® Inference Benchmark Suite
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

## MLPerf Inference v6.0 (submission deadline February 13, 2026)

For submissions, please use the master branch and any commit since the [6.0 seed release](https://github.com/mlcommons/inference/commit/f131a0d29ccae9a967d93ffe96f66b1be3537d3b) although it is best to use the latest commit in the [master branch](https://github.com/mlcommons/inference).

For power submissions please use [SPEC PTD 1.11.1](https://github.com/mlcommons/power) (needs special access) and any commit of the power-dev repository after the [code-freeze](https://github.com/mlcommons/power-dev/commit/c4b3ad8202fbd8ac28d77149e5e7aeadb725bbf2)


| model | reference app | framework | dataset | category
| ---- | ---- | ---- | ---- | ---- |
| resnet50-v1.5 | [vision/classification_and_detection](https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection) | tensorflow, onnx, tvm, ncnn | imagenet2012 | edge |
| yolo v11 | [vision/classification_and_detection](https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection/yolo) | pytorch, onnx | COCO safe subset | edge |
| bert | [language/bert](https://github.com/mlcommons/inference/tree/master/language/bert) | tensorflow, pytorch, onnx | squad-1.1 | edge |
| dlrm-v3 | [recommendation/dlrm_v3](https://github.com/mlcommons/inference/tree/master/recommendation/dlrm_v3/pytorch) | pytorch | Synthetic dataset | datacenter |
| 3d-unet | [vision/medical_imaging/3d-unet-kits19](https://github.com/mlcommons/inference/tree/master/vision/medical_imaging/3d-unet-kits19) | pytorch, tensorflow, onnx | KiTS19 | edge |
| Wan2.2-T2V-A14B-Diffusers | [text_to_video](https://github.com/mlcommons/inference/tree/master/text_to_video) | pytorch | COCO 2014 | datacenter |
| llama2-70b | [language/llama2-70b](https://github.com/mlcommons/inference/tree/master/language/llama2-70b) | pytorch | OpenOrca | datacenter |
| llama3.1-405b | [language/llama3-405b](https://github.com/mlcommons/inference/tree/master/language/llama3.1-405b) | pytorch | LongBench, LongDataCollections, Ruler, GovReport | datacenter |
| mixtral-8x7b | [language/mixtral-8x7b](https://github.com/mlcommons/inference/tree/master/language/mixtral-8x7b) | pytorch | OpenOrca, MBXP, GSM8K | datacenter |
| rgat | [graph/rgat](https://github.com/mlcommons/inference/tree/master/graph/R-GAT) | pytorch | IGBH | datacenter |
| pointpainting | [automotive/3d-object-detection](https://github.com/mlcommons/inference/tree/master/automotive/3d-object-detection) | pytorch, onnx | Waymo Open Dataset | edge |
| llama3.1-8b | [language/llama3.1-8b](https://github.com/mlcommons/inference/tree/master/language/llama3.1-8b)| pytorch | CNN-Daily Mail | edge,datacenter |
| deepseek-r1 | [language/deepseek-r1](https://github.com/mlcommons/inference/tree/master/language/deepseek-r1)| pytorch | AIME, MATH500, gpqa, MMLU-Pro, livecodebench(code_generation_lite) | datacenter |
| whisper | [speech2text](https://github.com/mlcommons/inference/tree/master/speech2text)| pytorch | LibriSpeech | edge,datacenter |
| GPT–OSS | [language/gpt-oss-120b](https://github.com/mlcommons/inference/tree/master/language/gpt-oss-120b)| pytorch | mlperf_gpt_oss_performance, mlperf_gpt_oss_accuracy | datacenter |
| VLM | [language/VLM](https://github.com/mlcommons/inference/tree/master/multimodal/qwen3-vl)| pytorch | Shopify-product-catalogue | datacenter |

## MLPerf Inference v5.1 (submission deadline August 1, 2025)

For submissions, please use the master branch and any commit since the [5.1 seed release]() although it is best to use the latest commit in the [master branch](https://github.com/mlcommons/inference).

For power submissions please use [SPEC PTD 1.11.1](https://github.com/mlcommons/power) (needs special access) and any commit of the power-dev repository after the [code-freeze](https://github.com/mlcommons/power-dev/commit/c4b3ad8202fbd8ac28d77149e5e7aeadb725bbf2)


| model | reference app | framework | dataset | category
| ---- | ---- | ---- | ---- | ---- |
| resnet50-v1.5 | [vision/classification_and_detection](https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection) | tensorflow, onnx, tvm, ncnn | imagenet2012 | edge |
| retinanet 800x800 | [vision/classification_and_detection](https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection) | pytorch, onnx | openimages resized to 800x800| edge,datacenter |
| bert | [language/bert](https://github.com/mlcommons/inference/tree/master/language/bert) | tensorflow, pytorch, onnx | squad-1.1 | edge |
| dlrm-v2 | [recommendation/dlrm_v2](https://github.com/mlcommons/inference/tree/master/recommendation/dlrm_v2/pytorch) | pytorch | Multihot Criteo Terabyte | datacenter |
| 3d-unet | [vision/medical_imaging/3d-unet-kits19](https://github.com/mlcommons/inference/tree/master/vision/medical_imaging/3d-unet-kits19) | pytorch, tensorflow, onnx | KiTS19 | edge |
| stable-diffusion-xl | [text_to_image](https://github.com/mlcommons/inference/tree/master/text_to_image) | pytorch | COCO 2014| edge,datacenter |
| llama2-70b | [language/llama2-70b](https://github.com/mlcommons/inference/tree/master/language/llama2-70b) | pytorch | OpenOrca | datacenter |
| llama3.1-405b | [language/llama3-405b](https://github.com/mlcommons/inference/tree/master/language/llama3.1-405b) | pytorch | LongBench, LongDataCollections, Ruler, GovReport | datacenter |
| mixtral-8x7b | [language/mixtral-8x7b](https://github.com/mlcommons/inference/tree/master/language/mixtral-8x7b) | pytorch | OpenOrca, MBXP, GSM8K | datacenter |
| rgat | [graph/rgat](https://github.com/mlcommons/inference/tree/master/graph/R-GAT) | pytorch | IGBH | datacenter |
| pointpainting | [automotive/3d-object-detection](https://github.com/mlcommons/inference/tree/master/automotive/3d-object-detection) | pytorch, onnx | Waymo Open Dataset | edge |
| llama3.1-8b | [language/llama3.1-8b](https://github.com/mlcommons/inference/tree/master/language/llama3.1-8b)| pytorch | CNN-Daily Mail | edge,datacenter |
| deepseek-r1 | [language/deepseek-r1](https://github.com/mlcommons/inference/tree/master/language/deepseek-r1)| pytorch | AIME, MATH500, gpqa, MMLU-Pro, livecodebench(code_generation_lite) | datacenter |
| whisper | [speech2text](https://github.com/mlcommons/inference/tree/master/speech2text)| pytorch | LibriSpeech | edge,datacenter |

* Framework here is given for the reference implementation. Submitters are free to use their own frameworks to run the benchmark.


## MLPerf Inference v5.0 (submission deadline February 28, 2025)

For submissions, please use the master branch and any commit since the [5.0 seed release](https://github.com/mlcommons/inference/commit/5d83ed5de438ffb55bca4cdb2966fba90a9dbca6) although it is best to use the latest commit in the [master branch](https://github.com/mlcommons/inference).

For power submissions please use [SPEC PTD 1.11.1](https://github.com/mlcommons/power) (needs special access) and any commit of the power-dev repository after the [code-freeze](https://github.com/mlcommons/power-dev/commit/65eedd4a60b5c50ac44cbae061d2a428e9fb190a)


| model | reference app | framework | dataset | category
| ---- | ---- | ---- | ---- | ---- |
| resnet50-v1.5 | [vision/classification_and_detection](https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection) | tensorflow, onnx, tvm, ncnn | imagenet2012 | edge,datacenter |
| retinanet 800x800 | [vision/classification_and_detection](https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection) | pytorch, onnx | openimages resized to 800x800| edge,datacenter |
| bert | [language/bert](https://github.com/mlcommons/inference/tree/master/language/bert) | tensorflow, pytorch, onnx | squad-1.1 | edge |
| dlrm-v2 | [recommendation/dlrm_v2](https://github.com/mlcommons/inference/tree/master/recommendation/dlrm_v2/pytorch) | pytorch | Multihot Criteo Terabyte | datacenter |
| 3d-unet | [vision/medical_imaging/3d-unet-kits19](https://github.com/mlcommons/inference/tree/master/vision/medical_imaging/3d-unet-kits19) | pytorch, tensorflow, onnx | KiTS19 | edge,datacenter |
| gpt-j | [language/gpt-j](https://github.com/mlcommons/inference/tree/master/language/gpt-j)| pytorch | CNN-Daily Mail | edge,datacenter |
| stable-diffusion-xl | [text_to_image](https://github.com/mlcommons/inference/tree/master/text_to_image) | pytorch | COCO 2014| edge,datacenter |
| llama2-70b | [language/llama2-70b](https://github.com/mlcommons/inference/tree/master/language/llama2-70b) | pytorch | OpenOrca | datacenter |
| llama3.1-405b | [language/llama3-405b](https://github.com/mlcommons/inference/tree/master/language/llama3.1-405b) | pytorch | LongBench, LongDataCollections, Ruler, GovReport | datacenter |
| mixtral-8x7b | [language/mixtral-8x7b](https://github.com/mlcommons/inference/tree/master/language/mixtral-8x7b) | pytorch | OpenOrca, MBXP, GSM8K | datacenter |
| rgat | [graph/rgat](https://github.com/mlcommons/inference/tree/master/graph/R-GAT) | pytorch | IGBH | datacenter |
| pointpainting | [automotive/3d-object-detection](https://github.com/mlcommons/inference/tree/master/automotive/3d-object-detection) | pytorch, onnx | Waymo Open Dataset | edge |

* Framework here is given for the reference implementation. Submitters are free to use their own frameworks to run the benchmark.


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
