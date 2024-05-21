# MLPerf Inference Benchmarks

Please visit the individual benchmark links to see the run commands using the unified CM interface.

1. [Image Classification](image_classification/resnet50.md) using ResNet50 model and Imagenet-2012 dataset

2. [Text to Image](text_to_image/sdxl.md) using Stable Diffusion model and Coco2014 dataset

3. [Object Detection](object_detection/retinanet.md) using Retinanet model and OpenImages dataset

4. [Image Segmentation](medical_imaging/3d-unet.md)  using 3d-unet model and KiTS19 dataset

5. [Question Answering](language/bert.md) using Bert-Large model and Squad v1.1 dataset

6. [Text Summarization](language/gpt-j.md) using GPT-J model and CNN Daily Mail dataset

7. [Text Summarization](language/llama2-70b.md) using LLAMA2-70b model and OpenORCA dataset

8. [Recommendation](recommendation/dlrm-v2.md) using DLRMv2 model and Criteo multihot dataset

All the eight benchmarks can participate in the datacenter category.
All the eight benchmarks except DLRMv2 and LLAMA2 and can participate in the edge category. 

`bert`, `llama2-70b`, `dlrm_v2` and `3d-unet` has a high accuracy (99.9%) variant, where the benchmark run  must achieve a higher accuracy of at least `99.9%` of the FP32 reference model
in comparison with the `99%` default accuracy requirement.

The `dlrm_v2` benchmark has a high-accuracy variant only. If this accuracy is not met, the submission result can be submitted only to the open division.

