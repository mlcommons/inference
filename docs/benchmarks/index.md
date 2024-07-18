# MLPerf Inference Benchmarks

Please visit the individual benchmark links to see the run commands using the unified CM interface.

1. [Image Classification](image_classification/resnet50.md) using ResNet50-v1.5 model and Imagenet-2012 (224x224) validation dataset. Dataset size is 50,000 and QSL size is 1024. Reference model accuracy is 76.46%. Server scenario latency constraint is 15ms.

2. [Text to Image](text_to_image/sdxl.md) using Stable Diffusion model and subset of Coco2014 dataset. Dataset size is 5000 amd QSL size is the same. Required accuracy for closed division is (23.01085758 <= FID <= 23.95007626, 32.68631873 <= CLIP <= 31.81331801).

3. [Object Detection](object_detection/retinanet.md) using Retinanet model and OpenImages dataset.Dataset size is 24781 and QSL size is 64. Reference model accuracy is 0.3755 mAP. Server scenario latency constraint is 100ms.

4. [Medical Image Segmentation](medical_imaging/3d-unet.md)  using 3d-unet model and KiTS2019 dataset. Dataset size is 42 and QSL size is the same. Reference model accuracy is 0.86330 mean DIXE score. Server scenario is not applicable.

5. [Question Answering](language/bert.md) using Bert-Large model and Squad v1.1 dataset with 384 sequence length. Dataset size is 10833 and QSL size is the same. Reference model accuracy is f1 score = 90.874%. Server scenario latency constraint is 130ms.

6. [Text Summarization](language/gpt-j.md) using GPT-J model and CNN Daily Mail v3.0.0 dataset. Dataset size is 13368 amd QSL size is the same. Reference model accuracy is (rouge1=42.9865, rouge2=20.1235, rougeL=29.9881, gen_len=4016878). Server scenario latency sconstraint is 20s.

7. [Question Answering](language/llama2-70b.md) using LLAMA2-70b model and OpenORCA (GPT-4 split, max_seq_len=1024) dataset. Dataset size is 24576 and QSL size is the same. Reference model accuracy is (rouge1=44.4312, rouge2=22.0352, rougeL=28.6162, tokens_per_sample=294.45). Server scenario latency constraint is TTFT=2000ms, TPOT=200ms.

8. [Question Answering, Math and Code Generation](language/mixtral-8x7b.md) using Mixtral-8x7B model and OpenORCA (5k samples of GPT-4 split, max_seq_len=2048), GSM8K (5k samples of the validation split, max_seq_len=2048), MBXP (5k samples of the validation split, max_seq_len=2048) datasets. Dataset size is 15000 and QSL size is the same. Reference model accuracy is (rouge1=45.4911, rouge2=23.2829, rougeL=30.3615, gsm8k accuracy = 73.78, mbxp accuracy = 60.12, tokens_per_sample=294.45). Server scenario latency constraint is TTFT=2000ms, TPOT=200ms.

9. [Recommendation](recommendation/dlrm-v2.md) using DLRMv2 model and Synthetic Multihot Criteo dataset. Dataset size is 204800 and QSL size is the same. Reference model accuracy is AUC=80.31%. Server scenario latency constraint is 60 ms. 

All the nine benchmarks can participate in the datacenter category.
All the nine benchmarks except DLRMv2, LLAMA2 and Mixtral-8x7B and can participate in the edge category. 

`bert`, `llama2-70b`, `dlrm_v2` and `3d-unet` has a high accuracy (99.9%) variant, where the benchmark run  must achieve a higher accuracy of at least `99.9%` of the FP32 reference model
in comparison with the `99%` default accuracy requirement.
