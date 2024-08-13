# MLPerf Inference Benchmarks

## Overview
This document provides details on various MLPerf Inference Benchmarks categorized by tasks, models, and datasets. Each section lists the models performing similar tasks, with details on datasets, accuracy, and server latency constraints.

---

## 1. Image Classification
### [ResNet50-v1.5](image_classification/resnet50.md)
- **Dataset**: Imagenet-2012 (224x224) Validation
  - **Size**: 50,000
  - **QSL Size**: 1,024
- **Reference Model Accuracy**: 76.46%
- **Server Scenario Latency Constraint**: 15ms

---

## 2. Text to Image
### [Stable Diffusion](text_to_image/sdxl.md)
- **Dataset**: Subset of Coco2014
  - **Size**: 5,000
  - **QSL Size**: 5,000
- **Required Accuracy (Closed Division)**:
  - FID: 23.01085758 ≤ FID ≤ 23.95007626
  - CLIP: 32.68631873 ≤ CLIP ≤ 31.81331801

---

## 3. Object Detection
### [Retinanet](object_detection/retinanet.md)
- **Dataset**: OpenImages
  - **Size**: 24,781
  - **QSL Size**: 64
- **Reference Model Accuracy**: 0.3755 mAP
- **Server Scenario Latency Constraint**: 100ms

---

## 4. Medical Image Segmentation
### [3d-unet](medical_imaging/3d-unet.md)
- **Dataset**: KiTS2019
  - **Size**: 42
  - **QSL Size**: 42
- **Reference Model Accuracy**: 0.86330 Mean DICE Score
- **Server Scenario**: Not Applicable

---

## 5. Language Tasks

### 5.1. Question Answering

### [Bert-Large](language/bert.md)
- **Dataset**: Squad v1.1 (384 Sequence Length)
  - **Size**: 10,833
  - **QSL Size**: 10,833
- **Reference Model Accuracy**: F1 Score = 90.874%
- **Server Scenario Latency Constraint**: 130ms

### [LLAMA2-70B](language/llama2-70b.md)
- **Dataset**: OpenORCA (GPT-4 split, max_seq_len=1024)
  - **Size**: 24,576
  - **QSL Size**: 24,576
- **Reference Model Accuracy**:
  - Rouge1: 44.4312
  - Rouge2: 22.0352
  - RougeL: 28.6162
  - Tokens_per_sample: 294.45
- **Server Scenario Latency Constraint**:
  - TTFT: 2000ms
  - TPOT: 200ms

### 5.2. Text Summarization

### [GPT-J](language/gpt-j.md)
- **Dataset**: CNN Daily Mail v3.0.0
  - **Size**: 13,368
  - **QSL Size**: 13,368
- **Reference Model Accuracy**:
  - Rouge1: 42.9865
  - Rouge2: 20.1235
  - RougeL: 29.9881
  - Gen_len: 4,016,878
- **Server Scenario Latency Constraint**: 20s

### 5.3. Mixed Tasks (Question Answering, Math, and Code Generation)

### [Mixtral-8x7B](language/mixtral-8x7b.md)
- **Datasets**:
  - OpenORCA (5k samples of GPT-4 split, max_seq_len=2048)
  - GSM8K (5k samples of the validation split, max_seq_len=2048)
  - MBXP (5k samples of the validation split, max_seq_len=2048)
  - **Size**: 15,000
  - **QSL Size**: 15,000
- **Reference Model Accuracy**:
  - Rouge1: 45.4911
  - Rouge2: 23.2829
  - RougeL: 30.3615
  - GSM8K Accuracy: 73.78%
  - MBXP Accuracy: 60.12%
  - Tokens_per_sample: 294.45
- **Server Scenario Latency Constraint**:
  - TTFT: 2000ms
  - TPOT: 200ms

---

## 6. Recommendation
### [DLRMv2](recommendation/dlrm-v2.md)
- **Dataset**: Synthetic Multihot Criteo
  - **Size**: 204,800
  - **QSL Size**: 204,800
- **Reference Model Accuracy**: AUC = 80.31%
- **Server Scenario Latency Constraint**: 60ms

---

### Participation Categories
- **Datacenter Category**: All nine benchmarks can participate.
- **Edge Category**: All benchmarks except DLRMv2, LLAMA2, and Mixtral-8x7B can participate.

### High Accuracy Variants
- **Benchmarks**: `bert`, `llama2-70b`, `dlrm_v2`, and `3d-unet`
- **Requirement**: Must achieve at least 99.9% of the FP32 reference model accuracy, compared to the default 99% accuracy requirement.
