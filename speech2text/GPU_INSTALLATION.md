# MLPerf Whisper Inference - Complete Installation Guide

## Target System: IBM Cloud Node (RHEL 9.6 with NVIDIA L40S GPUs)

**SSH Access:**
```bash
ssh -i /home/user/.ssh/key root@16.74.xx.xx
```

---

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Step 1: Clone MLPerf Inference Repository](#step-1-clone-mlperf-inference-repository)
3. [Step 2: Install Miniconda](#step-2-install-miniconda)
4. [Step 3: Create Python Environment](#step-3-create-python-environment)
5. [Step 4: Install PyTorch](#step-4-install-pytorch)
6. [Step 5: Install vLLM and Dependencies](#step-5-install-vllm-and-dependencies)
7. [Step 6: Install MLPerf LoadGen](#step-6-install-mlperf-loadgen)
8. [Step 7: Download Whisper Model](#step-7-download-whisper-model)
9. [Step 8: Download Dataset](#step-8-download-dataset)
10. [Step 9: Configure Environment Variables](#step-9-configure-environment-variables)
11. [Step 10: Run Benchmark](#step-10-run-benchmark)

---

## System Requirements

- **OS**: Red Hat Enterprise Linux 9.6
- **GPUs**: NVIDIA L40S (2x GPUs)
- **CUDA**: 12.8
- **Python**: 3.12
- **NUMA Nodes**: 2
- **CPU Cores**: 48 (24 per NUMA node)

---

## Step 1: Clone MLPerf Inference Repository

```bash
# Navigate to home directory
cd ~

# Clone the MLPerf inference repository
git clone --recurse-submodules https://github.com/mlcommons/inference.git

# Navigate to inference directory
cd inference

# Checkout master branch (for v6.0)
git checkout master
```

---

## Step 2: Install Miniconda

```bash
# Create directory for miniconda
mkdir -p ~/miniconda3

# Download Miniconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh

# Install Miniconda
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3

# Initialize conda for bash
~/miniconda3/bin/conda init bash

# Reload shell configuration
source ~/.bashrc
```

---

## Step 3: Create Python Environment

```bash
# Accept Conda Terms of Service
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create environment with Python 3.12
conda create -y -n whisper-gpu-fresh python=3.12

# Activate the environment
conda activate whisper-gpu-fresh
```

**Verify Python Installation:**
```bash
python --version
# Expected output: Python 3.12.12
```

---

## Step 4: Install PyTorch

```bash
# Install PyTorch 2.9.0 with CUDA 12.8 support
pip install --no-cache-dir \
    torch==2.9.0 \
    torchvision==0.24.0 \
    torchaudio==2.9.0
```

**Verify PyTorch Installation:**
```bash
python << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
EOF
```

**Expected Output:**
```
PyTorch version: 2.9.0
CUDA available: True
CUDA version: 12.8
GPU count: 2
GPU 0: NVIDIA L40S
GPU 1: NVIDIA L40S
```

---

## Step 5: Install vLLM and Dependencies

```bash
# Install vLLM 0.12.0
pip install --no-cache-dir \
    --index-url https://pypi.org/simple \
    vllm==0.12.0

# Install data processing libraries
pip install --no-cache-dir \
    pandas==2.2.2 \
    librosa==0.10.2 \
    numpy==2.0.1

# Install Whisper library
pip install --no-cache-dir \
    openai-whisper==20250625

# Install additional dependencies
pip install --no-cache-dir \
    toml==0.10.2 \
    unidecode==1.3.8 \
    inflect==7.3.1 \
    setuptools-scm \
    py-libnuma==1.2
```

**Verify vLLM Installation:**
```bash
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
```

---

## Step 6: Install MLPerf LoadGen

```bash
# Navigate to loadgen directory
cd ~/inference/loadgen

# Install LoadGen in editable mode
pip install --no-cache-dir \
    --index-url https://pypi.org/simple \
    -e .

# Verify installation
python -c "import mlperf_loadgen as lg; print('LoadGen installed successfully')"
```

---

## Step 7: Download Whisper Model

```bash
# Navigate to speech2text directory
cd ~/inference/speech2text

# Download Whisper Large V3 model using MLCommons R2 Downloader
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  -d ./model/whisper-large-v3 \
  https://inference.mlcommons-storage.org/metadata/whisper-model.uri
```

**Verify Model Download:**
```bash
ls -lh model/whisper-large-v3/
# Should contain model files including config.json and model weights
```

---

## Step 8: Download Dataset

```bash
# Navigate to speech2text directory
cd ~/inference/speech2text

# Download LibriSpeech dataset using MLCommons R2 Downloader
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  -d ./data \
  https://inference.mlcommons-storage.org/metadata/whisper-dataset.uri
```

**Note:** The dataset may be downloaded with a different manifest filename. Check the actual filename:
```bash
ls -lh data/data/*.json
```

If the manifest is named `dev-all-repack.json` but you need `dev-all-repack-fixed.json`, you may need to create or copy it.

---

## Step 9: Configure Environment Variables

```bash
# Navigate to speech2text directory
cd ~/inference/speech2text

# Set workspace paths
export WORKSPACE_DIR=$(pwd)
export DATA_DIR=${WORKSPACE_DIR}/data/data
export MODEL_PATH=${WORKSPACE_DIR}/model/whisper-large-v3
export MANIFEST_FILE="${DATA_DIR}/dev-all-repack-fixed.json"

# Set scenario (Offline for throughput testing)
export SCENARIO="Offline"

# Single GPU configuration
export CUDA_VISIBLE_DEVICES=0
export NUM_INSTS=1
export NUM_NUMA_NODES=1
export INSTS_PER_NODE=1
export CORES_PER_INST=24
export START_CORES="0"

# For Dual GPU configuration (recommended)
export CUDA_VISIBLE_DEVICES=0,1
export NUM_INSTS=2
export NUM_NUMA_NODES=2
export INSTS_PER_NODE=1
export CORES_PER_INST=24
export START_CORES="0,24"

# Additional settings
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

**Verify Dataset and Model:**
```bash
# Check dataset manifest exists
ls -lh $MANIFEST_FILE

# Check model directory exists
ls -lh $MODEL_PATH/

# Verify manifest has samples
python -c "import json; data=json.load(open('$MANIFEST_FILE')); print(f'Dataset has {len(data)} samples')"
```

**Expected Output:**
```
Dataset has 1633 samples
```

---

## Step 10: Run Benchmark

### Create Log Directory
```bash
mkdir -p logs
```

### Performance Run (No Accuracy Calculation)
```bash
python reference_mlperf.py \
    --dataset_dir ${DATA_DIR} \
    --model_path ${MODEL_PATH} \
    --manifest ${MANIFEST_FILE} \
    --scenario ${SCENARIO} \
    --log_dir logs/performance_2gpu \
    --num_workers ${NUM_INSTS}
```

### Accuracy Run
```bash
python reference_mlperf.py \
    --dataset_dir ${DATA_DIR} \
    --model_path ${MODEL_PATH} \
    --manifest ${MANIFEST_FILE} \
    --scenario ${SCENARIO} \
    --log_dir logs/accuracy_2gpu \
    --num_workers ${NUM_INSTS} \
    --accuracy
```

### Monitor GPU/CPU Utilization (in separate terminal)
```bash
# SSH to IBM node in a new terminal
ssh -i /home/user/.ssh/key root@163.74.xx.xx

# Monitor GPUs
nvidia-smi -l 

# Monitor GPUs
top
htop
```

---

## Expected Performance Results

### Dual GPU (L40S) - Offline Scenario
```
Samples per second: ~2,400
Tokens per second: ~2,300
GPU Utilization: 75-90% average (GPU 1 heavily utilized)
Duration: ~10-12 minutes for full dataset (1,633 samples)
Result: VALID
```

### Configuration Summary
- Dataset: 1,633 samples (~10.91 hours of audio)
- Precision: bfloat16
- Max Model Length: 448 tokens
- GPU Memory Utilization: 95%
- Max Batched Tokens: 32,000
- Max Sequences: 256

---

## Verify Results

```bash
# View performance summary
cat logs/performance_2gpu/mlperf_log_summary.txt

# Check if run is valid
grep "Result is" logs/performance_2gpu/mlperf_log_summary.txt

# View accuracy (if accuracy run was performed)
cat logs/accuracy_2gpu/accuracy.txt
```

---

**Last Updated**: 2026-01-28
**Environment**: IBM L40s RHEL 9.6
