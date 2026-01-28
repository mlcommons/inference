# MLPerf Whisper - Quick Reference Card

## SSH to Node
```bash
ssh -i /home/user/.ssh/key root@10.74.xx.xx
```

## Activate Environment
```bash
conda activate whisper-gpu
cd ~/inference/speech2text
```

## Environment Setup (Dual GPU)
```bash
export WORKSPACE_DIR=$(pwd)
export DATA_DIR=${WORKSPACE_DIR}/data/data
export MODEL_PATH=${WORKSPACE_DIR}/model/whisper-large-v3
export MANIFEST_FILE="${DATA_DIR}/dev-all-repack-fixed.json"
export SCENARIO="Offline"

# Dual GPU config
export CUDA_VISIBLE_DEVICES=0,1
export NUM_INSTS=2
export NUM_NUMA_NODES=2
export INSTS_PER_NODE=1
export CORES_PER_INST=24
export START_CORES="0,24"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

## Run Performance Test
```bash
python reference_mlperf.py \
    --dataset_dir ${DATA_DIR} \
    --model_path ${MODEL_PATH} \
    --manifest ${MANIFEST_FILE} \
    --scenario ${SCENARIO} \
    --log_dir logs/perf_$(date +%Y%m%d_%H%M%S) \
    --num_workers ${NUM_INSTS}
```

## Run Accuracy Test
```bash
python reference_mlperf.py \
    --dataset_dir ${DATA_DIR} \
    --model_path ${MODEL_PATH} \
    --manifest ${MANIFEST_FILE} \
    --scenario ${SCENARIO} \
    --log_dir logs/acc_$(date +%Y%m%d_%H%M%S) \
    --num_workers ${NUM_INSTS} \
    --accuracy
```

## Monitor GPUs (Separate Terminal)
```bash
nvidia-smi -l
```

## Check Results
```bash
# View summary
tail -50 logs/*/mlperf_log_summary.txt

# Check if valid
grep "Result is" logs/*/mlperf_log_summary.txt

# View metrics
grep "Samples per second" logs/*/mlperf_log_summary.txt
grep "Tokens per second" logs/*/mlperf_log_summary.txt
```

## Installed Versions
- Python: 3.12.12
- PyTorch: 2.9.0 (CUDA 12.8)
- vLLM: 0.12.0
- Whisper: 20250625
- Dataset: 1,633 samples

## Expected Performance (Dual L40S)
- Throughput: ~2,400 samples/sec
- Tokens/sec: ~2,300
- Duration: ~10-12 minutes
- GPU Util: 75-90% avg
