# E2E: RAG Benchmark

End-to-end retrieval-augmented generation benchmark for multi-hop question answering on the [FRAMES](https://huggingface.co/datasets/google/frames-benchmark) Wikipedia dataset.

**Key features:**
- MLPerf Loadgen integration for standardized benchmarking
- Dense vector retrieval with FAISS (HNSW/IVF/Flat indexing)
- ColBERTv2 reranking
- Iterative multi-shot retrieval with LLM-driven query decomposition
- Cross-vendor hardware support (Intel XPU, AMD GPU/ROCm, NVIDIA GPU/CUDA, CPU)
- Separate performance and accuracy testing modes

---

## Table of Contents

- [Environment Setup](#environment-setup)
- [Typical Workflow](#typical-workflow)
- [Step 1: Download Models and Data](#step-1-download-models-and-data-one-time)
- [Step 2: Build Vector Database](#step-2-build-vector-database-one-time-measured-operation)
- [Step 3: Run Question-Answering Workload](#step-3-run-question-answering-workload)
- [Common Configuration](#common-configuration)
- [Prerequisites](#prerequisites)
- [License](#license)

---

###  Environment setup

Use an Ubuntu Docker container and install dependencies:

```bash
# Start Ubuntu container
docker run -it --gpus all -v $(pwd):/workspace ubuntu:22.04

# Inside container: install dependencies
cd /workspace
bash setup.sh
```

The `setup.sh` script installs all required Python packages and system dependencies.


### Typical Workflow

```
1. Download Wikipedia documents → 2. Build vector database → 3. Run QA workload
   (download_docs.py)              (reference_mlperf_datasetup.sh)   (reference_mlperf_accuracy.sh)
```

### Step 1: Download models and data (one-time)

```bash
bash scripts/download_dataset_and_models.sh
```

This downloads all required models and datasets from MLCommons storage (~283GB).

**Then download Wikipedia documents:**
```bash
python3 download_docs.py --output_dir doc_html --format html --processes 30
```

### Step 2: Build vector database (one-time measured operation)

**Performance mode:**
```bash
DOCUMENTS_DIR=doc_html \
DATABASE=vector_html_hnsw_len768_ov32_word \
DEVICE=auto \
bash reference_mlperf_datasetup.sh
```

**Accuracy mode** (with verification):
```bash
bash reference_mlperf_datasetup_accuracy.sh
```

**Output:** Creates `${DATABASE}.db` and `${DATABASE}_data/` directory.

### Step 3: Run question-answering workload

**Performance mode** (with cached LLM responses):
```bash
DATABASE=vector_html_hnsw_len768_ov32_word.db \
PERF_CACHE_FILE=logs_result.json \
bash reference_mlperf_perf.sh
```

**Accuracy mode** (live LLM inference):
```bash
# Start LLM servers first (see Prerequisites)
DATABASE=vector_html_hnsw_len768_ov32_word.db \
LLM_SERVICE_URL=http://127.0.0.1:8123/v1/chat/completions \
bash reference_mlperf_accuracy.sh
```

**Output:** Results in `${WORKSPACE_DIR}/output/` and logs in `${WORKSPACE_DIR}/run_output/`.

### Common Configuration

Key environment variables (see [Running Workloads](#running-workloads) for full list):

**Database Setup:**
- `DOCUMENTS_DIR`: HTML documents (default: `doc_html`)
- `DATABASE`: Database name (default: `vector_html_hnsw_len768_ov32_word`)
- `CHUNK_SIZE`: Chunk size in chars (default: `768`)
- `MAX_WORKERS`: Parallel threads (default: `4`)
- `DEVICE`: Hardware device (default: `auto`)

**Question-Answering:**
- `DATABASE`: Database file (default: `vector_html_hnsw_len768_ov32_word.db`)
- `PERF_COUNT`: Number of queries (default: `824`)
- `MAX_WORKERS`: Parallel workers (default: `10`)
- `MAX_ITERATIONS`: Max retrieval rounds (default: `5`)
- `LLM_SERVICE_URL`: Answer generation endpoint
- `QUERY_SERVICE_URL`: Query decomposition endpoint
- `JUDGE_SERVICE_URL`: Evaluation endpoint (accuracy mode only)

---

## Prerequisites

Before running workloads, ensure you have:

### Required Models and Data

All required models and datasets are hosted on MLCommons storage.

#### Quick Download All (Recommended)

Download all models and datasets at once:

```bash
bash scripts/download_dataset_and_models.sh
```

This downloads from MLCommons storage (~283GB total):
- **FRAMES Dataset** (~674KB) → `data/frames_dataset.tsv`
- **Embedding Model** e5-base-v2 (~2.2GB) → `/data/model/e5-base-v2/`
- **Reranker Model** ColBERTv2.0 (~1.4GB) → `/data/model/colbertv2.0/`
- **GPT-OSS-120B Model** (~196GB) → `/data/model/gpt-oss-120b/`
- **GPT-OSS-20B Model** (~83GB) → `/data/model/gpt-oss-20b/`

**Note:** Ensure sufficient disk space before proceeding.

#### Individual Downloads

See [MLCOMMONS_ASSETS.md](MLCOMMONS_ASSETS.md) for individual download commands and detailed model information.

#### LLM Server Setup

After downloading models, start vLLM servers:

```bash
# Answer generation + document grading (port 8123)
vllm serve /data/model/gpt-oss-20b --port 8123

# Query generation + sufficiency checking (port 8124)
vllm serve /data/model/gpt-oss-120b --port 8124

# Judge for evaluation (port 8125) - use any instruction-tuned model
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8125
```

### Wikipedia Documents

After downloading the FRAMES dataset, download Wikipedia pages referenced in it:

```bash
python3 download_docs.py --output_dir doc_html --format html --processes 30
```

This downloads ~2,000 Wikipedia pages in HTML format.

### System Requirements

- **Disk**: ~50GB for documents + vector DB
- **Memory**: 32GB+ RAM recommended
- **GPU** (optional): NVIDIA/AMD/Intel XPU for faster embedding/reranking
- **Python**: 3.8+
- **OS**: Ubuntu 20.04+ (or compatible container)

---

## License

See LICENSE file in repository root.
