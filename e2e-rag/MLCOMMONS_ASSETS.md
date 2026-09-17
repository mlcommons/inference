# Dataset and Model Downloads

All required models and datasets for the E2E-RAG benchmark are hosted on MLCommons storage for reliability and reproducibility.

## Quick Download All Assets

```bash
bash scripts/download_dataset_and_models.sh
```

Total download size: **~283GB**

---

## Individual Asset Downloads

### 1. FRAMES Dataset

Multi-hop questions requiring information synthesis from multiple Wikipedia articles.

- **Size:** ~674KB
- **Description:** 824 multi-hop question-answer pairs with Wikipedia source URLs
- **Default location:** `data/frames_dataset.tsv`

**Download:**
```bash
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  https://inference.mlcommons-storage.org/metadata/frames-benchmark-dataset.uri
```

---

### 2. Embedding Model (intfloat/e5-base-v2)

Multilingual text embedding model for semantic search and retrieval.

- **Size:** ~2.2GB
- **Architecture:** BERT-based encoder (768-dim embeddings)
- **Max sequence length:** 512 tokens
- **Use case:** Document and query embedding for vector retrieval
- **Default location:** `/data/model/e5-base-v2/`

**Download:**
```bash
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  https://inference.mlcommons-storage.org/metadata/intfloat_e5-base-v2.uri
```

**Configuration:**
- Set `RETRIEVER_MODEL=/data/model/e5-base-v2` in workload scripts
- Used by both database setup and QA workloads

---

### 3. Reranker Model (ColBERTv2.0)

Late-interaction neural reranker for improving retrieval precision.

- **Size:** ~1.4GB
- **Architecture:** Token-level contextualized embeddings with MaxSim scoring
- **Use case:** Rerank top-K retrieved passages for precision
- **Performance gain:** ~5-10% improvement over retrieval alone
- **Default location:** `/data/model/colbertv2.0/`

**Download:**
```bash
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  https://inference.mlcommons-storage.org/metadata/colbert-ir_colbertv2.0.uri
```

**Configuration:**
- Set `RERANKER_MODEL=/data/model/colbertv2.0` in workload scripts
- Used in both database setup and QA workloads
- Can be disabled with `--no-rerank` flag for testing

---

### 4. GPT-OSS-120B Model

Large language model for answer generation, query rewriting, and sufficiency checking.

- **Size:** ~196GB
- **Use cases:**
  - Answer generation from retrieved passages
  - Query decomposition (multi-shot retrieval)
  - Sufficiency checking (deciding if enough info is retrieved)
- **Default location:** `/data/model/gpt-oss-120b/`

**Download:**
```bash
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  https://inference.mlcommons-storage.org/metadata/gpt-oss-model.uri
```

**Configuration:**
- Set `LLM_MODEL=gpt-oss-120b` in QA workload scripts
- Set `QUERY_MODEL=gpt-oss-120b` for query decomposition
- Requires vLLM server on `QUERY_SERVICE_URL` (default: `http://127.0.0.1:8124/v1/chat/completions`)

**vLLM Server:**
```bash
vllm serve /data/model/gpt-oss-120b --port 8124 --gpu-memory-util 0.95
```

---

### 5. GPT-OSS-20B Model

Smaller language model for document relevance grading.

- **Size:** ~83GB
- **Use cases:**
  - Document relevance grading (binary relevant/not relevant)
  - Fast classification tasks in multi-shot retrieval
- **Default location:** `/data/model/gpt-oss-20b/`

**Download:**
```bash
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  https://inference.mlcommons-storage.org/metadata/gpt-oss-20B.uri
```

**Configuration:**
- Set `LLM_MODEL=gpt-oss-20b` for grading tasks
- Requires vLLM server on `LLM_SERVICE_URL` (default: `http://127.0.0.1:8123/v1/chat/completions`)

**vLLM Server:**
```bash
vllm serve /data/model/gpt-oss-20b --port 8123 --gpu-memory-util 0.95
```

---

## Model Roles in Workloads

### Database Setup Workload

| Model | Purpose |
|---|---|
| e5-base-v2 | Generate embeddings for document chunks |
| ColBERTv2.0 | (Optional) Pre-compute reranker features |

### Question-Answering Workload (Multi-Shot)

| Model | Purpose | Config Variable |
|---|---|---|
| e5-base-v2 | Embed queries and retrieve passages | `RETRIEVER_MODEL` |
| ColBERTv2.0 | Rerank retrieved passages | `RERANKER_MODEL` |
| GPT-OSS-20B | Grade document relevance | `LLM_MODEL` |
| GPT-OSS-120B | Decompose queries, generate answers | `QUERY_MODEL` |
| Judge Model | Evaluate answer quality (accuracy mode) | `JUDGE_MODEL` |

---

## System Requirements

### Disk Space

- Models: ~283GB
- Wikipedia documents: ~5GB (HTML) or ~10GB (PDF)
- Vector database: ~15GB (768-char chunks, HNSW index)
- **Total recommended:** 350GB+

### Memory

- Database setup: 32GB+ RAM (64GB recommended)
- QA workload: 64GB+ RAM (128GB recommended with LLMs loaded)

### GPU (Optional)

- Embedding/reranking: 8GB+ VRAM
- LLM inference:
  - GPT-OSS-20B: ~40GB VRAM
  - GPT-OSS-120B: ~120GB VRAM (multi-GPU recommended)
- **Recommended:** Multi-GPU setup (A100/H100) for concurrent LLM serving

---

## Verification

After downloading, verify assets are in place:

```bash
# Check dataset
ls -lh data/frames_dataset.tsv

# Check models
ls -lh /data/model/e5-base-v2/
ls -lh /data/model/colbertv2.0/
ls -lh /data/model/gpt-oss-20b/
ls -lh /data/model/gpt-oss-120b/

# Check Wikipedia documents (after download_docs.py)
ls -lh doc_html/*.html | wc -l  # Should show ~2000 files
```

---

## Troubleshooting

### Download interrupted or failed

The MLCommons downloader supports resume. Simply re-run the download command.

### Disk space issues

Download models individually as needed:
- For development: Start with e5-base-v2 + ColBERTv2.0 (~4GB)
- For QA workload: Add GPT-OSS models as needed
- Use smaller models: Substitute with smaller instruction-tuned models (e.g., Llama-3.1-8B)

### Network issues

Use a download manager or set up a local mirror:
```bash
# Example with aria2c for faster parallel downloads
aria2c -x 16 -s 16 <MLCommons URL>
```

---

## Support

For issues with MLCommons downloader or hosted assets:
- MLCommons GitHub: https://github.com/mlcommons/r2-downloader
- MLCommons Inference: https://github.com/mlcommons/inference

For E2E-RAG benchmark issues:
- See main [README.md](README.md)
- Check [Troubleshooting](README.md#troubleshooting) section
