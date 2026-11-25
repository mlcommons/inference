# Dataset and Model Downloads

All required models and datasets for the E2E-RAG benchmark are hosted on MLCommons storage for reliability and reproducibility.

## Quick Download All Assets

```bash
bash scripts/setup/download_dataset_and_models.sh
```

Total download size: **~83GB**

---

## Individual Asset Downloads

### 1. FRAMES Dataset

Multi-hop questions requiring information synthesis from multiple Wikipedia articles.

- **Size:** ~674KB
- **Description:** 824 multi-hop question-answer pairs with Wikipedia source URLs
- **Default location:** `frames-benchmark-dataset/frames-benchmark/test.tsv`,
  linked as `frames-benchmark-dataset/frames_dataset.tsv` by the download script

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
- **Default location:** `intfloat_e5-base-v2/e5-base-v2/`

**Download:**
```bash
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  https://inference.mlcommons-storage.org/metadata/intfloat_e5-base-v2.uri
```

**Configuration:**
- `EMBEDDING_MODEL` (default `intfloat_e5-base-v2/e5-base-v2`)
- Used by both database setup and QA workloads

---

### 3. Reranker Model (ColBERTv2.0)

Late-interaction neural reranker for improving retrieval precision.

- **Size:** ~1.4GB
- **Architecture:** Token-level contextualized embeddings with MaxSim scoring
- **Use case:** Rerank top-K retrieved passages for precision
- **Performance gain:** ~5-10% improvement over retrieval alone
- **Default location:** `colbert-ir_colbertv2.0/colbertv2.0/`

**Download:**
```bash
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  https://inference.mlcommons-storage.org/metadata/colbert-ir_colbertv2.0.uri
```

**Configuration:**
- `RERANKER_MODEL` (default `colbert-ir_colbertv2.0/colbertv2.0`)
- Used in both database setup and QA workloads
- Can be disabled with `--no-rerank` flag for testing

---

### 4. GPT-OSS-120B Model

Large language model for answer generation, query rewriting, and sufficiency checking.

- **Size:** ~65GB
- **Use cases:**
  - Answer generation from retrieved passages
  - Query decomposition (multi-shot retrieval)
  - Sufficiency checking (deciding if enough info is retrieved)
- **Default location:** `gpt-oss_model/gpt-oss-120b/`

**Download:** `scripts/setup/download_dataset_and_models.sh` fetches only the files
vLLM loads, verified against the bucket's MD5 list. It skips the `original/`
and `metal/` weight copies that the bucket also holds; the generic
MLCommons downloader (`.../metadata/gpt-oss-model.uri`) would fetch everything.

**Files vLLM loads:** `config.json`, `generation_config.json`,
`tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`,
`chat_template.jinja`, `model.safetensors.index.json` and
`model-00000-of-00014.safetensors` … `model-00014-of-00014.safetensors` (15 shards).

**Configuration:**
- `QUERY_MODEL` (default `gpt-oss_model/gpt-oss-120b`); `SUFFICIENCY_MODEL`
  follows it (same server)
- Served on `QUERY_SERVICE_URL` (default `http://127.0.0.1:8123/v1/chat/completions`)

**vLLM Server:**
```bash
DEVICES=0,1,2,3 bash scripts/servers/launch_llm_server.sh query
```

---

### 5. GPT-OSS-20B Model

Smaller language model for document relevance grading.

- **Size:** ~14GB
- **Use cases:**
  - Document relevance grading (binary relevant/not relevant)
  - Fast classification tasks in multi-shot retrieval
- **Default location:** `gpt-oss-20B/gpt-oss-20b/`

**Download:** `scripts/setup/download_dataset_and_models.sh` fetches only the files
vLLM loads, verified against the bucket's MD5 list. It skips the `original/`
and `metal/` weight copies and `.git/` that the bucket also holds; the generic
MLCommons downloader (`.../metadata/gpt-oss-20B.uri`) would fetch everything.

**Files vLLM loads:** `config.json`, `generation_config.json`,
`tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`,
`chat_template.jinja`, `model.safetensors.index.json` and
`model-00000-of-00002.safetensors` … `model-00002-of-00002.safetensors` (3 shards).

**Configuration:**
- `GRADER_MODEL` (default `gpt-oss-20B/gpt-oss-20b`)
- Served on `GRADER_SERVICE_URL` (default `http://127.0.0.1:8192/v1/chat/completions`)

**vLLM Server:**
```bash
DEVICES=4,5 bash scripts/servers/launch_llm_server.sh grader
```

---

### 6. Judge Model (Llama-3.1-8B-Instruct)

Scores answers in accuracy runs; not needed for performance runs.

- **Not hosted on MLCommons storage.** `JUDGE_MODEL` defaults to the Hugging
  Face id `meta-llama/Llama-3.1-8B-Instruct`, which vLLM downloads on first use
  (gated: accept the license and set `HF_TOKEN`), or point it at a local copy.
- Served on `JUDGE_SERVICE_URL` (default `http://127.0.0.1:8125/v1/chat/completions`)

**vLLM Server:**
```bash
DEVICES=6 bash scripts/servers/launch_llm_server.sh judge
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
| e5-base-v2 | Embed queries and retrieve passages | `EMBEDDING_MODEL` |
| ColBERTv2.0 | Rerank retrieved passages | `RERANKER_MODEL` |
| GPT-OSS-20B | Grade document relevance | `GRADER_MODEL` |
| GPT-OSS-120B | Decompose queries, check sufficiency, generate answers | `QUERY_MODEL`, `SUFFICIENCY_MODEL` |
| Llama-3.1-8B-Instruct | Evaluate answer quality (accuracy mode) | `JUDGE_MODEL` |

---

## System Requirements

### Disk Space

- Models: ~83GB, plus ~16GB for the judge (downloaded from Hugging Face)
- FRAMES dataset + Wikipedia corpus: ~1.2GB
- Vector database: ~0.5GB (768-char chunks, HNSW index)
- **Total recommended:** 120GB+

### Memory

- Database setup: 32GB+ RAM (64GB recommended)
- QA workload: 64GB+ RAM (128GB recommended with LLMs loaded)

### GPU (Optional)

- Embedding/reranking: 8GB+ VRAM
- GPU memory for LLM weights (add headroom for the KV cache):
  - GPT-OSS-120B: ~65GB (MXFP4)
  - GPT-OSS-20B: ~13GB (MXFP4)
  - Llama-3.1-8B judge: ~16GB (BF16)
- Run the three servers on separate devices (see `scripts/servers/launch_llm_server.sh`)

---

## Verification

After downloading, verify assets are in place:

```bash
# Check dataset
ls -lh frames-benchmark-dataset/frames_dataset.tsv

# Check models
ls -lh intfloat_e5-base-v2/e5-base-v2/
ls -lh colbert-ir_colbertv2.0/colbertv2.0/
ls -lh gpt-oss-20B/gpt-oss-20b/
ls -lh gpt-oss_model/gpt-oss-120b/

# Check Wikipedia documents (extracted by the download script)
ls doc_html/*.html | wc -l  # Should show 2515 files
```

---

## Troubleshooting

### Download interrupted or failed

The MLCommons downloader supports resume. Simply re-run the download command.

### Disk space issues

Download models individually as needed:
- For development: Start with e5-base-v2 + ColBERTv2.0 (~4GB)
- For QA workload: Add GPT-OSS models as needed
- Development only: smaller instruction-tuned models can stand in; submissions
  must use the models listed above

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
- Check the [Troubleshooting](README.md#10-troubleshooting) section
