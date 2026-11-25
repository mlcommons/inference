# E2E: RAG Benchmark

End-to-end retrieval-augmented generation benchmark for multi-hop question
answering on the [FRAMES](https://huggingface.co/datasets/google/frames-benchmark)
Wikipedia dataset. It has two LoadGen workloads:

- **RAG-DB** (`e2e-rag-db`, "ingestion"): build the FAISS HNSW vector database
  from the frozen Wikipedia corpus (one-time, measured).
- **RAG-QnA** (`e2e-rag-qna`): answer the 824 FRAMES questions with iterative
  multi-shot retrieval over that database (Offline scenario).

---

## Table of Contents

1. [Overview](#1-overview)
2. [Install](#2-install)
3. [Download models and data](#3-download-models-and-data)
4. [Configure](#4-configure)
5. [Start the LLM servers](#5-start-the-llm-servers)
6. [Build the vector database (RAG-DB)](#6-build-the-vector-database-rag-db)
7. [Run question answering (RAG-QnA)](#7-run-question-answering-rag-qna)
8. [Compliance: TEST09](#8-compliance-test09)
9. [What you may and may not change](#9-what-you-may-and-may-not-change)
10. [Troubleshooting](#10-troubleshooting)
11. [System requirements](#11-system-requirements)

---

## 1. Overview

For each question the pipeline repeats, for up to 5 iterations: generate
sub-queries, retrieve passages (e5-base-v2 + FAISS HNSW), rerank (ColBERTv2),
grade each new passage, and check whether the kept passages are sufficient;
then it generates the final answer.

| Role | Model | Default endpoint | Used for |
|---|---|---|---|
| query | gpt-oss-120b | `:8123` | query generation, sufficiency check, final answer |
| grader | gpt-oss-20b | `:8192` | document grading |
| judge | Llama-3.1-8B-Instruct | `:8125` | answer scoring (accuracy runs only) |
| embedding | e5-base-v2 | in-process | passage and query embeddings |
| reranker | ColBERTv2.0 | in-process | reranking |

**Performance vs accuracy.** Performance runs still call the LLM servers (their
latency is measured) but replay recorded responses from `PERF_CACHE_FILE`, so
every run follows the same trajectory. Accuracy runs use the live responses and
score the answers with the judge. The judge is not needed for performance runs.

## 2. Install

Start a container with your accelerator's runtime and mount this directory, for
example:

```bash
docker run -it --gpus all -v $(pwd):/workspace ubuntu:22.04      # NVIDIA
# AMD:   --device /dev/kfd --device /dev/dri instead of --gpus all
# Intel: --device /dev/dri instead of --gpus all
```

Inside the container (from `/workspace`):

```bash
apt-get update && apt-get install -y python3 python3-pip git curl wget
bash scripts/setup/setup.sh             # Python requirements
bash scripts/setup/prepare_loadgen.sh   # builds and installs mlperf_loadgen
```

Python 3.9 or newer is required. The LLM servers run on vLLM; install your
vendor's vLLM build (or use its container).

## 3. Download models and data

```bash
bash scripts/setup/download_dataset_and_models.sh
```

Run it from this directory. It downloads the frozen MLCommons copies (~83 GB)
and extracts the document corpus. For the gpt-oss models it fetches only the
files vLLM loads (config, tokenizer and the top-level safetensors shards) and
skips the `original/` and `metal/` weight copies the bucket also holds:

| Asset | Lands in | Default variable |
|---|---|---|
| FRAMES questions | `frames-benchmark-dataset/frames-benchmark/test.tsv`, linked as `frames-benchmark-dataset/frames_dataset.tsv` | (run scripts) |
| Wikipedia corpus | `doc_html/` (2515 HTML pages, from `docs.tar.gz`) | `INGESTION_DOC_DIR` |
| gpt-oss-120b (~65 GB) | `gpt-oss_model/gpt-oss-120b/` | `QUERY_MODEL` |
| gpt-oss-20b (~14 GB) | `gpt-oss-20B/gpt-oss-20b/` | `GRADER_MODEL` |
| e5-base-v2 | `intfloat_e5-base-v2/e5-base-v2/` | `EMBEDDING_MODEL` |
| ColBERTv2.0 | `colbert-ir_colbertv2.0/colbertv2.0/` | `RERANKER_MODEL` |

The judge (`meta-llama/Llama-3.1-8B-Instruct`) is not in the bucket; vLLM
downloads it from Hugging Face (gated: accept the license and set `HF_TOKEN`).
See [MLCOMMONS_ASSETS.md](MLCOMMONS_ASSETS.md) for individual downloads.

> **Use the frozen corpus.** Build your vector database from `doc_html/`. Do
> **not** re-scrape Wikipedia with `ingestion/download_docs.py`: the live pages have
> changed and will not reproduce the reference corpus or pass DB-manifest
> verification. `ingestion/download_docs.py` is only for regenerating the snapshot.

## 4. Configure

Every run script and `scripts/servers/launch_llm_server.sh` load `config.sh` (yours,
optional: copy `config.template.sh`) and then `config.default.sh`, which holds
the defaults below. An exported variable wins over both.

**Models** — each `<X>_MODEL` is a local path or a Hugging Face id (downloaded
on first use; warm the cache before a timed run):

| Variable | Default |
|---|---|
| `QUERY_MODEL` | `gpt-oss_model/gpt-oss-120b` |
| `SUFFICIENCY_MODEL` | `${QUERY_MODEL}` (same server) |
| `GRADER_MODEL` | `gpt-oss-20B/gpt-oss-20b` |
| `JUDGE_MODEL` | `meta-llama/Llama-3.1-8B-Instruct` |
| `EMBEDDING_MODEL` | `intfloat_e5-base-v2/e5-base-v2` |
| `RERANKER_MODEL` | `colbert-ir_colbertv2.0/colbertv2.0` |

**LLM endpoints** — `QUERY_SERVICE_URL`, `SUFFICIENCY_SERVICE_URL`
(`http://127.0.0.1:8123/v1/chat/completions`), `GRADER_SERVICE_URL` (`:8192`),
`JUDGE_SERVICE_URL` (`:8125`). The launch script reads the same variables.

**RAG-DB (ingestion)**

| Variable | Default | Meaning |
|---|---|---|
| `INGESTION_DOC_DIR` | `doc_html` | HTML corpus |
| `INGESTION_DB` | `vector_html_hnsw_len768_ov32_word` | database to build |
| `INGESTION_DEVICE` | `auto` | `auto`, `cuda`, `xpu`, `hpu` or `cpu` |
| `INGESTION_NUM_EMBEDDING_DEVICES` | `1` | devices used for embedding |
| `INGESTION_MAX_WORKERS` | `4` | parallel document workers |
| `INGESTION_CHUNK_SIZE` / `INGESTION_CHUNK_OVERLAP` | `768` / `32` | passage chunking |

**RAG-QnA**

| Variable | Default | Meaning |
|---|---|---|
| `DATABASE` | `vector_html_hnsw_len768_ov32_word.db` | database to query |
| `QNA_MAX_WORKERS` | `10` | parallel questions in the SUT |
| `MAX_ITERATIONS` | `5` | retrieval iterations (fixed by the rules) |
| `MAX_SUB_QUERIES` | `3` | sub-queries per iteration |
| `TOP_K_RETRIEVER` | `10` | passages retrieved per sub-query |
| `PERF_CACHE_FILE` | `logs_result.json` | recorded LLM responses; required for performance and TEST09 runs |

Sample counts come from `user.conf` (824 questions, 2515 documents); a
non-zero `TOTAL_SAMPLE_COUNT` overrides it with the first N questions for a
short test. Every run issues each sample once
(`performance_issue_unique = 1`).

All runs write under `WORKSPACE_DIR` (default `/workspace`).

## 5. Start the LLM servers

Start the three vLLM servers with `scripts/servers/launch_llm_server.sh`, one call per
server:

```bash
DEVICES=0,1,2,3 bash scripts/servers/launch_llm_server.sh query
DEVICES=4,5     bash scripts/servers/launch_llm_server.sh grader
DEVICES=6       bash scripts/servers/launch_llm_server.sh judge    # accuracy runs only
```

`DEVICES` selects the accelerators. The script detects the vendor and exports
the matching variable (override with `VENDOR=nvidia|amd|intel|cpu`):

| Vendor | Variable set from `DEVICES` |
|---|---|
| NVIDIA | `CUDA_VISIBLE_DEVICES` |
| AMD | `HIP_VISIBLE_DEVICES` |
| Intel | `ZE_AFFINITY_MASK` |
| CPU | none (`DEVICES` is ignored) |

Tensor parallelism defaults to the number of `DEVICES`; extra arguments go
straight to `vllm serve` (e.g. `... query -tp 2 --gpu-memory-utilization 0.9`).
Each server logs to `logs/vllm_<role>.log` and the script returns once it
answers on its port. The model and port come from `<ROLE>_MODEL` and
`<ROLE>_SERVICE_URL`, the same variables the run scripts read.

## 6. Build the vector database (RAG-DB)

```bash
bash scripts/run_ingestion_perf.sh            # performance
bash scripts/run_ingestion_accuracy.sh   # accuracy: build, then verify
```

Both build `${INGESTION_DB}.db` and `${INGESTION_DB}_data/`. The accuracy run
then verifies the database: the MD5 reported by the run matches the saved
file, and the reference manifest (`MANIFEST`, default
`scripts/db/reference_manifest.json.gz`) matches in passage count, embedding
dimension and top-K retrieval overlap. Logs go to
`${WORKSPACE_DIR}/run_output_e2e-rag-db/{performance,accuracy}`.

## 7. Run question answering (RAG-QnA)

Start the query and grader servers first (and the judge for accuracy).

```bash
bash scripts/run_qna_perf.sh       # performance: replays PERF_CACHE_FILE
bash scripts/run_qna_accuracy.sh   # accuracy: live LLM responses + judge
```

LoadGen logs go to `${WORKSPACE_DIR}/run_output_e2e-rag-qna/{performance,accuracy}`
and results to `${WORKSPACE_DIR}/output_e2e-rag-qna/{performance,accuracy}`.
Accuracy runs score `results.json` with the judge (`evaluation/accuracy_eval_qna.py`) and
write `accuracy.txt` to the log directory. Only the Offline scenario is
configured (`user.conf`).

## 8. Compliance: TEST09

```bash
bash scripts/run_compliance_test09.sh
```

TEST09 checks that the mean output length of the final answer in a
performance run is within ±10% of the reference. It needs the TEST09 config
from the inference repo (`compliance/TEST09/e2e-rag-qna/`, or set
`COMPLIANCE_DIR`) and replays `PERF_CACHE_FILE` like the performance run; it
stops if the file is missing.

## 9. What you may and may not change

Fixed by the [rules](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc#e2e-rag):

- Retrieval iterations: `MAX_ITERATIONS=5`.
- RAG-DB index: FAISS HNSW with `M=32`, `efConstruction=200`, `efSearch=100`.
- Sampling for the GPT-OSS models: `temperature=1.0`, `top_p=1.0`.
- The models listed above, the frozen corpus, and the QnA run using the
  database produced by the measured RAG-DB run.
- No caching across questions (KV-cache within a question is allowed).

Reference defaults (keep them for comparability): `TOP_K_RETRIEVER=10`,
chunking `768`/`32`, `MAX_SUB_QUERIES=3`. Devices, worker counts and vLLM
serving options are yours to tune.

## 10. Troubleshooting

- **Model not found (HTTP 404) from an LLM server**: the client sends
  `<ROLE>_MODEL`; start the server with the same value (the launch script does).
- **`ERROR: PERF_CACHE_FILE not found`** from a performance or TEST09 run:
  point `PERF_CACHE_FILE` at the recorded responses (default
  `logs_result.json`).
- **HF id fails to load offline**: pre-fetch it (`hf download <id>`) or point
  `<X>_MODEL` at a local path.

## 11. System requirements

- **Disk**: ~120 GB: models ~83 GB (only the files vLLM loads), judge ~16 GB
  from Hugging Face, corpus and database a few GB.
- **Memory**: 32 GB+ RAM recommended for the SUT, plus what the LLM servers need.
- **Accelerators**: NVIDIA, AMD or Intel GPUs for the LLM servers; embedding
  and reranking run on any `--device` (`cuda`, `rocm`, `xpu`, `hpu`, `cpu`).
- **Python**: 3.9+. **OS**: Ubuntu 20.04+ (or a compatible container).

---

## License

See LICENSE file in repository root.
