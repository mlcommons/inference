# E2E: RAG benchmark

End-to-end retrieval-augmented generation benchmark for multi-hop QA on the
[FRAMES](https://huggingface.co/datasets/google/frames-benchmark) Wikipedia
dataset. Supports BM25 and dense vector retrieval, optional ColBERTv2 reranking,
and iterative multi-shot retrieval with LLM-driven query decomposition.

---

## Benchmark flow

We start with a corpus of documents and a set of user queries.

1. **Corpus preparation** (one-time): download Wikipedia pages → chunk into
   passages → build a vector DB.
2. **Inference** (per query): retrieve → rerank → generate answer.
3. **Evaluation**: LLM judge scores the answer against ground truth.

For multi-hop questions, step 2 iterates: the LLM evaluates retrieved docs,
decides if they're sufficient, and otherwise generates fresh sub-queries to
search again. See [Architecture: multi-shot retrieval](#architecture-multi-shot-retrieval).

---

## Quick start

For a system with the vector DB already built (and `data/frames_dataset.tsv`
present):

```bash
cp config.template.sh config.sh
$EDITOR config.sh                              # set device, paths, NUMA, model

export OPENROUTER_API_KEY="sk-or-v1-..."       # only needed for LLM judge

bash scripts/run_multi_shot.sh 50 10           # 50 queries, 10 parallel workers
```

Output lands in `output_multi_shot_n50_w10_<timestamp>/` with:
- `result_multi_shot_n50.json` — retrieval + answer per query
- `score_multi_shot_n50.txt` — LLM judge accuracy summary
- `run.log` — full stdout

---

## Setup from scratch

### Environment

Recommended: a Ubuntu PyTorch sandbox via enroot or Docker.

- [Ubuntu base image](https://hub.docker.com/_/ubuntu) (works, not optimal)
- [NVIDIA PyTorch image](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags) (CUDA)
- [AMD ROCm/PyTorch image](https://hub.docker.com/r/rocm/pytorch/tags) (ROCm)
- Intel XPU: build PyTorch from Intel's wheels manually inside the sandbox.

Enroot example:

```bash
mkdir -p containers/
enroot import -o containers/my_image.sqsh dockerd://pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime
enroot create --name my_sandbox containers/my_image.sqsh
enroot start --root --rw \
    --mount /actual/path:/mounted/path \
    --mount $(pwd):/work my_sandbox

cd /work && ./setup.sh                         # pip install + apt deps
```

### Corpus download

Pulls all Wikipedia URLs referenced in the FRAMES dataset and saves them as
PDFs or HTML, plus a `url_mapping.json` linking the original URL to each file
(used later for retrieval grading).

```bash
python3 download_docs.py --output_dir doc_html --format html --processes 30
```

| Option | Default | Notes |
|---|---|---|
| `--tsv_path` | (downloads FRAMES) | Local TSV instead |
| `--max_urls` | all | Cap for testing |
| `--output_dir` | `doc_pdf` / `doc_html` | Where to save documents |
| `--format` | `pdf` | `pdf` or `html` |
| `--processes` | 10 | Parallelism |

HTML is the default in our scripts since it scores ~1% better than PDF in
extraction quality. Use `download_docs.py --help` for the full list.

### Passage chunking (handled by `run_ingestion.sh`)

Embedding models (especially the ColBERTv2 reranker) cap input at ~512 tokens,
so we split each document into overlapping fixed-length passages. Default:
768-char passages with 32-char overlap and word-boundary splitting.

The chunker output is a JSON list:

```json
{
    "index": 0,
    "base_filename": "name_of_file",
    "original_url": "https://...",
    "passage": "Long passage from (part of) document"
}
```

`run_ingestion.sh` invokes `read_docs.py` for chunking and then
`single_shot_retrieval.py --ingest` to build the vector DB. To change chunk
size or paths, edit `INGESTION_*` variables in `config.sh`.

### Vector DB build

```bash
bash scripts/run_ingestion.sh
```

Outputs:
- `${INGESTION_PASSAGES_JSON}` — passage JSON
- `${INGESTION_DB}.db` — FAISS index file
- `${INGESTION_DB}_data/` — docstore + metadata

---

## Configuration (`config.sh`)

System-specific knobs live in a shell-sourced `config.sh` at the repo root.
Copy `config.template.sh` and edit. The file is gitignored.

```bash
cp config.template.sh config.sh
$EDITOR config.sh
```

### Variable namespacing

| Prefix | Used by | Examples |
|---|---|---|
| `INGESTION_*` | `run_ingestion.sh` | chunk size, embedding device for build, paths |
| `INFERENCE_*` | `run_multi_shot.sh`, `run_single_shot.sh` | device, retriever, top-k, LLM endpoints, judge |
| `INFERENCE_ORACLE_*` | `run_oracle.sh` | batch size, timeout, dataset, wiki dir |
| `CPU_*` | Python (interpreted by `utils.apply_cpu_threading_env`) | NUMA node, OMP thread count |

### Resolution order

For each variable:

1. Already-exported env var (one-off override). E.g. `INFERENCE_DEVICE=cpu bash scripts/run_multi_shot.sh 50`.
2. Value set in `config.sh`.
3. Built-in default in the script.

So setting a value in `config.sh` is per-system; setting it on the command line
is per-run.

### Pre-built reference configs

- `config.template.sh` — annotated template, copy this.
- `config.AMD.CPU.sh` — 2×96C Turin, embedding+reranker on CPU.
- `config.AMD.GPU.sh` — 2×96C Turin + 8 AMD GPUs, embedding+reranker on GPU.

For the full per-knob reasoning see `claude/design/config_layout.md`.

---

## Pipelines

All entry-point scripts source `config.sh` and pass appropriate flags to the
underlying Python. Direct Python invocation is still supported for ad-hoc work.

| Script | Purpose | Underlying script |
|---|---|---|
| `scripts/run_ingestion.sh` | Build vector DB from a doc directory | `read_docs.py` + `single_shot_retrieval.py --ingest` |
| `scripts/run_single_shot.sh [N]` | Single-shot retrieval + judge | `single_shot_retrieval.py` + `evaluate.py` |
| `scripts/run_multi_shot.sh [N] [WORKERS]` | Iterative multi-shot retrieval + judge | `multi_shot_retrieval.py` + `evaluate.py` |
| `scripts/run_oracle.sh [N]` | Upper-bound: ground-truth docs → LLM | `oracle_single_shot.py` + `evaluate.py` |
| `scripts/write_db_manifest.sh` | Generate cross-system DB fingerprint | `db_manifest.py write` |
| `scripts/verify_db_manifest.sh MANIFEST` | Verify local DB matches a reference manifest | `db_manifest.py verify` |

Positional args:
- `N` = number of queries (`all` = full 824-query dataset).
- `WORKERS` = parallel query threads (default 1).

Both override the config defaults (`INFERENCE_N_QUERIES`, `INFERENCE_NUM_WORKERS`).

### Direct Python invocation (ad-hoc)

```bash
# Single query, no eval
python3 single_shot_retrieval.py \
    --db vector_html_hnsw_len768_ov32_word.db \
    --retrieval_method vector \
    --query "Who won the French Open Mens Singles tournament the year that New York City FC won their first MLS Cup title?"

# BM25 instead of vector
python3 single_shot_retrieval.py --db bm25.db --retrieval_method bm25 --eval 50
```

Useful flags (for both single-shot and multi-shot):

| Flag | Notes |
|---|---|
| `--device {auto,cuda,rocm,xpu,hpu,cpu}` | Default device for embedding + reranker |
| `--embedding-device <...>` | Override just the embedder. Defaults to `--device`. |
| `--reranker-device <...>` | Override just the reranker. |
| `--retrieval_method {bm25,vector}` | Backend |
| `--vector_index_method {flat,hnsw,ivf}` | Vector index type (default: hnsw) |
| `--eval [N]` | Run on N queries from `--dataset` |
| `--top_k_retriever N` | Top-K from retriever |
| `--top_k_reranking N` | Top-K after reranker |
| `--no-rerank` | Skip reranker entirely |
| `--benchmark` | Performance monitoring |

---

## Cross-vendor support

This codebase originally targeted Intel CPU + Intel XPU. After the
cross-vendor refactor it also runs on:

- AMD CPU + AMD GPU (ROCm)
- AMD CPU + NVIDIA GPU
- Intel CPU + NVIDIA GPU

Key surfaces:

- **Device detection**: `utils.detect_device()` returns the best available.
  PyTorch ROCm exposes AMD GPUs through `torch.cuda.*`, so AMD GPU device
  strings are `"cuda:N"`. `--device rocm` is a cosmetic alias for `cuda` with
  ROCm-specific logging.
- **Per-model placement**: `--embedding-device` and `--reranker-device` let
  embedding and reranker live on different devices.
- **GPU index allocation**: `DeviceAllocator` picks empty GPUs (≥95% free VRAM
  via `mem_get_info`) and prevents within-process collisions. Override with
  `INFERENCE_EMBEDDING_GPU_DEVICES` / `INFERENCE_RERANKER_GPU_DEVICES`.
- **CPU NUMA pinning** (when a model is on CPU): each worker process pins
  itself to a NUMA node — `os.sched_setaffinity` for CPU, `set_mempolicy` for
  memory. Configure via `INFERENCE_RERANKER_NUMA_NODE`,
  `INFERENCE_EMBEDDING_NUMA_NODES`, `CPU_NUMA_NODE`.
- **Per-process reranker**: the reranker model loads in its own
  `multiprocessing.Process` so it has independent `OMP_NUM_THREADS` and NUMA
  placement, isolated from the main process and the embedder.

The full reasoning is in `claude/design/cross_vendor_plan.md`.

### Cross-system DB sanity check

To confirm a vector DB built on system A matches one built independently on
system B (same passages, same model, same parameters):

```bash
# System A:
bash scripts/write_db_manifest.sh manifest.json.gz
# Ship manifest.json.gz to system B.

# System B (after building its own DB):
bash scripts/verify_db_manifest.sh manifest.json.gz
# exits 0 on match; prints summary diff on mismatch.
```

The manifest contains a corpus sha256, sample embeddings at deterministic
indices, and probe-query top-K results. Tolerances:
- Corpus + counts: exact match.
- Sample embeddings: cosine ≥ 0.9999 (configurable via `--cosine-threshold`).
- Probe top-K: exact rank match for top-3 (configurable via `--top-k-depth`).

---

## Vector DB & indexing

### Retrieval methods

- **BM25**: sparse lexical retrieval (traditional keyword search). Uses [bm25s](https://github.com/xhluca/bm25s).
- **Vector**: dense semantic retrieval via [intfloat/e5-base-v2](https://huggingface.co/intfloat/e5-base-v2) embeddings + FAISS.

### Vector index types

| Type | Time | Best for |
|---|---|---|
| `flat` (L2) | O(n) | <10K passages |
| `hnsw` | O(log n) | balanced, default |
| `ivf` | O(√n) | >1M passages |

---

## Architecture: multi-shot retrieval

For multi-hop questions like:

> Who won the French Open Mens Singles tournament the year that New York City
> FC won their first MLS Cup title?

A single retrieval rarely surfaces both facts. Multi-shot iterates:

1. User query → query decomposer (LLM) generates k focused sub-queries.
2. For each sub-query: embed → vector search → retrieve top docs.
3. LLM evaluates relevance of new docs.
4. LLM checks if accumulated docs are sufficient.
5. If insufficient, generate new sub-queries and repeat (up to N iterations).
6. Final docs are reranked, then passed to answer generator.

Three LLM components, separable by URL + model:

- **Grader** (`INFERENCE_GRADER_URL`): binary relevance per doc. Cheap; small model.
- **Sufficiency checker** (`INFERENCE_SUFFICIENCY_URL`): "is this enough to answer?" Bigger model.
- **Query generator** (`INFERENCE_QUERY_URL`): generates new sub-queries when insufficient.

Each defaults to `INFERENCE_LLM_URL` if not set explicitly. Lets you put the
small grader on a dedicated vLLM and the larger sufficiency/query model on a
beefier endpoint.

The judge model used by `evaluate.py` is independent
(`INFERENCE_JUDGE_URL` / `INFERENCE_JUDGE_MODEL`).

### Query rewriter prompt (sketch)

```
You are an expert at generating search queries to help answer complex questions
using a collection of Wikipedia articles.

Given:
- The user's original question.
- Relevant facts or documents already gathered so far (if any).

Your task:
Generate [k] concise, focused search queries that could be used to find specific
information from Wikipedia to help answer the question.
- Make each query target a different aspect of the problem or missing information.
- Avoid duplicating information already in the context.
- Do not reference source filenames, document titles, or include any special characters.
- Think step by step before writing each query.
- List the missing pieces of information, then write k queries that could best retrieve them.

[User Question:]
{user_question}

[Known Facts / Retrieved Documents:]
{summarized_partial_context}
```

---

## Evaluation

`evaluate.py` runs an LLM judge against the result JSON, scoring each model
answer against the gold answer. Configurable via:

- `INFERENCE_JUDGE_URL` / `INFERENCE_JUDGE_MODEL` (config.sh)
- `--judge-url` / `--judge-model` (CLI)
- `--batch-size N` parallelism

The default judge is `openai/gpt-oss-20b` via OpenRouter (set
`OPENROUTER_API_KEY`). Bigger judges produce more accurate scores; the 20B
judge has known length-limit issues on long multi-hop questions.

Results from any pipeline can be re-scored later:

```bash
python3 evaluate.py output_multi_shot_n50_w10_<timestamp>/result_multi_shot_n50.json
```

---

## Outputs

Each pipeline writes to a timestamped `output_<pipeline>_<tag>_<workers?>_<timestamp>/`
directory containing:

```
result_<pipeline>_<tag>.json    # retrieval results + generated answers
score_<pipeline>_<tag>.txt      # LLM judge per-query scores + summary
run.log                         # full stdout
```

Result schema:

```json
{
    "query": "...",
    "retrieved_urls": [...],
    "correct_urls": [...],
    "llm_answer": "...",
    "ground_truth_answer": "..."
}
```

---
