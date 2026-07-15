# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a RAG (Retrieval-Augmented Generation) benchmark system for evaluating multi-hop question answering using Wikipedia documents from the FRAMES dataset. The system supports multiple retrieval methods (BM25, vector search), reranking, and multi-shot iterative retrieval with query decomposition.

## Architecture

The codebase follows a modular pipeline architecture:

1. **Document Ingestion** → **Passage Chunking** → **Vector/BM25 Indexing**
2. **Query** → **Retrieval** → **Optional Reranking** → **LLM Answer Generation** → **Evaluation**

### Core Components

- **`retrieve/` module**: Defines abstract `RagDB` base class with two implementations:
  - `VectorDB`: Dense vector search using FAISS (flat, HNSW, or IVF indexing)
  - `BM25DB`: Sparse lexical search using bm25s library
  - Both support reranking with ColBERTv2 or similar models
  
- **Retrieval Scripts**:
  - `single_shot_retrieval.py`: Single-step retrieval and evaluation
  - `multi_shot_retrieval.py`: Multi-hop retrieval with LLM-based query decomposition (iterative retrieval with query rewriting)
  - `oracle_single_shot.py`: Upper-bound evaluation using ground truth documents

- **Parameter Management**: `params.py` centralizes all CLI parameters with Optuna optimization metadata

- **Utilities**:
  - `download_docs.py`: Downloads Wikipedia pages from FRAMES dataset URLs
  - `read_docs.py`: Extracts text and chunks documents into passages (uses `text_splitter.py`)
  - `evaluate.py`: LLM judge-based evaluation of generated answers
  - `utils.py`: Common helpers (device config, LLM setup, seeding)

## Common Development Workflows

### Initial Setup

```bash
# Install dependencies (requires Ubuntu-based environment)
./setup.sh

# Download Wikipedia documents from FRAMES dataset
python3 download_docs.py --output_dir doc_html --format html --processes 30

# Build vector database (run once)
bash scripts/run_ingestion.sh
```

The setup script will:
1. Extract passages from HTML documents → `passages/doc_html_len2048_ov32_word.json`
2. Build FAISS HNSW index → `vector_html_hnsw_len2048_ov32_word.db`

### Running Experiments

**Single-shot retrieval:**
```bash
# Run evaluation on existing database
python3 single_shot_retrieval.py \
    --db vector_html_hnsw_len2048_ov32_word.db \
    --retrieval_method vector \
    --eval 100 \
    --generate-answer

# Compare BM25 vs Vector
python3 single_shot_retrieval.py --db bm25.db --retrieval_method bm25 --eval
```

**Multi-shot retrieval with query decomposition:**
```bash
# Run multi-shot experiment (requires LLM server on port 8123)
bash scripts/run_multi_shot.sh 50  # Evaluate 50 queries
bash scripts/run_multi_shot.sh all # Full evaluation
```

**Oracle upper bound (using ground truth docs):**
```bash
python3 oracle_single_shot.py \
    --dataset data/frames_dataset.tsv \
    --wiki-articles-dir wiki_articles \
    --batch-size 16
```

### LLM Server Setup

The system expects a vLLM-compatible OpenAI API server:

```bash
# Start vLLM server (example from scripts/start_vllm_server.sh)
python3 -m vllm.entrypoints.openai.api_server \
    --model /model/gpt-oss-20b-mxfp4 \
    --dtype bfloat16 \
    --host 0.0.0.0 \
    --port 8123 \
    --gpu-memory-util=0.95 \
    --enable-prefix-caching \
    --max-model-len=131072
```

Default service URL: `http://127.0.0.1:8123/v1/chat/completions`

### Evaluation

Evaluation uses an LLM judge to score answers:

```bash
# Score results from any experiment
python3 evaluate.py result_single_shot.json
python3 evaluate.py result_multi_shot.json
python3 evaluate.py oracle_checkpoint.pkl  # For oracle results
```

## Key Parameters (via params.py)

All parameters are centralized in `params.py` with CLI definitions. Key categories:

**Retrieval Method:**
- `--retrieval_method {bm25,vector}`: Choose retrieval backend
- `--vector_index_method {flat,hnsw,ivf}`: FAISS index type (default: hnsw)
- `--bm25_k1`, `--bm25_b`, `--bm25_stemmer`: BM25 tuning parameters

**Retrieval Strategy:**
- `--retrieval_strategy {fixed_k,top_p,relative}`: How many docs to retrieve
- `--top_k_retriever N`: Number of docs to retrieve (default: 10)
- `--top_k_reranking N`: Number of docs after reranking (default: 10)

**Device & Performance:**
- `--device {auto,xpu,cuda,hpu,cpu}`: Hardware accelerator
- `--num_embedding_devices N`: Parallel embedding generation across devices
- `--benchmark`: Enable performance monitoring

**Multi-shot specific (multi_shot_retrieval.py):**
- `--max-iterations N`: Max retrieval rounds (default: 5)
- `--max-sub-queries N`: Sub-queries per iteration (default: 3)

**Oracle specific (oracle_single_shot.py):**
- `--batch-size N`: Batch requests to LLM (default: 1)
- `--timeout N`: Request timeout in seconds (default: 2400)
- `--enable-thinking`: Use chain-of-thought reasoning

## Hardware Support

The system supports multiple accelerators via `--device`:
- **XPU** (Intel GPUs): Primary development target
- **CUDA** (NVIDIA GPUs)
- **HPU** (Habana Gaudi): Embeddings/reranking fall back to CPU due to compatibility
- **CPU**: Fallback option

Device selection is abstracted in `utils.py:get_device_config()` and `ragdb.py:_determine_device()`.

## Database Persistence

- Vector databases: `.db` file (FAISS index) + `_data/` directory (docstore, metadata)
- BM25 databases: Pickled retriever object in `.db` file
- Embeddings cache: `.emb.pkl` files (use `--load-embeddings` to reuse)
- Checkpoints: `oracle_checkpoint.pkl` for oracle runs (resumable via pandas DataFrame)

## Multi-shot Retrieval Logic

The multi-shot system (multi_shot_retrieval.py) implements iterative retrieval:
1. LLM evaluates retrieved docs and decides if sufficient to answer
2. If insufficient, generates up to k focused sub-queries
3. Each sub-query retrieves additional docs
4. Process repeats for max N iterations
5. Final docs are reranked and passed to answer generator

The query rewriter prompt includes failure analysis to escalate search strategies when stuck (e.g., switching from specific queries to broader entity searches).

## Evaluation Methodology

- **Retrieval Accuracy**: Checks if correct Wikipedia URLs are in top-K results
- **Answer Quality**: LLM judge scores generated answers against ground truth
- **Difficulty Filtering**: `--difficulty N` filters queries by number of required source documents

Results are saved to JSON files with schema:
```json
{
  "query": "...",
  "retrieved_urls": [...],
  "correct_urls": [...],
  "llm_answer": "...",
  "ground_truth_answer": "..."
}
```

## Testing & Debugging

- Use `--eval N` to test on first N queries (faster iteration)
- Use `--no-rerank` to compare retrieval methods fairly
- Use `--no-save` to skip writing database during optimization
- Use `--benchmark` to track component performance
- Check logs: Scripts redirect output to `log_*.txt` files

## Important Notes

- **LLM timeout**: Oracle and multi-shot runs may need `--timeout` adjustment for reasoning models
- **Checkpointing**: Oracle script saves progress after each batch and can resume from checkpoint
- **Determinism**: Use `--seed` for reproducible results (affects sampling, not LLM generation)

---

## Experiments and Results

### Accuracy Benchmarks

| Type | Queries | Precision@N | Recall@N | F1@N | LLM Judge Accuracy |
|---|---|---|---|---|---|
| Oracle | 824 | 100% | 100% | 100% | 68% |
| Single-shot retrieval | 827 | 39% | 70% | 42% | 20% |
| Multi-shot baseline | 50 | 12% | 36% | 16% | 20% |
| Multi-shot + fixes below | 50 | 69% | 64% | 61% | 42% |
| Multi-shot + fixes below | 400 | 73% | 67% | 66% | 34% |
| **Multi-shot + fixes below** | **824** | **72%** | **67%** | **66%** | **36%** |

The retrieval recall is the primary bottleneck: theoretical max accuracy ≈ Recall × Oracle_accuracy.

---

### Fix 1: Split Monolithic Query Rewriter (HIGH IMPACT)

**Root Cause:** The original `query_rewriter()` was a single LLM call with 3 simultaneous tasks:
1. Evaluate relevance of new documents
2. Decide if accumulated docs are sufficient to answer
3. Generate new search queries

This cognitive overload caused the LLM to mark **all documents as irrelevant** (relevance: [0,0,0,...]), leading to 0 kept docs and 60% "Unknown" answers.

**Fix:** Split into two focused LLM calls:
- `evaluate_document_relevance()` — binary relevance classification only (temp=0.0, short prompt)
- `generate_search_queries()` — query generation or final answer (temp=0.1, full context)

**Additional fixes bundled with Fix 1:**
- Added best-effort final answer after max iterations (instead of always returning "Unknown")
- Added "return Unknown if insufficient" guard in prompt to reduce hallucination
- Added fallback to original query when no sub-queries are generated
- Guarded `sufficient=True` to require non-empty `kept_docs`
- Added `reasoning_content` fallback + `max_tokens=10240` for thinking-model compatibility
- Fixed `UnboundLocalError` from `import re` inside `try` blocks (4 locations)

**Result:** Accuracy 20% → 30%, Recall 36.8% → 51.8% (n=50)

---

### Fix 2: Context Length Reduction (REVERTED — made things worse)

**Motivation:** Fix 1 still produced empty LLM responses due to long prompts when many docs accumulated (11+ docs × 1200 chars ≈ 20KB+ prompts hitting token limits).

**Change:** Limit kept-doc context shown to LLM — query generation: 10 most recent docs; relevance check: 5 most recent docs.

**Result:** Fewer empty responses, but accuracy dropped 30% → 28%, Recall 51.8% → 46.4%.

**Why it failed:** Multi-hop reasoning needs to connect facts across all retrieved documents, not just the most recent ones. Truncating context broke cross-document reasoning chains.

**Decision:** Reverted Fix 2. The right fix is instead: increase `max_tokens` for the judge/LLM calls so they don't hit length limits.

---

### Chunking Strategy Experiments

**Baseline:** 2048-char chunks, 32-char overlap (1.5% overlap) — too large, dilutes embeddings.

**Hypothesis:** Smaller chunks produce more focused embeddings → better retrieval precision for multi-hop facts.

#### Results Across Chunk Sizes (multi-shot, n=50)

| Chunk Size | Overlap | Recall@N | Precision@N | LLM Accuracy |
|---|---|---|---|---|
| 2048 chars | 32 (1.5%) | 51.8% | 62.5% | 30% |
| 512 chars | 100 (20%) | — | — | Tested, no improvement |
| **768 chars** | **32 (4%)** | **67.7%** | **73.0%** | **34–37%** |

**Winner: 768-char chunks with 32-char overlap** — significant retrieval improvement over 2048.

**Why 768 works better than 512:**
- 512-char chunks split related facts across too many chunks; retrieval becomes noisy
- 768-char chunks fit 2–3 complete sentences; good balance of focus vs. context
- More manageable passage count than 512

**Overlap finding:** The 1.5% overlap (32/2048) in the baseline was too low. With 768-char chunks, even 32-char overlap (4%) provides measurably better boundary coverage than 32/2048 did.

**Strategies considered but not implemented:**
- **Hierarchical chunking** (512 retrieval / 2048 context): Promising but complex; worth trying if further gains needed
- **Semantic sentence grouping**: More implementation complexity for marginal benefit over fixed-length with word boundary
- **Token-based chunking**: Aligns better with embedding model limits (e5-base-v2 = 512 tokens); worth trying

**Key insight for future experiments:** The chunk size primarily affects retrieval recall. Every ~10% recall improvement translates to ~7% accuracy gain (based on Recall × Oracle_accuracy formula).

---

### Iteration Distribution (full 824-query run, max_iterations=5)

| Iterations Used | Questions | % |
|---|---|---|
| 2 | 297 | 36.0% |
| 3 | 113 | 13.7% |
| 4 | 37 | 4.5% |
| 5 | 377 | 45.8% |

45.8% of queries hit the max-iterations limit — suggesting accuracy gains are available by increasing `--max-iterations` to 7 or 10.

---

### Future Experiment Candidates

In priority order based on findings above:

1. **Increase `--max-iterations`** (7 or 10) — 45.8% of queries are cut off at 5 iterations
2. **Hierarchical chunking** (retrieve 512-char children, answer with 2048-char parents)
3. **Token-based chunking** at 256 tokens to align with e5-base-v2 limits
4. **Hybrid BM25 + vector retrieval** — lexical search catches exact name matches that dense search misses
5. **Larger top_k_retriever** (15 → 25) given high iteration cutoff rate
6. **Better judge model** — current judge (same gpt-oss-20b) hits token limits on complex questions; a stronger judge would give more accurate accuracy measurements
