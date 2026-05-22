# =============================================================================
# config.template.sh — copy to config.sh and edit per system. config.sh is
# gitignored.
#
# Resolution order per variable:
#   1. Already-exported env var  (e.g. DEVICE=cpu bash scripts/run_*.sh)
#   2. Value set in config.sh
#   3. Built-in default in the script
#
# Anything you don't set falls back to the script's built-in default.
# =============================================================================

# ── Ingestion pipeline (scripts/run_ingestion.sh) ─────────────────────────────
INGESTION_DEVICE="cpu"
INGESTION_EMBEDDING_DEVICE="${INGESTION_EMBEDDING_DEVICE:-${INGESTION_DEVICE}}"
INGESTION_NUM_EMBEDDING_DEVICES=4
INGESTION_CHUNK_LEN=768
INGESTION_CHUNK_OVERLAP=32
INGESTION_RETRIEVER_MODEL="/data/model/e5-base-v2"
INGESTION_DOC_DIR="doc_html"
INGESTION_PASSAGES_JSON="passages/doc_html_len768_ov32_word.json"
INGESTION_DB="vector_html_hnsw_len768_ov32_word"

# ── Inference pipeline (run_multi_shot.sh, run_single_shot.sh) ────────────────
INFERENCE_DEVICE="cpu"
INFERENCE_EMBEDDING_DEVICE="${INFERENCE_EMBEDDING_DEVICE:-${INFERENCE_DEVICE}}"
INFERENCE_RERANKER_DEVICE="${INFERENCE_RERANKER_DEVICE:-${INFERENCE_DEVICE}}"
INFERENCE_DB="vector_html_hnsw_len768_ov32_word"
INFERENCE_RETRIEVER_MODEL="/data/model/e5-base-v2"
INFERENCE_TOP_K_RETRIEVER=15
INFERENCE_MAX_ITERATIONS=5
INFERENCE_MAX_SUB_QUERIES=3
INFERENCE_TEMPERATURE=1.0
INFERENCE_MAX_RETRIES=5
INFERENCE_N_QUERIES=5
INFERENCE_NUM_WORKERS=1

# LLM endpoints (vLLM, OpenRouter, etc.)
INFERENCE_LLM_URL="http://127.0.0.1:8123/v1/chat/completions"
INFERENCE_MODEL="/model/gpt-oss-20b-mxfp4"
INFERENCE_QUERY_MODEL="/model/gpt-oss-120b-mxfp4"

# Judge (used by evaluate.py at the end of the run scripts)
INFERENCE_JUDGE_URL="https://openrouter.ai/api/v1/chat/completions"
INFERENCE_JUDGE_MODEL="openai/gpt-oss-20b"

# ── Oracle evaluation (run_oracle.sh) ─────────────────────────────────────────
INFERENCE_ORACLE_BATCH_SIZE=4
INFERENCE_ORACLE_TIMEOUT=2400
# INFERENCE_ORACLE_ENABLE_THINKING=1   # uncomment to pass --enable-thinking
INFERENCE_ORACLE_DATASET="data/frames_dataset.tsv"
INFERENCE_ORACLE_WIKI_DIR="wiki_articles"

# Override GPU index allocator (e.g. when auto-detect picks the wrong devices).
# Comma-separated 0-based indices; subset of available CUDA/XPU devices.
# Per-component:
# INFERENCE_EMBEDDING_GPU_DEVICES="0,1"   # one entry per --num_embedding_devices worker
# INFERENCE_RERANKER_GPU_DEVICES="2"

# ── Python-side env vars (CPU_*) ──────────────────────────────────────────────
# These only fire when a model lands on CPU (gated in apply_cpu_threading_env).
# Uncomment to override Python defaults:
# CPU_DISABLE_NUMA=1
# CPU_NUMA_NODE=0
# CPU_NUMA_CORES="43-85"
# CPU_OMP_NUM_THREADS=43
