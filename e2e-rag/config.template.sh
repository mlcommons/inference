# =============================================================================
# config.template.sh — copy to config.sh and edit per system. config.sh is
# gitignored.
#
# Resolution order per variable:
#   1. Already-exported env var  (e.g. QNA_DEVICE=cpu bash scripts/run_*.sh)
#   2. Value set in config.sh
#   3. Built-in default in the script
#
# Anything you don't set falls back to the script's built-in default.
# =============================================================================

# ── Shared by ingestion and QnA ───────────────────────────────────────────────
EMBEDDING_MODEL="intfloat_e5-base-v2/e5-base-v2"

# ── Ingestion pipeline (scripts/run_ingestion.sh) ─────────────────────────────
INGESTION_DEVICE="cpu"
INGESTION_EMBEDDING_DEVICE="${INGESTION_EMBEDDING_DEVICE:-${INGESTION_DEVICE}}"
INGESTION_NUM_EMBEDDING_DEVICES=4
INGESTION_CHUNK_SIZE=768
INGESTION_CHUNK_OVERLAP=32
INGESTION_DOC_DIR="doc_html"
INGESTION_PASSAGES_JSON="passages/doc_html_len768_ov32_word.json"
INGESTION_DB="vector_html_hnsw_len768_ov32_word"

# ── Inference pipeline (run_multi_shot.sh, run_single_shot.sh) ────────────────
QNA_DEVICE="cpu"
QNA_EMBEDDING_DEVICE="${QNA_EMBEDDING_DEVICE:-${QNA_DEVICE}}"
RERANKER_DEVICE="${RERANKER_DEVICE:-${QNA_DEVICE}}"
DATABASE="vector_html_hnsw_len768_ov32_word"
TOP_K_RETRIEVER=10
MAX_ITERATIONS=5
MAX_SUB_QUERIES=3
TEMPERATURE=1.0
MAX_RETRIES=5
N_QUERIES=5
QNA_MAX_WORKERS=1

# LLM endpoints (vLLM, OpenRouter, etc.)
GRADER_SERVICE_URL="http://127.0.0.1:8192/v1/chat/completions"
GRADER_MODEL="gpt-oss-20b-mxfp4"
QUERY_MODEL="gpt-oss-120b-mxfp4"

# Query and sufficiency (+ final answer) on the 120B server. Unset -> the
# Python defaults apply (query :8123; sufficiency follows query).
# QUERY_SERVICE_URL="http://127.0.0.1:8123/v1/chat/completions"
# SUFFICIENCY_SERVICE_URL="http://127.0.0.1:8123/v1/chat/completions"
# SUFFICIENCY_MODEL="gpt-oss-120b-mxfp4"

# Judge (used by evaluate.py at the end of the run scripts)
JUDGE_SERVICE_URL="http://127.0.0.1:8125/v1/chat/completions"
JUDGE_MODEL="meta-llama/Llama-3.1-8B-Instruct"

# ── Oracle evaluation (run_oracle.sh) ─────────────────────────────────────────
ORACLE_BATCH_SIZE=4
ORACLE_TIMEOUT=2400
# ORACLE_ENABLE_THINKING=1   # uncomment to pass --enable-thinking
ORACLE_DATASET="frames-benchmark-dataset/frames_dataset.tsv"
ORACLE_WIKI_DIR="wiki_articles"

# Override GPU index allocator (e.g. when auto-detect picks the wrong devices).
# Comma-separated 0-based indices; subset of available CUDA/XPU devices.
# Per-component:
# EMBEDDING_GPU_DEVICES="0,1"   # one entry per --num_embedding_devices worker
# RERANKER_GPU_DEVICES="2"

# Per-component NUMA / OMP placement (applied inside the per-process worker).
# Pins the worker's CPU set + memory to the given node; OMP threads default
# to the node's physical core count if not set.
# RERANKER_NUMA_NODE=0
# RERANKER_OMP_NUM_THREADS=21
# EMBEDDING_NUMA_NODES="0,0,1,1"     # one node per embedding worker
# EMBEDDING_OMP_NUM_THREADS=21       # cap per worker; default = even split

# ── Python-side env vars (CPU_*) ──────────────────────────────────────────────
# These only fire when a model lands on CPU (gated in apply_cpu_threading_env).
# Uncomment to override Python defaults:
# CPU_DISABLE_NUMA=1
# CPU_NUMA_NODE=0
# CPU_NUMA_CORES="43-85"
# CPU_OMP_NUM_THREADS=43
