# =============================================================================
# config.template.sh: copy to config.sh (gitignored) and uncomment or add the
# settings you want to change. Defaults and the full list are in
# config.default.sh; keep the ${VAR:-value} guard so an exported env var still
# wins. Values are exported to Python automatically.
# =============================================================================

# ── Models stored outside this directory ─────────────────────────────────────
# QUERY_MODEL="${QUERY_MODEL:-/data/models/gpt-oss-120b}"
# GRADER_MODEL="${GRADER_MODEL:-/data/models/gpt-oss-20b}"
# JUDGE_MODEL="${JUDGE_MODEL:-/data/models/Llama-3.1-8B-Instruct}"
# EMBEDDING_MODEL="${EMBEDDING_MODEL:-/data/models/e5-base-v2}"
# RERANKER_MODEL="${RERANKER_MODEL:-/data/models/colbertv2.0}"

# ── Devices and parallelism ──────────────────────────────────────────────────
# INGESTION_DEVICE="${INGESTION_DEVICE:-xpu}"
# INGESTION_NUM_EMBEDDING_DEVICES="${INGESTION_NUM_EMBEDDING_DEVICES:-4}"
# QNA_DEVICE="${QNA_DEVICE:-xpu}"
# QNA_MAX_WORKERS="${QNA_MAX_WORKERS:-32}"

# ── LLM servers on another host ──────────────────────────────────────────────
# QUERY_SERVICE_URL="${QUERY_SERVICE_URL:-http://10.0.0.2:8123/v1/chat/completions}"

# ── Output location ──────────────────────────────────────────────────────────
# WORKSPACE_DIR="${WORKSPACE_DIR:-${HOME}/e2e-rag-runs}"

# ── Component placement (read by Python) ─────────────────────────────────────
# EMBEDDING_GPU_DEVICES="${EMBEDDING_GPU_DEVICES:-0,1}"
# RERANKER_NUMA_NODE="${RERANKER_NUMA_NODE:-0}"
