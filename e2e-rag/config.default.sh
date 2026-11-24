# =============================================================================
# config.default.sh: defaults for every run script. Do not edit; put your
# changes in config.sh (copy config.template.sh) or export them.
#
# Every script sources config.sh, then this file, so per variable (first wins):
#   1. Exported env var
#   2. config.sh
#   3. This file
# Both files are sourced with `set -a`, so every value reaches Python too.
# =============================================================================

# ── Paths ─────────────────────────────────────────────────────────────────────
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"        # run logs and outputs
DATASET_PATH="${DATASET_PATH:-frames-benchmark-dataset/frames_dataset.tsv}"

# ── Models: local path or Hugging Face id ────────────────────────────────────
EMBEDDING_MODEL="${EMBEDDING_MODEL:-intfloat_e5-base-v2/e5-base-v2}"
RERANKER_MODEL="${RERANKER_MODEL:-colbert-ir_colbertv2.0/colbertv2.0}"
QUERY_MODEL="${QUERY_MODEL:-gpt-oss_model/gpt-oss-120b}"
SUFFICIENCY_MODEL="${SUFFICIENCY_MODEL:-${QUERY_MODEL}}"
GRADER_MODEL="${GRADER_MODEL:-gpt-oss-20B/gpt-oss-20b}"
JUDGE_MODEL="${JUDGE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"

# ── LLM endpoints (scripts/launch_llm_server.sh serves on these ports) ────────
QUERY_SERVICE_URL="${QUERY_SERVICE_URL:-http://127.0.0.1:8123/v1/chat/completions}"
SUFFICIENCY_SERVICE_URL="${SUFFICIENCY_SERVICE_URL:-${QUERY_SERVICE_URL}}"
GRADER_SERVICE_URL="${GRADER_SERVICE_URL:-http://127.0.0.1:8192/v1/chat/completions}"
JUDGE_SERVICE_URL="${JUDGE_SERVICE_URL:-http://127.0.0.1:8125/v1/chat/completions}"

# ── RAG-DB (reference_mlperf_datasetup*.sh, run_datasetup_kpi.sh,
#    scripts/run_ingestion.sh) ─────────────────────────────────────────────────
INGESTION_DOC_DIR="${INGESTION_DOC_DIR:-doc_html}"
INGESTION_CHUNK_SIZE="${INGESTION_CHUNK_SIZE:-768}"
INGESTION_CHUNK_OVERLAP="${INGESTION_CHUNK_OVERLAP:-32}"
INGESTION_TEXT_BOUNDARY="${INGESTION_TEXT_BOUNDARY:-word}"
INGESTION_DB="${INGESTION_DB:-vector_html_hnsw_len${INGESTION_CHUNK_SIZE}_ov${INGESTION_CHUNK_OVERLAP}_${INGESTION_TEXT_BOUNDARY}}"
INGESTION_DB="${INGESTION_DB%.db}"                  # name without .db
INGESTION_PASSAGES_JSON="${INGESTION_PASSAGES_JSON:-passages/doc_html_len${INGESTION_CHUNK_SIZE}_ov${INGESTION_CHUNK_OVERLAP}_${INGESTION_TEXT_BOUNDARY}.json}"
INGESTION_DEVICE="${INGESTION_DEVICE:-auto}"        # auto, cuda, xpu, hpu, cpu
INGESTION_EMBEDDING_DEVICE="${INGESTION_EMBEDDING_DEVICE:-${INGESTION_DEVICE}}"
INGESTION_NUM_EMBEDDING_DEVICES="${INGESTION_NUM_EMBEDDING_DEVICES:-1}"
INGESTION_MAX_WORKERS="${INGESTION_MAX_WORKERS:-4}"
INGESTION_VECTOR_INDEX_METHOD="${INGESTION_VECTOR_INDEX_METHOD:-hnsw}"
INGESTION_BENCHMARK="${INGESTION_BENCHMARK:-false}"

# DB verification against the reference manifest (empty MANIFEST: skip)
MANIFEST="${MANIFEST-scripts/db_manifest_intel_xpu.json.gz}"
COSINE_THRESHOLD="${COSINE_THRESHOLD:-0.9999}"
TOP_K_DEPTH="${TOP_K_DEPTH:-3}"

# ── RAG-QnA (reference_mlperf_{perf,accuracy}.sh, run_compliance_test09.sh,
#    scripts/run_*.sh) ─────────────────────────────────────────────────────────
DATABASE="${DATABASE:-${INGESTION_DB}.db}"
SCENARIO="${SCENARIO:-Offline}"
QNA_DEVICE="${QNA_DEVICE:-auto}"                    # auto, cuda, xpu, hpu, cpu
QNA_MAX_WORKERS="${QNA_MAX_WORKERS:-10}"            # parallel questions
MAX_ITERATIONS="${MAX_ITERATIONS:-5}"               # fixed by the rules
MAX_SUB_QUERIES="${MAX_SUB_QUERIES:-3}"
TOP_K_RETRIEVER="${TOP_K_RETRIEVER:-10}"
TOTAL_SAMPLE_COUNT="${TOTAL_SAMPLE_COUNT:-0}"       # 0: user.conf decides
PERF_CACHE_FILE="${PERF_CACHE_FILE:-logs_result.json}"   # recorded LLM responses

# scripts/run_{multi_shot,single_shot}.sh only (reference_SUT.py uses QNA_DEVICE)
QNA_EMBEDDING_DEVICE="${QNA_EMBEDDING_DEVICE:-${QNA_DEVICE}}"
RERANKER_DEVICE="${RERANKER_DEVICE:-${QNA_DEVICE}}"

# ── Read by Python only; unset means automatic ───────────────────────────────
# GPU indices per component (subset of the visible CUDA/XPU devices):
#   EMBEDDING_GPU_DEVICES="0,1"     one entry per embedding worker
#   RERANKER_GPU_DEVICES="2"
# NUMA / OMP placement of the per-component workers:
#   RERANKER_NUMA_NODE=0            RERANKER_OMP_NUM_THREADS=21
#   EMBEDDING_NUMA_NODES="0,0,1,1"  EMBEDDING_OMP_NUM_THREADS=21
# When a model runs on CPU (apply_cpu_threading_env):
#   CPU_DISABLE_NUMA=1  CPU_NUMA_NODE=0  CPU_NUMA_CORES="43-85"  CPU_OMP_NUM_THREADS=43
