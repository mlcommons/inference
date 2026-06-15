#!/bin/bash
# Complete Indexing Pipeline with Chunking + Validation
#
# This script runs:
# 1. Document chunking (optional, if raw documents provided)
# 2. Indexing performance measurement
# 3. Database correctness verification (optional, if manifest provided)
# 4. Combines results into unified JSON output

set -e  # Exit on error

echo "============================================================"
echo "Complete Indexing Pipeline (Chunking + Performance + Validation)"
echo "============================================================"
echo "Time Start: $(date +%s)"
echo ""

# Configuration
export DOCUMENTS_DIR=${DOCUMENTS_DIR:-""}
export PASSAGES_FILE=${PASSAGES_FILE:-"passages/doc_html_len768_ov32_word.json"}
export DATABASE=${DATABASE:-"vector_complete"}
export DATASET=${DATASET:-"data/frames_dataset.tsv"}

# Chunking configuration
export CHUNK_SIZE=${CHUNK_SIZE:-768}
export CHUNK_OVERLAP=${CHUNK_OVERLAP:-32}
export TEXT_BOUNDARY=${TEXT_BOUNDARY:-"word"}
export CHUNKING_PROCESSES=${CHUNKING_PROCESSES:-4}

# Vector database configuration
export VECTOR_INDEX_METHOD=${VECTOR_INDEX_METHOD:-"hnsw"}
export DEVICE=${DEVICE:-"auto"}
export NUM_EMBEDDING_DEVICES=${NUM_EMBEDDING_DEVICES:-1}

# Model paths
export RETRIEVER_MODEL=${RETRIEVER_MODEL:-/data/model/e5-base-v2}
export RERANKER_MODEL=${RERANKER_MODEL:-/data/model/colbertv2.0}

# Performance options
export LOAD_EMBEDDINGS=${LOAD_EMBEDDINGS:-false}
export BENCHMARK=${BENCHMARK:-false}
export HIERARCHICAL=${HIERARCHICAL:-false}

# Manifest options
export MANIFEST_FILE=${MANIFEST_FILE:-""}
export SKIP_MANIFEST_VERIFY=${SKIP_MANIFEST_VERIFY:-false}
export CREATE_MANIFEST=${CREATE_MANIFEST:-false}

# Output files
export TEMP_KPI_METRICS="temp_complete_kpi_$$.json"
export FINAL_OUTPUT=${FINAL_OUTPUT:-"data_setup_performance.json"}

# Determine input mode
if [ -n "${DOCUMENTS_DIR}" ]; then
    INPUT_MODE="complete_with_chunking"
    INPUT_SOURCE="${DOCUMENTS_DIR}"
elif [ -n "${PASSAGES_FILE}" ]; then
    INPUT_MODE="indexing_only"
    INPUT_SOURCE="${PASSAGES_FILE}"
else
    echo "ERROR: Either DOCUMENTS_DIR or PASSAGES_FILE must be set"
    exit 1
fi

# Print configuration
echo "Configuration:"
echo "  Mode: ${INPUT_MODE}"
echo "  Input: ${INPUT_SOURCE}"
echo "  DATABASE: ${DATABASE}"
if [ "${INPUT_MODE}" = "complete_with_chunking" ]; then
    echo "  CHUNK_SIZE: ${CHUNK_SIZE}"
    echo "  CHUNK_OVERLAP: ${CHUNK_OVERLAP}"
    echo "  CHUNKING_PROCESSES: ${CHUNKING_PROCESSES}"
fi
echo "  VECTOR_INDEX_METHOD: ${VECTOR_INDEX_METHOD}"
echo "  DEVICE: ${DEVICE}"
echo "  NUM_EMBEDDING_DEVICES: ${NUM_EMBEDDING_DEVICES}"
echo "  MANIFEST_FILE: ${MANIFEST_FILE:-none (skip validation)}"
echo "  CREATE_MANIFEST: ${CREATE_MANIFEST}"
echo "  FINAL_OUTPUT: ${FINAL_OUTPUT}"
echo "============================================================"
echo ""

# ============================================================
# CHUNKING WARNING (if applicable)
# ============================================================
if [ "${INPUT_MODE}" = "complete_with_chunking" ]; then
    echo "============================================================"
    echo "⚠️  DOCUMENT CHUNKING WILL BE PERFORMED"
    echo "============================================================"
    echo ""
    echo "Documents will be chunked from scratch with the following parameters:"
    echo "  📄 Source directory: ${INPUT_SOURCE}"
    echo "  📏 Chunk size: ${CHUNK_SIZE} characters"
    echo "  🔗 Chunk overlap: ${CHUNK_OVERLAP} characters"
    echo "  📐 Text boundary: ${TEXT_BOUNDARY}"
    echo "  ⚙️  Parallel processes: ${CHUNKING_PROCESSES}"
    echo ""
    echo "This will process all documents in the directory regardless of"
    echo "whether passages already exist. The chunking step cannot be skipped"
    echo "when using DOCUMENTS_DIR mode."
    echo ""
    echo "To use pre-chunked passages instead, use:"
    echo "  PASSAGES_FILE=passages/your_file.json ./run_complete_pipeline_with_chunking.sh"
    echo "============================================================"
    echo ""
fi

# ============================================================
# STEP 1: CHUNKING + INDEXING PERFORMANCE
# ============================================================
echo "============================================================"
echo "STEP 1/3: Indexing Performance Measurement"
if [ "${INPUT_MODE}" = "complete_with_chunking" ]; then
    echo "         (includes chunking)"
fi
echo "============================================================"
echo ""

INDEXING_START=$(date +%s)

# Build arguments for indexing script
if [ "${INPUT_MODE}" = "complete_with_chunking" ]; then
    ARGS="--documents ${DOCUMENTS_DIR}"
    ARGS="${ARGS} --chunk_size ${CHUNK_SIZE}"
    ARGS="${ARGS} --chunk_overlap ${CHUNK_OVERLAP}"
    ARGS="${ARGS} --text_boundary ${TEXT_BOUNDARY}"
    ARGS="${ARGS} --chunking_processes ${CHUNKING_PROCESSES}"
else
    ARGS="--ingest ${PASSAGES_FILE}"
fi

ARGS="${ARGS} --database ${DATABASE}"
ARGS="${ARGS} --output_metrics ${TEMP_KPI_METRICS}"
ARGS="${ARGS} --vector_index_method ${VECTOR_INDEX_METHOD}"
ARGS="${ARGS} --device ${DEVICE}"
ARGS="${ARGS} --num_embedding_devices ${NUM_EMBEDDING_DEVICES}"
ARGS="${ARGS} --retriever_model ${RETRIEVER_MODEL}"
ARGS="${ARGS} --reranker_model ${RERANKER_MODEL}"

if [ "${LOAD_EMBEDDINGS}" = "true" ]; then
    ARGS="${ARGS} --load_embeddings"
fi

if [ "${BENCHMARK}" = "true" ]; then
    ARGS="${ARGS} --benchmark"
fi

if [ "${HIERARCHICAL}" = "true" ]; then
    ARGS="${ARGS} --hierarchical"
fi

# Run indexing with or without chunking
python3 measure_indexing_with_chunking.py ${ARGS}
INDEXING_EXIT=$?

INDEXING_END=$(date +%s)
INDEXING_DURATION=$((INDEXING_END - INDEXING_START))

if [ ${INDEXING_EXIT} -ne 0 ]; then
    echo ""
    echo "❌ Indexing measurement FAILED"
    exit ${INDEXING_EXIT}
fi

echo ""
echo "✅ Indexing measurement completed in ${INDEXING_DURATION}s"
echo ""

# Extract key metrics
VECTOR_COUNT=$(jq -r '.vector_count' ${TEMP_KPI_METRICS} 2>/dev/null || echo "0")
THROUGHPUT=$(jq -r '.throughput_docs_per_second' ${TEMP_KPI_METRICS} 2>/dev/null || echo "0")
TOTAL_TIME=$(jq -r '.total_indexing_time_seconds' ${TEMP_KPI_METRICS} 2>/dev/null || echo "0")

echo "Performance Summary:"
echo "  Vectors indexed: ${VECTOR_COUNT}"
echo "  Total time: ${TOTAL_TIME}s"
echo "  Throughput: ${THROUGHPUT} docs/sec"
echo ""

# ============================================================
# STEP 2: CREATE MANIFEST (Optional)
# ============================================================
if [ "${CREATE_MANIFEST}" = "true" ]; then
    echo "============================================================"
    echo "STEP 2A: Creating Database Manifest"
    echo "============================================================"
    echo ""

    MANIFEST_OUTPUT="db_manifest_$(hostname -s)_$(date +%Y%m%d_%H%M%S).json"

    python3 db_manifest.py write \
        --db ${DATABASE} \
        --retriever_model ${RETRIEVER_MODEL} \
        --dataset ${DATASET} \
        --output ${MANIFEST_OUTPUT}

    MANIFEST_CREATE_EXIT=$?

    if [ ${MANIFEST_CREATE_EXIT} -ne 0 ]; then
        echo "⚠️  Manifest creation failed (non-fatal)"
    else
        echo "✅ Manifest created: ${MANIFEST_OUTPUT}"
    fi
    echo ""
fi

# ============================================================
# STEP 3: DATABASE CORRECTNESS VERIFICATION (Optional)
# ============================================================
MANIFEST_VERIFY_RESULT="skipped"
MANIFEST_VERIFY_DETAILS=""
MANIFEST_CHECKS_PASSED=0
MANIFEST_CHECKS_TOTAL=0

if [ -n "${MANIFEST_FILE}" ] && [ "${SKIP_MANIFEST_VERIFY}" != "true" ]; then
    echo "============================================================"
    echo "STEP 2/3: Database Correctness Verification"
    echo "============================================================"
    echo ""

    if [ ! -f "${MANIFEST_FILE}" ]; then
        echo "⚠️  Manifest file not found: ${MANIFEST_FILE}"
        echo "   Skipping verification"
        MANIFEST_VERIFY_RESULT="manifest_not_found"
    else
        VERIFY_START=$(date +%s)

        # Run verification and capture output
        VERIFY_OUTPUT=$(python3 db_manifest.py verify \
            --db ${DATABASE} \
            --manifest ${MANIFEST_FILE} 2>&1)
        VERIFY_EXIT=$?

        VERIFY_END=$(date +%s)
        VERIFY_DURATION=$((VERIFY_END - VERIFY_START))

        if [ ${VERIFY_EXIT} -eq 0 ]; then
            MANIFEST_VERIFY_RESULT="passed"
            echo "✅ Database correctness verification PASSED"

            # Parse checks from output
            MANIFEST_CHECKS_PASSED=$(echo "${VERIFY_OUTPUT}" | grep -c "✓" || echo "0")
            MANIFEST_CHECKS_TOTAL=${MANIFEST_CHECKS_PASSED}
        else
            MANIFEST_VERIFY_RESULT="failed"
            echo "❌ Database correctness verification FAILED"

            # Extract failure info
            MANIFEST_CHECKS_TOTAL=$(echo "${VERIFY_OUTPUT}" | grep -c "checking" || echo "0")
            MANIFEST_CHECKS_PASSED=$((MANIFEST_CHECKS_TOTAL - $(echo "${VERIFY_OUTPUT}" | grep -c "FAIL" || echo "0")))
        fi

        MANIFEST_VERIFY_DETAILS=$(echo "${VERIFY_OUTPUT}" | tail -20)

        echo ""
        echo "Verification Summary:"
        echo "  Result: ${MANIFEST_VERIFY_RESULT}"
        echo "  Checks passed: ${MANIFEST_CHECKS_PASSED}/${MANIFEST_CHECKS_TOTAL}"
        echo "  Duration: ${VERIFY_DURATION}s"
        echo ""
    fi
else
    echo "============================================================"
    echo "STEP 2/3: Database Correctness Verification - SKIPPED"
    echo "============================================================"
    echo ""
    echo "No manifest file provided. Set MANIFEST_FILE to enable verification."
    echo ""
fi

# ============================================================
# STEP 4: COMBINE RESULTS INTO UNIFIED JSON
# ============================================================
echo "============================================================"
echo "STEP 3/3: Generating Unified Results"
echo "============================================================"
echo ""

# Create unified JSON output
python3 - <<EOF
import json
import sys
from datetime import datetime

# Load indexing KPI metrics
try:
    with open("${TEMP_KPI_METRICS}") as f:
        kpi_metrics = json.load(f)
except Exception as e:
    print(f"Error loading KPI metrics: {e}", file=sys.stderr)
    kpi_metrics = {}

# Build unified result
result = {
    "pipeline_version": "2.0",
    "timestamp": datetime.now().isoformat(),
    "hostname": "$(hostname)",
    "status": "completed",

    "configuration": {
        "mode": "${INPUT_MODE}",
        "input_source": "${INPUT_SOURCE}",
        "database": "${DATABASE}",
        "dataset": "${DATASET}",
        "vector_index_method": "${VECTOR_INDEX_METHOD}",
        "device": "${DEVICE}",
        "num_embedding_devices": int("${NUM_EMBEDDING_DEVICES}"),
        "retriever_model": "${RETRIEVER_MODEL}",
        "reranker_model": "${RERANKER_MODEL}",
        "load_embeddings": "${LOAD_EMBEDDINGS}" == "true",
        "benchmark": "${BENCHMARK}" == "true",
        "hierarchical": "${HIERARCHICAL}" == "true"
    },

    "chunking": kpi_metrics.get("chunking", {
        "enabled": False,
        "time_seconds": 0
    }),

    "performance": {
        "vector_count": kpi_metrics.get("vector_count", 0),
        "indexing_time_seconds": kpi_metrics.get("indexing_time_seconds", 0),
        "save_time_seconds": kpi_metrics.get("save_time_seconds", 0),
        "total_indexing_time_seconds": kpi_metrics.get("total_indexing_time_seconds", 0),
        "total_time_with_save_seconds": kpi_metrics.get("total_time_with_save_seconds", 0),
        "throughput_docs_per_second": kpi_metrics.get("throughput_docs_per_second", 0),
        "embedding_model": kpi_metrics.get("configuration", {}).get("embedding_model", ""),
        "embedding_dimension": kpi_metrics.get("configuration", {}).get("embedding_dimension", 0)
    },

    "validation": {
        "status": "${MANIFEST_VERIFY_RESULT}",
        "manifest_file": "${MANIFEST_FILE}" if "${MANIFEST_FILE}" else None,
        "checks_passed": int("${MANIFEST_CHECKS_PASSED}"),
        "checks_total": int("${MANIFEST_CHECKS_TOTAL}"),
        "details": """${MANIFEST_VERIFY_DETAILS}""" if "${MANIFEST_VERIFY_DETAILS}" else None
    },

    "summary": {
        "performance_passed": kpi_metrics.get("throughput_docs_per_second", 0) > 0,
        "validation_passed": "${MANIFEST_VERIFY_RESULT}" in ["passed", "skipped"],
        "overall_status": "passed" if (
            kpi_metrics.get("throughput_docs_per_second", 0) > 0 and
            "${MANIFEST_VERIFY_RESULT}" in ["passed", "skipped"]
        ) else "failed"
    }
}

# Write unified output
with open("${FINAL_OUTPUT}", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
EOF

# Clean up temporary files
rm -f ${TEMP_KPI_METRICS}

echo ""
echo "============================================================"
echo "PIPELINE COMPLETE"
echo "============================================================"
echo ""

# Read and display summary
OVERALL_STATUS=$(jq -r '.summary.overall_status' ${FINAL_OUTPUT})
PERFORMANCE_PASSED=$(jq -r '.summary.performance_passed' ${FINAL_OUTPUT})
VALIDATION_PASSED=$(jq -r '.summary.validation_passed' ${FINAL_OUTPUT})

echo "Summary:"
echo "  Performance: $([ "$PERFORMANCE_PASSED" = "true" ] && echo "✅ PASSED" || echo "❌ FAILED")"
echo "  Validation: $([ "$VALIDATION_PASSED" = "true" ] && echo "✅ PASSED" || echo "❌ FAILED")"
echo "  Overall: $([ "$OVERALL_STATUS" = "passed" ] && echo "✅ PASSED" || echo "❌ FAILED")"
echo ""
echo "Detailed results saved to: ${FINAL_OUTPUT}"
echo ""
echo "View results:"
echo "  cat ${FINAL_OUTPUT} | jq '.'"
echo ""
echo "Time Stop: $(date +%s)"

# Exit with appropriate code
if [ "$OVERALL_STATUS" = "passed" ]; then
    exit 0
else
    exit 1
fi
