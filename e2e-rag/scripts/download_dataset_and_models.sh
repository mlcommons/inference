#!/bin/bash
# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

# Download all required datasets and models from MLCommons storage

set -e

# Download an MLCommons asset without the folders vLLM does not load
# (original/ and metal/ weight copies, .git/). Uses the bucket's own .uri and
# .md5 list, verifies every file, and skips files already downloaded.
download_without_extras() {   # <asset name, as in metadata/<name>.uri>
    local meta="https://inference.mlcommons-storage.org/metadata"
    local base dir hash path
    base=$(curl -sfL "${meta}/$1.uri")
    dir="${base##*/}"
    mkdir -p "${dir}"
    curl -sfL "${meta}/$1.md5" | grep -v -E '/(original|metal|\.git)/' > "${dir}/$1.md5"
    while read -r hash path; do
        if [ -f "${dir}/${path}" ] && echo "${hash}  ${dir}/${path}" | md5sum -c --status; then
            continue
        fi
        echo "  ${dir}/${path}"
        mkdir -p "${dir}/$(dirname "${path}")"
        # Resume a partial file; if it still fails the check, fetch it fresh.
        curl -sfL --retry 3 -C - -o "${dir}/${path}" "${base}/${path}" || true
        if ! echo "${hash}  ${dir}/${path}" | md5sum -c --status; then
            rm -f "${dir}/${path}"
            curl -sfL --retry 3 -o "${dir}/${path}" "${base}/${path}"
            echo "${hash}  ${dir}/${path}" | md5sum -c --status || {
                echo "ERROR: MD5 mismatch for ${dir}/${path}" >&2; exit 1; }
        fi
    done < "${dir}/$1.md5"
}

echo "============================================================"
echo "E2E-RAG Dataset and Model Downloader"
echo "============================================================"
echo ""
echo "This script will download:"
echo "  - FRAMES Dataset (~674KB)"
echo "  - Embedding Model e5-base-v2 (~2.2GB)"
echo "  - Reranker Model ColBERTv2.0 (~1.4GB)"
echo "  - GPT-OSS-120B Model (~65GB)"
echo "  - GPT-OSS-20B Model (~14GB)"
echo ""
echo "Total download size: ~83GB"
echo "Ensure you have sufficient disk space before proceeding."
echo ""

# Prompt for confirmation
read -p "Continue with download? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Download cancelled."
    exit 0
fi

echo ""
echo "============================================================"
echo "Downloading FRAMES Dataset"
echo "============================================================"
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
    https://inference.mlcommons-storage.org/metadata/frames-benchmark-dataset.uri
# The bucket ships the questions as frames-benchmark/test.tsv; the run scripts
# read frames-benchmark-dataset/frames_dataset.tsv.
ln -sfn frames-benchmark/test.tsv frames-benchmark-dataset/frames_dataset.tsv

echo ""
echo "============================================================"
echo "Extracting frozen document corpus (docs.tar.gz)"
echo "============================================================"
# The benchmark ships a FROZEN Wikipedia corpus as docs.tar.gz. Every submission
# MUST build its vector DB from this exact corpus -- do NOT re-scrape Wikipedia
# with download_docs.py, which fetches whatever revision is live today and
# produces a different corpus (different bytes, passage counts, and retrieval
# results), causing DB-manifest verification to fail across systems.
CORPUS_ARCHIVE="frames-benchmark-dataset/doc_html/docs.tar.gz"
CORPUS_DIR="doc_html"
if [ -f "${CORPUS_ARCHIVE}" ]; then
    mkdir -p "${CORPUS_DIR}"
    tar -xzf "${CORPUS_ARCHIVE}" -C "${CORPUS_DIR}"
    # Ship the fixed URL mapping alongside the HTML so ingestion records the
    # canonical original_url for each document.
    cp "frames-benchmark-dataset/doc_html/url_mapping.json" "${CORPUS_DIR}/" 2>/dev/null || true
    HTML_COUNT=$(find "${CORPUS_DIR}" -maxdepth 1 -name '*.html' | wc -l)
    echo "Extracted ${HTML_COUNT} HTML documents to ${CORPUS_DIR}/"
else
    echo "WARNING: ${CORPUS_ARCHIVE} not found; corpus was not extracted."
fi

echo ""
echo "============================================================"
echo "Downloading Embedding Model (e5-base-v2)"
echo "============================================================"
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
    https://inference.mlcommons-storage.org/metadata/intfloat_e5-base-v2.uri

echo ""
echo "============================================================"
echo "Downloading Reranker Model (ColBERTv2.0)"
echo "============================================================"
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
    https://inference.mlcommons-storage.org/metadata/colbert-ir_colbertv2.0.uri

echo ""
echo "============================================================"
echo "Downloading GPT-OSS-120B Model"
echo "============================================================"
download_without_extras gpt-oss-model

echo ""
echo "============================================================"
echo "Downloading GPT-OSS-20B Model"
echo "============================================================"
download_without_extras gpt-oss-20B

echo ""
echo "============================================================"
echo "Download Complete!"
echo "============================================================"
echo ""
echo "Downloaded files:"
echo "  - Dataset: data/frames_dataset.tsv"
echo "  - Document corpus: doc_html/ (extracted from frozen docs.tar.gz)"
echo "  - Embedding model: intfloat_e5-base-v2/e5-base-v2/"
echo "  - Reranker model: colbert-ir_colbertv2.0/colbertv2.0/"
echo "  - GPT-OSS-120B: gpt-oss-model/"
echo "  - GPT-OSS-20B: gpt-oss-20B/"
echo ""
echo "NOTE: The document corpus is FROZEN. Build your vector DB from the"
echo "      extracted doc_html/ above. Do NOT re-scrape Wikipedia with"
echo "      download_docs.py -- doing so fetches a different (live) revision and"
echo "      will fail cross-system DB-manifest verification."
echo ""
echo "Next steps:"
echo "  1. Build vector database (uses doc_html/ extracted above):"
echo "     bash reference_mlperf_datasetup.sh"
echo ""
echo "  2. Start LLM servers (adjust paths to your downloaded models):"
echo "     vllm serve gpt-oss-20B/ --port 8123"
echo "     vllm serve gpt-oss-model/ --port 8124"
echo "     vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8125"
echo ""
echo "  3. Run QA workload:"
echo "     bash reference_mlperf_accuracy.sh"
echo ""
