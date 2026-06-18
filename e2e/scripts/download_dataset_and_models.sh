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

echo "============================================================"
echo "E2E-RAG Dataset and Model Downloader"
echo "============================================================"
echo ""
echo "This script will download:"
echo "  - FRAMES Dataset (~674KB)"
echo "  - Embedding Model e5-base-v2 (~2.2GB)"
echo "  - Reranker Model ColBERTv2.0 (~1.4GB)"
echo "  - GPT-OSS-120B Model (~196GB)"
echo "  - GPT-OSS-20B Model (~83GB)"
echo ""
echo "Total download size: ~283GB"
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
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
    https://inference.mlcommons-storage.org/metadata/gpt-oss-model.uri

echo ""
echo "============================================================"
echo "Downloading GPT-OSS-20B Model"
echo "============================================================"
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
    https://inference.mlcommons-storage.org/metadata/gpt-oss-20B.uri

echo ""
echo "============================================================"
echo "Download Complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Download Wikipedia documents:"
echo "     python3 download_docs.py --output_dir doc_html --format html --processes 30"
echo ""
echo "  2. Build vector database:"
echo "     bash reference_mlperf_datasetup.sh"
echo ""
echo "  3. Start LLM servers:"
echo "     vllm serve /data/model/gpt-oss-20b --port 8123"
echo "     vllm serve /data/model/gpt-oss-120b --port 8124"
echo "     vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8125"
echo ""
echo "  4. Run QA workload:"
echo "     bash reference_mlperf_accuracy.sh"
echo ""
