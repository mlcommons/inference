#! /usr/bin/env bash

# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

export LANG=C.UTF-8
export LC_ALL=C.UTF-8

# OUTPUT_DIR=${1:-"data"}
OUTPUT_DIR="$PWD/outputs"
echo $OUTPUT_DIR

echo "Writing to ${OUTPUT_DIR}. To change this, set the OUTPUT_DIR environment variable."

OUTPUT_DIR_DATA="${OUTPUT_DIR}/data"

mkdir -p $OUTPUT_DIR_DATA

echo "Downloading dev/test sets"
wget -nc -nv -O  ${OUTPUT_DIR_DATA}/dev.tgz \
  http://data.statmt.org/wmt16/translation-task/dev.tgz

mkdir -p "${OUTPUT_DIR_DATA}/dev"
tar -xvzf "${OUTPUT_DIR_DATA}/dev.tgz" -C "${OUTPUT_DIR_DATA}/dev"

# Clone Moses
if [ ! -d "${OUTPUT_DIR}/mosesdecoder" ]; then
  echo "Cloning moses for data processing"
  git clone https://github.com/moses-smt/mosesdecoder.git "${OUTPUT_DIR}/mosesdecoder"
  cd ${OUTPUT_DIR}/mosesdecoder
  git reset --hard 8c5eaa1a122236bbf927bde4ec610906fea599e6
  cd -
fi

# Convert SGM files
# Convert newstest2014 data into raw text format
${OUTPUT_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
  < ${OUTPUT_DIR_DATA}/dev/dev/newstest2014-deen-src.de.sgm \
  > ${OUTPUT_DIR_DATA}/dev/dev/newstest2014.de
${OUTPUT_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
  < ${OUTPUT_DIR_DATA}/dev/dev/newstest2014-deen-ref.en.sgm \
  > ${OUTPUT_DIR_DATA}/dev/dev/newstest2014.en

# Copy dev/test data to output dir
cp ${OUTPUT_DIR_DATA}/dev/dev/newstest2014.de ${OUTPUT_DIR}
cp ${OUTPUT_DIR_DATA}/dev/dev/newstest2014.en ${OUTPUT_DIR}

# Modify dataset
echo "Modifying Dataset..."
python modify_gnmt_data.py --filename="${OUTPUT_DIR}/newstest2014"
mv "${OUTPUT_DIR}/newstest2014.en" "${OUTPUT_DIR}/newstest2014.original.en"
mv "${OUTPUT_DIR}/newstest2014.new.en" "${OUTPUT_DIR}/newstest2014.en"

# Tokenize data
for f in ${OUTPUT_DIR}/*.de; do
  echo "Tokenizing $f..."
  ${OUTPUT_DIR}/mosesdecoder/scripts/tokenizer/tokenizer.perl -q -l de -threads 8 < $f > ${f%.*}.tok.de
done

for f in ${OUTPUT_DIR}/*.en; do
  echo "Tokenizing $f..."
  ${OUTPUT_DIR}/mosesdecoder/scripts/tokenizer/tokenizer.perl -q -l en -threads 8 < $f > ${f%.*}.tok.en
done

# Clean all corpora
#for f in ${OUTPUT_DIR}/*.en; do
#  fbase=${f%.*}
#  echo "Cleaning ${fbase}..."
#  ${OUTPUT_DIR}/mosesdecoder/scripts/training/clean-corpus-n.perl $fbase de en "${fbase}.clean" 1 80
#done

# # Create dev dataset
# cat "${OUTPUT_DIR}/newstest2015.tok.clean.en" \
#    "${OUTPUT_DIR}/newstest2016.tok.clean.en" \
#    > "${OUTPUT_DIR}/newstest_dev.tok.clean.en"

# cat "${OUTPUT_DIR}/newstest2015.tok.clean.de" \
#    "${OUTPUT_DIR}/newstest2016.tok.clean.de" \
#    > "${OUTPUT_DIR}/newstest_dev.tok.clean.de"

# # Filter datasets
# python3 pytorch/scripts/filter_dataset.py -f1 ${OUTPUT_DIR}/train.tok.clean.en -f2 ${OUTPUT_DIR}/train.tok.clean.de
# python3 pytorch/scripts/filter_dataset.py -f1 ${OUTPUT_DIR}/newstest_dev.tok.clean.en -f2 ${OUTPUT_DIR}/newstest_dev.tok.clean.de

# Generate Subword Units (BPE)
# Clone Subword NMT
if [ ! -d "${OUTPUT_DIR}/subword-nmt" ]; then
  git clone https://github.com/rsennrich/subword-nmt.git "${OUTPUT_DIR}/subword-nmt"
  cd ${OUTPUT_DIR}/subword-nmt
  git reset --hard 48ba99e657591c329e0003f0c6e32e493fa959ef
  cd -
fi

# # Learn Shared BPE
# for merge_ops in 32000; do
#   echo "Learning BPE with merge_ops=${merge_ops}. This may take a while..."
#   cat "${OUTPUT_DIR}/train.tok.de" "${OUTPUT_DIR}/train.tok.en" | \
#     ${OUTPUT_DIR}/subword-nmt/learn_bpe.py -s $merge_ops > "${OUTPUT_DIR}/bpe.${merge_ops}"

#   echo "Apply BPE with merge_ops=${merge_ops} to tokenized files..."
#   for lang in en de; dols
#     for f in ${OUTPUT_DIR}/*.tok.${lang} ${OUTPUT_DIR}/*.tok.${lang}; do
#       outfile="${f%.*}.bpe.${merge_ops}.${lang}"
#       ${OUTPUT_DIR}/subword-nmt/apply_bpe.py -c "${OUTPUT_DIR}/bpe.${merge_ops}" < $f > "${outfile}"
#       echo ${outfile}
#     done
#   done

#   # Create vocabulary file for BPE
#   cat "${OUTPUT_DIR}/train.tok.bpe.${merge_ops}.en" "${OUTPUT_DIR}/train.tok.bpe.${merge_ops}.de" | \
#     ${OUTPUT_DIR}/subword-nmt/get_vocab.py | cut -f1 -d ' ' > "${OUTPUT_DIR}/vocab.bpe.${merge_ops}"
# done
ORIGINAL_DATASET="/gpfs/fs1/datasets/mlperf_inference/preprocessed_data/nmt/GNMT/"
CUSTOM_DATASET_OUTPUT="/gpfs/fs1/anirbang/custom_datasets/mlperf_inference/preprocessed_data/nmt/GNMT"
BPE_CODES_ORIGINAL="${ORIGINAL_DATASET}/bpe.32000"
BPE_CODES="${OUTPUT_DIR}/bpe.32000"
cp ${BPE_CODES_ORIGINAL} ${BPE_CODES}

fbase="${OUTPUT_DIR}/newstest2014"
${OUTPUT_DIR}/mosesdecoder/scripts/tokenizer/tokenizer.perl -q -l de -threads 8 < ${fbase}.de > ${fbase}.tok.de
${OUTPUT_DIR}/mosesdecoder/scripts/tokenizer/tokenizer.perl -q -l de -threads 8 < ${fbase}.en > ${fbase}.tok.en
${OUTPUT_DIR}/subword-nmt/subword_nmt/apply_bpe.py -c $BPE_CODES < ${fbase}.tok.en > ${fbase}.tok.bpe.en
${OUTPUT_DIR}/subword-nmt/subword_nmt/apply_bpe.py -c $BPE_CODES < ${fbase}.tok.de > ${fbase}.tok.bpe.de 


echo "Copying original dataset to custom directory"
cp -r ${ORIGINAL_DATASET}/* ${CUSTOM_DATASET_OUTPUT}/
echo "Replacing original dataset files with custom dataset files"
cp "${OUTPUT_DIR}/newstest2014.tok.bpe.en" "${CUSTOM_DATASET_OUTPUT}/newstest2014.tok.bpe.32000.en"
cp "${OUTPUT_DIR}/newstest2014.tok.bpe.de" "${CUSTOM_DATASET_OUTPUT}/newstest2014.tok.bpe.32000.de"

echo "Preparing perf mode dataset"
rm "${CUSTOM_DATASET_OUTPUT}/newstest2014.tok.bpe.32000.en.large"
for i in {1..1300};do cat "${CUSTOM_DATASET_OUTPUT}/newstest2014.tok.bpe.32000.en" >> "${CUSTOM_DATASET_OUTPUT}/newstest2014.tok.bpe.32000.en.large"; done

echo "All done."
