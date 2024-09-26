#! /usr/bin/env bash

# Copyright 2017 Google Inc.
# Modifications copyright (C) 2019 The MLPerf Authors.
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


#
# @description: This script preprocesses raw input text files, before running it through GNMT.
# It must be applied on a pair of input files: an English text, along with its German translation.
# Example usage:
# * Assume two text files are present: newstest2014.en and newstest2014.de
# * Run ./preprocess_input.sh newstest2014
# * Example output: newstest2014.tok.bpe.en and newstest2014.tok.bpe.de
#
# @note: If the input corpus is in the SGM format, one must use the 
# /mosesdecoder/scripts/ems/support/input-from-sgm.perl script to convert
#
# @note: This script is based on https://github.com/mlperf/training/blob/master/rnn_translator/download_dataset.sh
#

fbase=$1 # basename of the input text file (e.g., newstest2014 for {newstest2014.en, newstest2014.de})
SCRIPTS="helper_scripts"
BPE_CODES="./nmt/data/bpe.32000"

# Clone Moses
if [ ! -d "${SCRIPTS}/mosesdecoder" ]; then
  echo "Cloning moses for data processing"
  git clone https://github.com/moses-smt/mosesdecoder.git "${SCRIPTS}/mosesdecoder"
  cd ${SCRIPTS}/mosesdecoder
  git reset --hard 8c5eaa1a122236bbf927bde4ec610906fea599e6
  cd -
fi

# Clone Subword NMT
if [ ! -d "${SCRIPTS}/subword-nmt" ]; then
  echo "Cloning subword-nmt for data processing"
  git clone https://github.com/rsennrich/subword-nmt.git "${SCRIPTS}/subword-nmt"
  cd ${SCRIPTS}/subword-nmt
  git reset --hard 48ba99e657591c329e0003f0c6e32e493fa959ef
  cd -
fi

# Assumes raw text input
$SCRIPTS/mosesdecoder/scripts/tokenizer/tokenizer.perl -q -l de -threads 8 < ${fbase}.de > ${fbase}.tok.de
$SCRIPTS/mosesdecoder/scripts/tokenizer/tokenizer.perl -q -l en -threads 8 < ${fbase}.en > ${fbase}.tok.en

$SCRIPTS/subword-nmt/subword_nmt/apply_bpe.py -c $BPE_CODES < ${fbase}.tok.en > ${fbase}.tok.bpe.en
$SCRIPTS/subword-nmt/subword_nmt/apply_bpe.py -c $BPE_CODES < ${fbase}.tok.de > ${fbase}.tok.bpe.de 
