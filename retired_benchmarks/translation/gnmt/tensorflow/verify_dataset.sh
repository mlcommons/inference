#!/bin/bash

set -e

data_dir=${DATA_DIR:-./}
pushd $data_dir
ACTUAL_SRC_TEST=`cat nmt/data/newstest2014.tok.bpe.32000.en |md5sum`
EXPECTED_SRC_TEST='cb014e2509f86cd81d5a87c240c07464  -'
if [[ $ACTUAL_SRC_TEST = $EXPECTED_SRC_TEST ]]; then
  echo "OK: correct nmt/data/newstest2014.tok.bpe.32000.en"
else
  echo "ERROR: incorrect nmt/data/newstest2014.tok.bpe.32000.en"
  echo "ERROR: expected $EXPECTED_SRC_TEST"
  echo "ERROR: found $ACTUAL_SRC_TEST"
fi


ACTUAL_TGT_TEST=`cat nmt/data/newstest2014.tok.bpe.32000.de |md5sum`
EXPECTED_TGT_TEST='d616740f6026dc493e66efdf9ac1cb04  -'
if [[ $ACTUAL_TGT_TEST = $EXPECTED_TGT_TEST ]]; then
  echo "OK: correct nmt/data/newstest2014.tok.bpe.32000.de"
else
  echo "ERROR: incorrect nmt/data/newstest2014.tok.bpe.32000.de"
  echo "ERROR: expected $EXPECTED_TGT_TEST"
  echo "ERROR: found $ACTUAL_TGT_TEST"
fi


ACTUAL_VOCAB_DE=`cat nmt/data/vocab.bpe.32000.de |md5sum`
EXPECTED_VOCAB_DE='bbf724dc2e0ad4b35926d8557e34f465  -'
if [[ $ACTUAL_VOCAB_DE = $EXPECTED_VOCAB_DE ]]; then
  echo "OK: correct nmt/data/vocab.bpe.32000.de"
else
  echo "ERROR: incorrect nmt/data/vocab.bpe.32000.de"
  echo "ERROR: expected $EXPECTED_VOCAB_DE"
  echo "ERROR: found $ACTUAL_VOCAB_DE"
fi

ACTUAL_VOCAB_EN=`cat nmt/data/vocab.bpe.32000.en |md5sum`
EXPECTED_VOCAB_EN='bbf724dc2e0ad4b35926d8557e34f465  -'
if [[ $ACTUAL_VOCAB_EN = $EXPECTED_VOCAB_EN ]]; then
  echo "OK: correct nmt/data/vocab.bpe.32000.en"
else
  echo "ERROR: incorrect nmt/data/vocab.bpe.32000.en"
  echo "ERROR: expected $EXPECTED_VOCAB_EN"
  echo "ERROR: found $ACTUAL_VOCAB_EN"
fi
popd
