#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

set -e
set -u

ROOD_DIR="$(realpath $(dirname "$0"))"
DST_DIR="$ROOD_DIR/pre-trained_language_models"

mkdir -p "$DST_DIR"
cd "$DST_DIR"

echo "BERT TINY LOWERCASED"
if [[ ! -f bert/uncased_L-2_H-128_A-2/bert_config.json ]]; then
  mkdir -p 'bert'
  cd bert
  git clone https://huggingface.co/prajjwal1/bert-tiny uncased_L-2_H-128_A-2
  cd uncased_L-2_H-128_A-2
  wget -c "https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-2_H-128_A-2.zip"
  unzip -o uncased_L-2_H-128_A-2.zip
  rm uncased_L-2_H-128_A-2.zip
  rm bert_model*
  cd ../../
fi

echo "BERT MINI LOWERCASED"
if [[ ! -f bert/uncased_L-4_H-256_A-4/bert_config.json ]]; then
  mkdir -p 'bert'
  cd bert
  git clone https://huggingface.co/prajjwal1/bert-mini uncased_L-4_H-256_A-4
  cd uncased_L-4_H-256_A-4
  wget -c "https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-4_H-256_A-4.zip"
  unzip -o uncased_L-4_H-256_A-4.zip
  rm uncased_L-4_H-256_A-4.zip
  rm bert_model*
  cd ../../
fi

echo "BERT SMALL LOWERCASED"
if [[ ! -f bert/uncased_L-4_H-512_A-8/bert_config.json ]]; then
  mkdir -p 'bert'
  cd bert
  git clone https://huggingface.co/prajjwal1/bert-small uncased_L-4_H-512_A-8
  cd uncased_L-4_H-512_A-8
  wget -c "https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-4_H-512_A-8.zip"
  unzip -o uncased_L-4_H-512_A-8.zip
  rm uncased_L-4_H-512_A-8.zip
  rm bert_model*
  cd ../../
fi

echo "BERT MEDIUM LOWERCASED"
if [[ ! -f bert/uncased_L-8_H-512_A-8/bert_config.json ]]; then
  mkdir -p 'bert'
  cd bert
  git clone https://huggingface.co/prajjwal1/bert-medium uncased_L-8_H-512_A-8
  cd uncased_L-8_H-512_A-8
  wget -c "https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-8_H-512_A-8.zip"
  unzip -o uncased_L-8_H-512_A-8.zip
  rm uncased_L-8_H-512_A-8.zip
  rm bert_model*
  cd ../../
fi

echo "BERT BASE LOWERCASED"
if [[ ! -f bert/uncased_L-12_H-768_A-12/bert_config.json ]]; then
  mkdir -p 'bert'
  cd bert
  wget -c "https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip"
  unzip uncased_L-12_H-768_A-12.zip
  rm uncased_L-12_H-768_A-12.zip
  cd uncased_L-12_H-768_A-12
  wget -c "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz"
  tar -xzf bert-base-uncased.tar.gz
  rm bert-base-uncased.tar.gz
  rm bert_model*
  cd ../../
fi

echo "BERT LARGE LOWERCASED"
if [[ ! -f bert/uncased_L-24_H-1024_A-16/bert_config.json ]]; then
  mkdir -p 'bert'
  cd bert
  wget -c "https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip"
  unzip uncased_L-24_H-1024_A-16.zip
  rm uncased_L-24_H-1024_A-16.zip
  cd uncased_L-24_H-1024_A-16
  wget -c "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz"
  tar -xzf bert-large-uncased.tar.gz
  rm bert-large-uncased.tar.gz
  rm bert_model*
  cd ../../
fi

echo "BERT BASE MULTILINGUAL LOWERCASED"
if [[ ! -f bert/multilingual_L-12_H-768_A-12/bert_config.json ]]; then
  mkdir -p 'bert'
  cd bert
  wget -c "https://storage.googleapis.com/bert_models/2018_11_03/multilingual_L-12_H-768_A-12.zip"
  unzip multilingual_L-12_H-768_A-12.zip
  rm multilingual_L-12_H-768_A-12.zip
  cd multilingual_L-12_H-768_A-12
  wget -c "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz"
  tar -xzf bert-base-multilingual-uncased.tar.gz
  rm bert-base-multilingual-uncased.tar.gz
  rm bert_model*
  cd ../../
fi

echo 'cased models'

echo "BERT BASE CASED"
if [[ ! -f bert/cased_L-12_H-768_A-12/bert_config.json ]]; then
  mkdir -p 'bert'
  cd bert
  wget -c "https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip"
  unzip cased_L-12_H-768_A-12
  rm cased_L-12_H-768_A-12.zip
  cd cased_L-12_H-768_A-12
  wget -c "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz"
  tar -xzf bert-base-cased.tar.gz
  rm bert-base-cased.tar.gz
  rm bert_model*
  cd ../../
fi

echo "BERT LARGE CASED"
if [[ ! -f bert/cased_L-24_H-1024_A-16/bert_config.json ]]; then
  mkdir -p 'bert'
  cd bert
  wget -c "https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip"
  unzip cased_L-24_H-1024_A-16.zip
  rm cased_L-24_H-1024_A-16.zip
  cd cased_L-24_H-1024_A-16
  wget -c "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz"
  tar -xzf bert-large-cased.tar.gz
  rm bert-large-cased.tar.gz
  rm bert_model*
  cd ../../
fi

echo "BERT BASE MULTILINGUAL CASED"
if [[ ! -f bert/multi_cased_L-12_H-768_A-12/bert_config.json ]]; then
  mkdir -p 'bert'
  cd bert
  wget -c "https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip"
  unzip multi_cased_L-12_H-768_A-12.zip
  rm multi_cased_L-12_H-768_A-12.zip
  cd multi_cased_L-12_H-768_A-12
  wget -c "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz"
  tar -xzf bert-base-multilingual-cased.tar.gz
  rm bert-base-multilingual-cased.tar.gz
  rm bert_model*
  cd ../../
fi


cd "$ROOD_DIR"
echo 'Building common vocab'
if [ ! -f "$DST_DIR/common_vocab_cased.txt" ]; then
  python lama/vocab_intersection.py
else
  echo 'Already exists. Run to re-build:'
  echo 'python util_KB_completion.py'
fi

