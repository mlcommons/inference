#!/bin/bash
# This example will start serving the 345M model.
DISTRIBUTED_ARGS="--nproc_per_node 8 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

CHECKPOINT=<Path to checkpoint>
VOCAB_FILE=$HOME/inference/language/gpt3/data/vocab.json
MERGE_FILE=$HOME/inference/language/gpt3/data/merges.txt

export CUDA_DEVICE_MAX_CONNECTIONS=1

pip install flask-restful

torchrun $DISTRIBUTED_ARGS text_generation_server.py   \
       --tensor-model-parallel-size 8  \
       --pipeline-model-parallel-size 1  \
       --num-layers 96  \
       --hidden-size 12288  \
       --num-attention-heads 96  \
       --max-position-embeddings 2048  \
       --tokenizer-type GPT2BPETokenizer  \
       --fp16  \
       --micro-batch-size 1  \
       --seq-length 2048  \
       --vocab-file $VOCAB_FILE  \
       --merge-file $MERGE_FILE  \
       --seed 42
       --load ${CHECKPOINT}  \