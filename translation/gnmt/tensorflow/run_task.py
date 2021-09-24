#!/usr/bin/env python
import sys
import os
import argparse
import time
import re
from os.path import expanduser
import numpy as np
import json
import subprocess

import os.path

iterations = 1

parser = argparse.ArgumentParser()

parser.add_argument('--run', type=str, default='performance',
                    help="Specify either 'accuracy' for BLEU metric or "
                    "'performance' (default) for prediction latency and throughput"
)


parser.add_argument('--batch_size', type=str, default='32',
                    help="Specify inference batch size"
)

parser.add_argument('--num_inter_threads', type=str, default='0',
                    help="Specify inference num_inter_threads"
)

parser.add_argument('--num_intra_threads', type=str, default='0',
                    help="Specify inference num_intra_threads"
)

parser.add_argument('--dataset_path', type=str, 
                    default=os.path.join(os.getcwd(), 'nmt', 'data'),
                    help="Specify dataset directory path"
)

parser.add_argument('--model_path', type=str, 
                    default=os.path.join(os.getcwd(), 'ende_gnmt_model_4_layer'),
                    help="Specify model directory path"
)

parser.add_argument('--output_path', type=str,
                    default=os.path.join(os.getcwd(), 'nmt', 'data'),
                    help="Specify output directory path"
)



args = parser.parse_args()

cpk_path = os.path.join(args.model_path, 'translate.ckpt')

haparams_path = os.path.join(os.getcwd(), 'nmt', 'standard_hparams',
                             'wmt16_gnmt_4_layer.json')

vocab_prefix = os.path.join(args.dataset_path, 'vocab.bpe.32000')

inference_ref_file = os.path.join(args.dataset_path, 'newstest2014.tok.bpe.32000.de')

inference_input_file = os.path.join(args.dataset_path, 'newstest2014.tok.bpe.32000.en')

out_dir = os.path.join(args.output_path, 'result', 'output')

inference_output_file = os.path.join(args.output_path, 'output', 'g_nmt-out')

outpath = os.path.join(args.output_path, 'output', 'console_out_gnmt.txt')

cmd = "python -m nmt.nmt \
    --src=en --tgt=de \
    --ckpt="+cpk_path+" \
    --hparams_path="+haparams_path+" \
    --out_dir="+out_dir+" \
    --vocab_prefix="+vocab_prefix+" \
    --inference_input_file="+inference_input_file+" \
    --inference_output_file="+inference_output_file+" \
    --inference_ref_file="+inference_ref_file+" \
    --infer_batch_size="+args.batch_size+" \
    --num_inter_threads="+args.num_inter_threads+" \
    --num_intra_threads="+args.num_intra_threads+" \
    --iterations="+str(iterations)+" \
    --run="+args.run

return_code = subprocess.call(cmd, shell=True)
