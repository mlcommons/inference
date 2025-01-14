#!/usr/bin/env python3
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
import torch
import pandas as pd
import time
from pathlib import Path
import argparse


def run_infer(df, ckpt_path, bs):
    """
    dataset                                                           GSM8K
    id                                                            train.548
    question              Gary manages two Amazon distribution centers. ...
    input                 <s> [INST] As an expert problem solver solve s...
    ref_output            The first center processes 10000 packages per ...
    gt_output                                                         14000
    tok_input             [1, 1, 28705, 733, 16289, 28793, 1136, 396, 75...
    tok_ref_output        [415, 907, 4982, 9537, 28705, 28740, 28734, 28...
    stop_sequence                                                      </s>
    tok_stop_sequence                                                   [2]
    tok_input_len                                                       662
    tok_ref_output_len                                                  174
    Name: 0, dtype: object
    """
    device = "cuda"  # the device to load the model onto

    # Load the model from local if possible.
    model_path = Path(ckpt_path)
    if not model_path.exists():
        raise RuntimeError(
            f"{ckpt_path} not existed. Please download the checkpoint from mlcommon")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, padding_side="left", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # gen parameter. We stop at 1024. Starting from v5.0, min_token is set to
    # 2 to avoid 0-output issue
    gen_kwargs = {
        # "min_new_tokens": 1,
        "min_new_tokens": 2,
        "max_new_tokens": 1024,
        "do_sample": False,
        "temperature": None,
        "top_p": None,
    }

    # Start inference
    BS = bs
    bidx = 0
    model.eval()

    input_tokens = []
    input_tokens_lens = []
    output_tokens = []
    output_tokens_lens = []
    output_texts = []

    tic = time.time()
    for idx in range(0, len(df), BS):
        tac = time.time()
        print(f"Processing {idx}/{len(df)}, time: {tac - tic}s")
        sidx = idx
        eidx = min(sidx + BS, len(df))

        # We use batch_encode_plus for batch inference.
        # Note 9/29/2024: Mixtral changed its tokenizer in Jun. Using the Feb
        # 29 2024 version.
        batch_texts = df['input'][sidx:eidx].tolist()
        batch_ids = tokenizer.batch_encode_plus(
            batch_texts, return_tensors="pt", padding=True)
        # tok_input_length = batch_ids['attention_mask'].sum(
        #     axis=1).to(torch.int32).tolist()
        # input_tokens_lens += tok_input_length
        tok_input_id = batch_ids['input_ids'].to(torch.int32).tolist()
        # Remove eos from the input id
        tok_input_id = [[element for element in sublist if element !=
                        tokenizer.eos_token_id] for sublist in tok_input_id]
        input_tokens += tok_input_id
        tok_input_length = [len(seq) for seq in tok_input_id]
        input_tokens_lens += tok_input_length

        batch_ids = batch_ids.to(device)
        _, length = batch_ids.input_ids.shape
        outputs = model.generate(**batch_ids, num_return_sequences=1,
                                 **gen_kwargs)

        output_ids = outputs[:, length:].cpu().tolist()
        output_tokens += output_ids

        # Filter out EOS
        id_filtered = [[num for num in sublist if num !=
                        tokenizer.eos_token_id] for sublist in output_ids]
        output_id_len = [len(out) for out in id_filtered]
        output_tokens_lens += output_id_len

        # Detokenizer
        output_msgs = tokenizer.batch_decode(
            output_ids, skip_special_tokens=True)
        output_texts += output_msgs
        bidx += 1

    # Assemble the output
    output_df = df[:len(output_tokens)].copy()
    output_df["infer_tok_input"] = input_tokens
    output_df["infer_tok_input_length"] = input_tokens_lens
    output_df["infer_ref_output"] = output_texts
    output_df["infer_tok_ref_output"] = output_tokens
    output_df["infer_tok_ref_output_length"] = output_tokens_lens

    # output_df.to_pickle(f"mixtral_8x7b_all15k_{len(output_tokens)}_BS{BS}_greedy_reference_fp16_mintoken1.pkl")

    return output_df


def trim_twos(df):
    # Remove all trailing 2s except for 1
    def remove_trailing_twos(lst):
        count = 0
        for num in reversed(lst):
            if num == 2:
                count += 1
            else:
                break
        return lst[:-count] if count > 0 else lst

    df['infer_tok_ref_output'] = df['infer_tok_ref_output'].apply(
        remove_trailing_twos)
    df['trim_lengths'] = df['infer_tok_ref_output'].apply(len)
    df['tok_ref_output'] = df['tok_ref_output'].apply(remove_trailing_twos)
    df['tok_ref_output_len'] = df['tok_ref_output'].apply(len)
    return df


def mbxp_stop(df):
    stop_tokens = [13, 13940, 28832, 13]

    def modify_list(lst):
        for i in range(len(lst) - len(stop_tokens) + 1):
            if lst[i:i + len(stop_tokens)] == stop_tokens:
                return lst[:i + len(stop_tokens)]
        return lst

    df.loc[df['dataset'] == 'MBXP', 'infer_tok_ref_output'] = df[df['dataset']
                                                                 == 'MBXP']['infer_tok_ref_output'].apply(modify_list)
    df['trim_lengths'] = df['infer_tok_ref_output'].apply(len)
    return df


def fix_name(df):
    df.drop(columns=['ref_output'], inplace=True)
    df.drop(columns=['tok_ref_output'], inplace=True)
    df.drop(columns=['tok_ref_output_len'], inplace=True)
    df.drop(columns=['infer_tok_ref_output_length'], inplace=True)
    df.drop(columns=['infer_tok_input'], inplace=True)
    df.drop(columns=['infer_tok_input_length'], inplace=True)
    df.rename(columns={'infer_ref_output': 'ref_output'}, inplace=True)
    df.rename(columns={'infer_tok_ref_output': 'tok_ref_output'}, inplace=True)
    df.rename(columns={'trim_lengths': 'tok_ref_output_len'}, inplace=True)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pkl", type=str, default="09292024_mixtral_15k_mintoken2_v1.pkl",
                        help="The path to the input pkl file")
    parser.add_argument("--output_pkl", type=str, default="mixtral_8x7b_15000_greedy_reference_fp16_mintoken2.pkl",
                        help="The path to the output pickle.")
    parser.add_argument("--checkpoint_path", type=str, default="/raid/data/mlperf-llm/Mixtral-8x7B-Instruct-v0.1",
                        help="The path to the mixtral checkpoint")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size of the refernece inference")
    args = parser.parse_args()

    df = pd.read_pickle(args.input_pkl)
    df = run_infer(df, args.checkpoint_path, args.batch_size)

    df = trim_twos(df)
    df = mbxp_stop(df)
    df = fix_name(df)

    df.to_pickle(args.output_pkl)
