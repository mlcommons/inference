# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import argparse
import pandas as pd
import numpy as np
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from transformers import LlamaTokenizerFast
from typing import Dict

__doc__ = """
This script takes the open_orca GPT4 dataset parquet and perform the following preprocessing and filtering steps:
1. filter out all queries with non-ascii characters, except for normal unicode quotes and hyphens.
2. filter out all queries with out-of-bound input/output sequence lengths
3. filter out all queries with expected answers shorter than 2 words (known to cause issues for Llama2)
4. filter out all queries with prompts that generate bad output texts using Llama2 models
4. sample equally from the sub-dataset (i.e. COT, NIV, FLAN, T0) and form the final dataset.
"""

llama_prompt_system = "<s>[INST] <<SYS>>\n{}\n<</SYS>>\n\n{} [/INST]"
llama_prompt_no_system = "<s>[INST] {} [/INST]"

def format_llama_input(row):
    if row['system_prompt']:
        return llama_prompt_system.format(row['system_prompt'], row['question'])
    else:
        return llama_prompt_no_system.format(row['question'])

def is_english(s):
    for c in s:
        allowed = c.isascii()
        allowed = allowed or (c in ['’', '–', '“', '”', '—'])  # Taken from Habana: Unicode quotes and hyphens
        if not allowed:
            return False
    return True


def _tokenize_helper(x, llama_tokenizer=None, append_response_init_token=True):
    if not isinstance(x, str):
        return []

    tokens = llama_tokenizer(x)["input_ids"]

    if append_response_init_token:
        # Workaround to enable cheat checking for first token: Llama always outputs token 29871 first
        # It is possible for submitters to just immediately output this token to achieve a very fast TTFT.
        tokens.append(29871)
    return tokens


@dataclass
class Keyphrase:
    col: str
    phrase: str
    startswith: bool = False
    case: bool = False


class OpenOrcaDatasetGenerator:
    def __init__(self,
                 pq_path: os.PathLike,
                 model_dir: os.PathLike,
                 io_token_limit: int,
                 calibration_subset_size: int = 1000):
        self.pq_path = Path(pq_path)
        self.model_dir = Path(model_dir)
        self.io_token_limit = io_token_limit
        self.keyphrases = []
        self.calibration_subset_size = calibration_subset_size

    def load_parquet(self) -> pd.DataFrame:
        llama_tokenizer = LlamaTokenizerFast.from_pretrained(self.model_dir)

        tik = time.time()
        df = pd.read_parquet(self.pq_path)
        print(f"Tokenizing input")
        df.rename(columns={'response': 'output'}, inplace=True)
        df['input'] = df.apply(format_llama_input, axis=1)

        input_tokenizer = partial(_tokenize_helper, llama_tokenizer=llama_tokenizer)
        output_tokenizer = partial(_tokenize_helper, llama_tokenizer=llama_tokenizer, append_response_init_token=False)
        df['tok_input'] = df['input'].apply(input_tokenizer)
        df['tok_output'] = df['output'].apply(output_tokenizer)
        tok = time.time()
        print(f"Loaded parquet and tokenized in {tok-tik} sec.")
        return df

    def filter_english(self, df: pd.DataFrame) -> pd.DataFrame:
        df['input_english'] = df['input'].apply(is_english)
        df['output_english'] = df['output'].apply(is_english)
        df['all_english'] = df['input_english'] & df['output_english']

        # Filter based on english tokens
        df = df[df['all_english']].drop(["input_english", "output_english", "all_english"], axis=1)
        return df.reset_index(drop=True)

    def filter_seqlen_oob(self, df: pd.DataFrame) -> pd.DataFrame:
        df['tok_input_length'] = df['tok_input'].apply(lambda x: len(x))
        df['tok_output_length'] = df['tok_output'].apply(lambda x: len(x))

        # Filter based on sequence length
        df = df[df["tok_input_length"] < self.io_token_limit]
        df = df[df["tok_output_length"] < self.io_token_limit]
        return df.reset_index(drop=True)

    def filter_short_expected_response(self, df: pd.DataFrame) -> pd.DataFrame:
        # We have found that short expected responses (such as for yes/no and true/false questions, or multiple choice
        # questions where the expected response is just the choice, i.e. (B)), disproportionately have lower Rouge1
        # scores (< 0.02).

        # Filter out 1 and 2 word expected responses. We've seen best results when this is filtered to >= 6, but it is
        # hard to justify removing that many samples.
        df = df[df["tok_output_length"] >= 3]
        return df.reset_index(drop=True)

    def filter_bad_prompts(self, df: pd.DataFrame, only_niv_t0: bool = True) -> pd.DataFrame:
        # Some prompts underperform and cause very bad Rouge scores for a significant percentage of samples with these
        # prompts. See Jupyter notebook for analysis.
        # These generally only affect NIV and t0 and do not exist in flan or cot.
        # Set 'only_niv_t0' to True to explicitly only remove these prompts from niv and t0 samples.
        bad_prompts = ['',
                       'You are an AI assistant that follows instruction extremely well. Help as much as you can.',
                       'You are an AI assistant. Provide a detailed answer so user don’t need to search outside to understand the answer.',
                       "You are an AI assistant. Provide a detailed answer so user don't need to search outside to understand the answer.",
                       'User will you give you a task with some instruction. Your job is follow the instructions as faithfully as you can. While answering think step-by-step and justify your answer.',
                       'Explain how you used the definition to come up with the answer.',
                       ]
        for prompt in bad_prompts:
            criteria = (df.system_prompt == prompt)
            if only_niv_t0:
                criteria = criteria & ((df.origin == "niv") | (df.origin == "t0"))
            df = df[~criteria]

        return df.reset_index(drop=True)

    def register_keyphrase(self, keyphrase: Keyphrase):
        self.keyphrases.append(keyphrase)

    def filter_keyphrases(self, df: pd.DataFrame) -> pd.DataFrame:
        # Filter out registered keyphrases. This is unused for the final dataset as there are no registered keyphrases.
        for kp in self.keyphrases:
            if kp.startswith:
                selector = df[kp.col].str.startswith(kp.phrase)
            else:
                selector = df[kp.col].str.contains(kp.phrase, case=kp.case)
            df = df[~selector]
        return df.reset_index(drop=True)

    def set_origins(self, df: pd.DataFrame) -> pd.DataFrame:
        get_sample_origin = lambda x: x.split(".")[0]
        df['origin'] = df['id'].apply(get_sample_origin)
        return df

    def _per_origin_split(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        print(f"Unique sample origin datasets: {df.origin.unique()}")
        dfs_by_origin = dict(tuple(df.groupby('origin')))
        for origin, sub_df in dfs_by_origin.items():
            sub_df = sub_df.reset_index(drop=True, inplace=True)
        return dfs_by_origin

    def _get_sampling(self, df, N, rng_seed: int = 1337):
        _N = min(df.shape[0], N)
        if _N < N:
            raise RuntimeError(f"Not enough samples. Requires {N - _N} more.")
        return df.sample(n=_N, random_state=rng_seed)

    def sample(self, dfs_by_origin: Dict[str, pd.DataFrame], n_total, rng_seed: int = 1337) -> pd.DataFrame:
        nways = len(dfs_by_origin)
        assert n_total % nways == 0, f"Total number of samples ({n_total}) must be divisible by n_origins ({nways})"

        split_size = n_total // nways
        samplings = []
        for origin, df in dfs_by_origin.items():
            print(f"Sampling {split_size} from {origin}")
            sample = self._get_sampling(df, split_size, rng_seed=rng_seed)
            samplings.append(sample)

        sampled_df = pd.concat(samplings)
        sampled_df = sampled_df.reset_index(drop=True)
        return sampled_df

    def generate(self,
                 export_dir: os.PathLike,
                 n_samples: int = 24576,
                 use_cached: bool = True,
                 calib_rng_seed: int = 12345):
        export_dir = Path(export_dir)
        if not export_dir.exists():
            print(f"Creating {export_dir}")
            export_dir.mkdir(parents=True)
        if export_dir.is_file():
            raise ValueError(f"Cannot export to file {export_dir}. Must be a directory.")

        full_fpath = export_dir / f"open_orca_gpt4_tokenized_llama.full.pkl"
        if full_fpath.exists() and use_cached:
            df = pd.read_pickle(full_fpath)
        else:
            df = self.load_parquet()
            df = self.set_origins(df)

            # Apply filters
            df = self.filter_english(df)
            df = self.filter_seqlen_oob(df)
            df = self.filter_short_expected_response(df)
            df = self.filter_bad_prompts(df)
            df = self.filter_keyphrases(df)
            df.to_pickle(full_fpath)

        dfs_by_origin = self._per_origin_split(df)

        # Export base files
        for origin, sub_df in dfs_by_origin.items():
            print(f"Subset '{origin}' has {sub_df.shape[0]} samples")
            origin_fpath = export_dir / f"open_orca_gpt4_tokenized_llama.{origin}.pkl"
            if not origin_fpath.exists() or not use_cached:
                sub_df.to_pickle(origin_fpath)

        # Strategy:
        # After some analysis, we found that OpenOrca's dataset has a skewed "origin-dataset" distribution:
        # cot and niv have significantly fewer samples (71K and 58K) compared to flan and t0 (375K and 278K).
        # cot has a higher rouge score from a 100k sampling (of the whole dataset) than the rest, while niv has lower.
        # Sample from each dataset equally.
        sampled_df = self.sample(dfs_by_origin, n_samples)
        sampled_fpath = export_dir / f"open_orca_gpt4_tokenized_llama.sampled_{n_samples}.pkl"
        sampled_df.to_pickle(sampled_fpath)

        # Calibration dataset
        calib_ds = sampled_df.sample(n=self.calibration_subset_size,
                                     random_state=calib_rng_seed)
        calib_ds = calib_ds.reset_index(drop=True)
        calib_fpath = export_dir / f"open_orca_gpt4_tokenized_llama.calibration_{self.calibration_subset_size}.pkl"
        calib_ds.to_pickle(calib_fpath)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_pq_path', type=str,
                        default='/raid/data/mlperf-llm/OpenOrca/1M-GPT4-Augmented.parquet',
                        help="the path to the open_orca GPT4 parquet.")
    parser.add_argument('--model_dir', type=str, default='/raid/data/mlperf-llm/Llama-2-70b-chat-hf')
    parser.add_argument('--seqlen_limit', type=int, default=1024, help="Upper limit of the input/output sequence lengths")
    parser.add_argument('--export_dir', type=str,
                        default="/raid/data/mlperf-llm/OpenOrca/llama/filtered",
                        help="Path to the output pkl file.")
    parser.add_argument('--num_total_samples', type=int, default=24576, help="Number of samples to generate")
    parser.add_argument('--calibration_subset_size', type=int, default=1000, help="Number of samples for calibration subset")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    ds_gen = OpenOrcaDatasetGenerator(
        pq_path=args.dataset_pq_path,
        model_dir=args.model_dir,
        io_token_limit=args.seqlen_limit,
        calibration_subset_size=args.calibration_subset_size,
    )
    ds_gen.generate(
        export_dir=args.export_dir,
        n_samples=args.num_total_samples,
    )

    # Sample command to run:
    # python3 processorca.py --dataset_pq_path=/raid/data/mlperf-llm/OpenOrca/1M-GPT4-Augmented.parquet --model_dir=/raid/data/mlperf-llm/Llama-2-70b-chat-hf --seqlen_limit=1024 --export_dir=/raid/data/mlperf-llm/OpenOrca/llama/filtered --num_total_samples=24576
