import os
import time
import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from typing import Optional, Dict, Sequence
import io
import utils
import copy

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


class Dataset():
    def __init__(self, dataset_path, batch_size=1, pad_val=1, pad_max=196, total_count_override=None, perf_count_override=None):
        print("Constructing QSL")

        self.dataset = "cnn_dailymail"
        self.model_name = "EleutherAI/gpt-j-6B"
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.pad_val = pad_val
        self.pad_max = pad_max

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            model_max_length=2048,
            padding_side="left",
            use_fast=False,)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.list_data_dict = utils.jload(self.dataset_path)

        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        self.sources = [prompt_input.format_map(
            example) for example in self.list_data_dict]
        self.targets = [
            f"{example['output']}" for example in self.list_data_dict]

        self.source_encoded_input_ids, self.source_encoded_attn_masks = self.encode_samples()

        self.count = total_count_override or len(self.sources)
        self.perf_count = perf_count_override or self.count

    def encode_samples(self):
        print("Encoding Samples")

        total_samples = len(self.sources)

        source_encoded_input_ids = []
        source_encoded_attn_masks = []

        for i in range(total_samples):
            source_encoded = self.tokenizer(self.sources[i], return_tensors="pt",
                                            padding=True, truncation=True,
                                            max_length=1919)
            source_encoded_input_ids.append(source_encoded.input_ids)
            source_encoded_attn_masks.append(source_encoded.attention_mask)

        return source_encoded_input_ids, source_encoded_attn_masks

    def LoadSamplesToRam(self, sample_list):
        pass

    def UnloadSamplesFromRam(self, sample_list):
        pass

    def __del__(self):
        print("Finished destroying QSL.")
