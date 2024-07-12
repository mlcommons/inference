# TODO:submission 전 삭제 필요, test 용 파일입니다. 
import os
import time
import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from tokenizer_GPTJ import get_transformer_autotokenizer
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from typing import Optional, Dict, Sequence
import io
import copy
import json


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

        self.tokenizer = get_transformer_autotokenizer(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.list_data_dict = jload(self.dataset_path)
        
        NUM_SPLITS=os.environ.get('NUM_SPLITS', None)
        SPLIT_IDX=os.environ.get('SPLIT_IDX', None)

        assert NUM_SPLITS is not None
        assert SPLIT_IDX is not None

        NUM_SPLITS = int(NUM_SPLITS)
        SPLIT_IDX = int(SPLIT_IDX)
        
        if NUM_SPLITS > 1:
            n_splited_data = int(len(self.list_data_dict)/NUM_SPLITS)
            start_idx = SPLIT_IDX*n_splited_data
            end_idx= (SPLIT_IDX+1)*n_splited_data if SPLIT_IDX!=NUM_SPLITS-1 else len(self.list_data_dict) + 1
            self.list_data_dict = self.list_data_dict[start_idx:end_idx]

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


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict
