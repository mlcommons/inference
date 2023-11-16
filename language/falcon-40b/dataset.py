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
#import utils
import copy

import random
random.seed(9973)

#TODO Prompt may have to be experimented with. 
PROMPT_INPUT = "### System:\n{system_prompt}\n### Human:\n{question}\n### Assistant:\n"

class Dataset():
    def __init__(self, total_sample_count=24576, perf_count_override=None, dataset_path=None, device="cpu"):
        self.model_name = "tiiuae/falcon-40b-instruct"
        self.dataset_path = dataset_path
        self.max_length = 1024
        self.device = device

        self.total_sample_count = total_sample_count

        self.load_tokenizer()
        self.load_dataset()

        self.total_sample_count = len(self.input_ids)
        self.perf_count = perf_count_override or self.total_sample_count

    def load_tokenizer(self):
        """ Returns tokenizer """
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            model_max_length=1024,
            padding_side="left",
            use_fast=False,)

        self.tokenizer.pad_token = self.tokenizer.eos_token


    def load_dataset(self):
        """ Loads dataset. This may change after we finish creating the validation set"""

        list_data_dict = load_dataset("Open-Orca/OpenOrca")['train']
        num_samples = min(len(list_data_dict), self.total_sample_count)

        list_data_dict = random.choices(list_data_dict, k=num_samples)

        sources = [PROMPT_INPUT.format_map(example) for example in list_data_dict]
        targets = [ f"{example['response']}" for example in list_data_dict]

        self.input_ids = []
        self.input_lens = []
        self.attention_masks = []
        for i in range(len(sources)):
            tok_input = self.tokenize_function(sources[i])
            self.input_ids.append(tok_input.input_ids.to(self.device))
            self.attention_masks.append(tok_input.attention_mask.to(self.device))

    def postProcess(self, out_tokens, query_id_list=None, sample_index_list=None, input_seq_lens=None):
        """ Postprocesses output prediction """

        #TODO: Create response object in postProcess(?)
        preds = []
        for i in range(out_tokens.shape[0]):
            pred = out_tokens[i].reshape(-1).cpu().numpy() # Slice up to original input length as below?

            #input_len = input_seq_lens[i]
            #pred = out_tokens[i, input_len:].reshape(-1).cpu().numpy()
            preds.append(pred)
        
        return preds

    @torch.no_grad()
    def tokenize_function(self, text):
        example = self.tokenizer(text, padding=True, pad_to_multiple_of=64, truncation=True, max_length=self.max_length, return_tensors="pt")
        return example

    def LoadSamplesToRam(self, sample_list):
        pass

    def UnloadSamplesFromRam(self, sample_list):
        pass

    def __del__(self):
        pass
