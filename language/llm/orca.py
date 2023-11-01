"""
implementation of orca dataset
"""

# pylint: disable=unused-argument,missing-docstring

import json
import logging
import os
import time

import pandas as pd
import dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("orca")

class Orca(dataset.Dataset):
    def __init__(self, data_path, name = None, use_tokens = True, **kwargs):
        super().__init__()
        self.prompts_df = pd.read_pickle(data_path)
        self.use_tokens = use_tokens
        self.name = name
        self.inputs = []
        self.token_inputs = []
        self.outputs = []
        self.token_outputs = []

        for i, row in self.prompts_df.iterrows():
            self.inputs.append(row["input"])
            self.token_inputs.append(row["tok_input"])
            self.outputs.append(row["outputs"])
            self.token_outputs.append(row["tok_outputs"])


    def get_item(self, id):
        if self.use_tokens:
            return self.token_inputs[id], self.token_outputs[id]
        else:
            return self.inputs[id], self.outputs[id]
    
    def get_item_count(self):
        return len(self.prompts_df)
