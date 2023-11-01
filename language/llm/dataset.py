"""
dataset related classes and methods
"""

# pylint: disable=unused-argument,missing-docstring

import logging
import sys
import time

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("dataset")

class Dataset():
    def __init__(self):
        self.arrival = None
        self.input_text_list = []
        self.text_list_inmemory = {}
        self.last_loaded = -1

    def preprocess(self, use_cache=True):
        raise NotImplementedError("Dataset:preprocess")

    def get_item_count(self):
        return len(self.input_text_list)

    def get_list(self):
        raise NotImplementedError("Dataset:get_list")

    def load_query_samples(self, sample_list):
        self.text_list_inmemory = {}
        for sample in sample_list:
            self.text_list_inmemory[sample] = self.get_item(sample)
        self.last_loaded = time.time()

    def unload_query_samples(self, sample_list):
        if sample_list:
            for sample in sample_list:
                if sample in self.text_list_inmemory :
                    del self.text_list_inmemory[sample]
        else:
            self.text_list_inmemory = {}

    def get_samples(self, id_list):
        prompts = [self.text_list_inmemory[id]["prompt"] for id in id_list]
        label = [self.text_list_inmemory[id]["output"] for id in id_list]
        return prompts, label

    def get_item(self, id):
        raise NotImplementedError("Dataset:get_item")