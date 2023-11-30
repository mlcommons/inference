"""
dataset related classes and methods
"""

# pylint: disable=unused-argument,missing-docstring

import logging
import sys
import time

import cv2
import numpy as np


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("dataset")


class Dataset:
    def __init__(self):
        self.arrival = None
        self.image_list = []
        self.caption_list = []
        self.items_inmemory = {}
        self.last_loaded = -1

    def preprocess(self, use_cache=True):
        raise NotImplementedError("Dataset:preprocess")

    def get_item_count(self):
        return len(self.image_list)

    def get_list(self):
        raise NotImplementedError("Dataset:get_list")

    def load_query_samples(self, sample_list):
        self.items_inmemory = {}
        for sample in sample_list:
            self.items_inmemory[sample] = self.get_item(sample)
        self.last_loaded = time.time()

    def unload_query_samples(self, sample_list):
        if sample_list:
            for sample in sample_list:
                if sample in self.items_inmemory:
                    del self.items_inmemory[sample]
        else:
            self.items_inmemory = {}

    def get_samples(self, id_list):
        data = [
            {
                "input_tokens": self.items_inmemory[id]["input_tokens"],
                "input_tokens_2": self.items_inmemory[id]["input_tokens_2"],
                "latents": self.items_inmemory[id]["latents"],
            }
            for id in id_list
        ]
        images = [self.items_inmemory[id]["file_name"] for id in id_list]
        return data, images

    def get_item(self, id):
        raise NotImplementedError("Dataset:get_item")


def preprocess(img):
    return img
