"""
dataset related classes and methods
"""

# pylint: disable=unused-argument,missing-docstring

import logging
import sys
import time

import numpy as np


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("dataset")


class Dataset:
    def __init__(self):
        self.items_inmemory = {}

    def preprocess(self, use_cache=True):
        raise NotImplementedError("Dataset:preprocess")

    def get_item_count(self):
        raise NotImplementedError("Dataset:get_item_count")

    def get_list(self):
        raise NotImplementedError("Dataset:get_list")

    def load_query_samples(self, sample_list):
        raise NotImplementedError("Dataset:load_query_samples")

    def unload_query_samples(self, sample_list):
        raise NotImplementedError("Dataset:unload_query_samples")

    def get_samples(self, id_list):
        raise NotImplementedError("Dataset:get_samples")

    def get_item(self, id):
        raise NotImplementedError("Dataset:get_item")


def preprocess(img):
    return img
