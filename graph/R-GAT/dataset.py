"""
dataset related classes and methods
"""

# pylint: disable=unused-argument,missing-docstring

import logging


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("dataset")


class Dataset:
    def __init__(self):
        pass

    def preprocess(self, use_cache=True):
        raise NotImplementedError("Dataset:preprocess")

    def get_item_count(self):
        return NotImplementedError("Dataset:get_item_count")

    def get_list(self):
        raise NotImplementedError("Dataset:get_list")

    def load_query_samples(self, sample_list):
        pass

    def unload_query_samples(self, sample_list):
        pass

    def get_samples(self, id_list):
        pass

    def get_item(self, id):
        raise NotImplementedError("Dataset:get_item")


def preprocess(id):
    return id
