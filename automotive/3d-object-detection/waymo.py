"""
implementation of coco dataset
"""

# pylint: disable=unused-argument,missing-docstring

import json
import logging
import os
import time

from PIL import Image
import numpy as np
import pandas as pd
import dataset

import torch


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("coco")


class Waymo(dataset.Dataset):
    def __init__(
        self,
        # TODO: Dataset Arguments
        **kwargs,
    ):
        super().__init__()
        # TODO: initialize dataset arguments

    def preprocess(self, input):
        # TODO: implement preprocess of single input, alternatively
        # if it is possible to preprocess all the val dataset this  
        # can be called in the init function and preprocess the whole
        # dataset
        pass

    def get_item_count(self):
        # TODO: Get the size of the val dataset
        pass

    def get_list(self):
        raise NotImplementedError("Dataset:get_list")

    def load_query_samples(self, sample_list):
        # TODO: Load queries into memory, if needed
        pass

    def unload_query_samples(self, sample_list):
        # TODO: Unload queries from memory, if needed
        pass

    def get_samples(self, id_list):
        return [self.get_item(id) for id in id_list]

    def get_item(self, id):
        # TODO: Get an item from the dataset, corresponding
        # to the given id 
        pass


class PostProcessWaymo:
    def __init__(
        self, # Postprocess parameters
    ):
        self.content_ids = []
        # TODO: Init Postprocess parameters
        pass

    def add_results(self, results):
        self.results.extend(results)

    def __call__(self, results, ids):
        self.content_ids.extend(ids)
        # TODO: Postprocess results
        return []

    def start(self):
        self.results = []

    def finalize(self, result_dict, ds=None):
        

        return result_dict
