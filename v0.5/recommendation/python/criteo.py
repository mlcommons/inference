"""
implementation of imagenet dataset
"""

# pylint: disable=unused-argument,missing-docstring

import logging
import os
import sys
import re
import time

import numpy as np
import inspect
# pytorch
import torch
from torch.utils.data import Dataset, RandomSampler

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("criteo")

# add dlrm code path
try:
    dlrm_dir_path = os.environ['DLRM_DIR']
    sys.path.append(dlrm_dir_path)
except KeyError:
    print("ERROR: Please set DLRM_DIR environment variable to the dlrm code location")
    sys.exit(0)
#import dataset
#import data_loader_terabyte as dltb
import dlrm_data_pytorch as dp

class Criteo(Dataset):

    def __init__(self, data_path, name, pre_process, use_cache, count, test_num_workers, max_ind_range=-1, sub_sample_rate=0.0, randomize="total", memory_map=False):
        # debug print
        print('Criteo __init__', data_path, name, pre_process, use_cache, count, test_num_workers, max_ind_range, sub_sample_rate, randomize, memory_map)
        super().__init__()

        if True:
            dataset_name = "kaggle"
            #raw_data_file = data_path + "/train.txt"
            raw_data_file = data_path + "/train_tiny2.txt"
            print('CriteoTerabyte  raw_data_file ', raw_data_file)
            processed_data_file = data_path + "/kaggleAdDisplayChallenge_processed.npz"
            print('CriteoTerabyte  processed_data_file ', processed_data_file)
        else:
            dataset_name = "terabyte"
            raw_data_file = data_path + "/day"
            processed_data_file = data_path + "/terabyte_processed.npz"

        self.test_data = dp.CriteoDataset(
            dataset=dataset_name,
            max_ind_range=max_ind_range,
            sub_sample_rate=sub_sample_rate,
            randomize=randomize,
            split="test",
            raw_path=raw_data_file,
            pro_data=processed_data_file,
            memory_map=memory_map
        )


        self.test_loader = torch.utils.data.DataLoader(
            self.test_data,
            batch_size=1,
            shuffle=False,
            num_workers=test_num_workers,
            collate_fn=dp.collate_wrapper_criteo,
            pin_memory=False,
            drop_last=False,  # True
        )

    def get_item_count(self):
        return len(self.test_data)

    ''' lg compatibilty routine '''
    def unload_query_samples(self, sample_list):
        self.items_in_memory = {}

    ''' lg compatibilty routine '''
    def load_query_samples(self, sample_list):

        self.items_in_memory = {}

        for l in sample_list:

            self.items_in_memory[l] = (self.test_data.X_int[l], self.test_data.X_cat[l], self.test_data.y[l])

        self.last_loaded = time.time()

    ''' lg compatibilty routine '''
    def get_samples(self, id_list):

        ls = []

        # build list tuples as need by the batch conversion routine
        for i in id_list:
            ls.append(self.items_in_memory[i])

        X, lS_o, lS_i, T = self.test_loader.collate_fn(ls)
        # debug print
        # print('get_samples', (X, lS_o, lS_i, T))

        return (X, lS_o, lS_i, T)


# Pre  processing
def pre_process_criteo_dlrm(x):
    return x


# Post processing
class DlrmPostProcess:
    def __init__(self):
        self.good = 0
        self.total = 0

    def __call__(self, results, expected=None, result_dict=None):
        processed_results = []
        n = len(results)
        for idx in range(0, n):
            result = results[idx]
            processed_results.append([result])
            # debug prints
            # print(result.__class__)
            # print(result.type())
            # print(result)
            # print(expected[idx].__class__)
            # print(expected[idx].type())
            # print(expected[idx])

            if result.round() == expected[idx]:
                self.good += 1
        self.total += n
        return processed_results

    def add_results(self, results):
        pass

    def start(self):
        self.good = 0
        self.total = 0

    def finalize(self, results, ds=False,  output_dir=None):
        results["good"] = self.good
        results["total"] = self.total
