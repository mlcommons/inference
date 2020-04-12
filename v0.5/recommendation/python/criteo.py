"""
implementation of criteo dataset
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
import dlrm_data_pytorch as dp
import data_loader_terabyte


class Criteo(Dataset):

    def __init__(self, data_path, name, pre_process, use_cache, count, test_num_workers, max_ind_range=-1, sub_sample_rate=0.0, mlperf_bin_loader=False, randomize="total", memory_map=False):
        super().__init__()
        # debug prints
        # print('__init__', data_path, name, pre_process, use_cache, count, test_num_workers, max_ind_range, sub_sample_rate, randomize, memory_map)

        if name == "kaggle":
            raw_data_file = data_path + "/train.txt"
            processed_data_file = data_path + "/kaggleAdDisplayChallenge_processed.npz"
        elif name == "terabyte":
            raw_data_file = data_path + "/day"
            processed_data_file = data_path + "/terabyte_processed.npz"
        else:
            raise ValueError("only kaggle|terabyte dataset options are supported")
        self.use_mlperf_bin_loader = mlperf_bin_loader and memory_map and name == "terabyte"
        # debug prints
        # print("dataset filenames", raw_data_file, processed_data_file)

        self.test_data = dp.CriteoDataset(
            dataset=name,
            max_ind_range=max_ind_range,
            sub_sample_rate=sub_sample_rate,
            randomize=randomize,
            split="test",
            raw_path=raw_data_file,
            pro_data=processed_data_file,
            memory_map=memory_map
        )

        if self.use_mlperf_bin_loader:

            test_file = data_path + "/terabyte_processed_test.bin"
            counts_file = raw_data_file + '_fea_count.npz'

            data_loader_terabyte.numpy_to_binary(
                input_files=[raw_data_file + '_23_reordered.npz'],
                output_file_path=data_path + "/terabyte_processed_test.bin",
                split="test")

            self.test_data = data_loader_terabyte.CriteoBinDataset(
                data_file=test_file,
                counts_file=counts_file,
                batch_size=1, # FIGURE this out
                max_ind_range=max_ind_range
            )

            self.test_loader = torch.utils.data.DataLoader(
                self.test_data,
                batch_size=None,
                batch_sampler=None,
                shuffle=False,
                num_workers=0,
                collate_fn=None,
                pin_memory=False,
                drop_last=False,
            )
        else:

            self.test_loader = torch.utils.data.DataLoader(
                self.test_data,
                batch_size=1,  # FIGURE this out
                shuffle=False,
                num_workers=test_num_workers,
                collate_fn=dp.collate_wrapper_criteo,
                pin_memory=False,
                drop_last=False,
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
            self.items_in_memory[l] = self.test_data[l]

        self.last_loaded = time.time()

    ''' lg compatibilty routine '''
    def get_samples(self, id_list):

        # build list tuples as need by the batch conversion routine
        ls = []
        for i in id_list:
            ls.append(self.items_in_memory[i])

        # collate a mini-batch of samples
        if self.use_mlperf_bin_loader:
            ls_t = list(zip(*ls))

            X = torch.cat(ls_t[0])
            (num_s, len_ls) = torch.cat(ls_t[1], dim=1).size()
            lS_o = torch.stack([torch.tensor(range(len_ls)) for _ in range(num_s)])
            lS_i = torch.cat(ls_t[2], dim=1)
            T = torch.cat(ls_t[3])
        else:
            X, lS_o, lS_i, T = self.test_loader.collate_fn(ls)
        # debug prints
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
            # NOTE: copy from GPU to CPU while post processing, if needed. Alternatively,
            # we could do this on the output of predict function in backend_pytorch_native.py
            result = results[idx].detach().cpu()
            processed_results.append([result])
            # debug prints
            # print(result)
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
