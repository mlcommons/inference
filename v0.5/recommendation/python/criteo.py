"""
implementation of criteo dataset
"""

# pylint: disable=unused-argument,missing-docstring

import logging
import os
import sys
import re
import time
import random

import numpy as np
import sklearn.metrics
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

    def __init__(self, data_path, name, pre_process, use_cache, 
                 count=None, 
                 samples_to_aggregate=None, 
                 min_samples_to_aggregate=None, 
                 max_samples_to_aggregate=None, 
                 test_num_workers=0, 
                 max_ind_range=-1, 
                 sub_sample_rate=0.0, 
                 mlperf_bin_loader=False, 
                 randomize="total", 
                 memory_map=False):
        super().__init__()

        self.count = count
        self.samples_to_aggregate = 1 if samples_to_aggregate is None else samples_to_aggregate
        self.min_samples_to_aggregate = None if min_samples_to_aggregate is None else min_samples_to_aggregate
        self.max_samples_to_aggregate = None if max_samples_to_aggregate is None else max_samples_to_aggregate

        self.random_offsets = []
        
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
        self.num_individual_samples = len(self.test_data)

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
                batch_size=self.samples_to_aggregate,
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
                batch_size=self.samples_to_aggregate,
                shuffle=False,
                num_workers=test_num_workers,
                collate_fn=dp.collate_wrapper_criteo,
                pin_memory=False,
                drop_last=False,
            )

    def get_item_count(self):
        # get number of items in the dataset

        # WARNING: Note that the orignal dataset returns number of samples, while the
        # binary dataset returns the number of batches. Therefore, when using a mini-batch
        # of size samples_to_aggregate as an item we need to adjust the original dataset item_count.
        # On the other hand, data loader always returns number of batches.
        if self.use_mlperf_bin_loader:
            self.num_aggregated_samples = len(self.test_data)
            # self.num_aggregated_samples2 = len(self.test_loader)
        else:

            if self.min_samples_to_aggregate is None and self.max_samples_to_aggregate is None:
                self.num_aggregated_samples = (self.num_individual_samples + self.samples_to_aggregate - 1) // self.samples_to_aggregate
            else:
                # generate random offsets for variable query sizes
                done = False
                qo = 0
                self.random_offsets.append(int(0))
                while done == False:
                
                    qs = random.randint(self.min_samples_to_aggregate,self.max_samples_to_aggregate)
                    if qo+qs < self.num_individual_samples:
                        self.random_offsets.append(int(qo+qs))
                        qo = qo + qs
                    else:
                        done = True
                
                self.num_aggregated_samples = len(self.random_offsets)

            # self.num_aggregated_samples2 = len(self.test_loader)

        # limit number of items to count if needed
        if self.count is not None:
            self.num_aggregated_samples = min(self.count, self.num_aggregated_samples)


        return self.num_aggregated_samples

    ''' lg compatibilty routine '''
    def unload_query_samples(self, sample_list):
        self.items_in_memory = {}

    ''' lg compatibilty routine '''
    def load_query_samples(self, sample_list):

        self.items_in_memory = {}

        # WARNING: notice that while DataLoader is iterable-style, the Dataset
        # can be iterable- or map-style, and Criteo[Bin]Dataset are the latter
        # This means that we can not index into DataLoader, but can enumerate it,
        # while we can index into the dataset itself.
        for l in sample_list:
            # approach 1: single sample as an item
            '''
            self.items_in_memory[l] = self.test_data[l]
            '''
            if self.min_samples_to_aggregate is None and self.max_samples_to_aggregate is None:
                # approach 2: multiple samples as an item
                s = l * self.samples_to_aggregate
                e = min((l + 1) * self.samples_to_aggregate, self.num_individual_samples)
            else:
                s = self.random_offsets[l]
                
                if l < len(self.random_offsets)-1:
                    e = self.random_offsets[l+1]
                else:
                    e = self.num_individual_samples
            
            ls = [self.test_data[i] for i in range(s, e)]
            if self.use_mlperf_bin_loader:
                # NOTE: in binary dataset the values are transformed
                ls_t = list(zip(*ls))
                X = torch.cat(ls_t[0])
                (num_s, len_ls) = torch.cat(ls_t[1], dim=1).size()
                lS_o = torch.stack([torch.tensor(range(len_ls)) for _ in range(num_s)])
                lS_i = torch.cat(ls_t[2], dim=1)
                T = torch.cat(ls_t[3])
                self.items_in_memory[l] = (X, lS_o, lS_i, T)
            else:
                # NOTE: in original dataset the values are not transformed
                # and collate besides stacking them also transforms them
                self.items_in_memory[l] = self.test_loader.collate_fn(ls)

        self.last_loaded = time.time()

    ''' lg compatibilty routine '''
    def get_samples(self, id_list):

        # build list tuples as need by the batch conversion routine
        ls = []
        for i in id_list:
            ls.append(self.items_in_memory[i])

        # approach 1: collate a mini-batch of single samples
        '''
        if self.use_mlperf_bin_loader:
            # NOTE: in binary dataset the values are transformed
            ls_t = list(zip(*ls))

            X = torch.cat(ls_t[0])
            (num_s, len_ls) = torch.cat(ls_t[1], dim=1).size()
            lS_o = torch.stack([torch.tensor(range(len_ls)) for _ in range(num_s)])
            lS_i = torch.cat(ls_t[2], dim=1)
            T = torch.cat(ls_t[3])
        else:
            # NOTE: in original dataset the values are not transformed and collate besides stacking transforms them
            X, lS_o, lS_i, T = self.test_loader.collate_fn(ls)
        '''
        # approach 2: collate a mini-batch of multiple samples
        # NOTE: recall that the samples have already been transformed for both datasets
        # (by earlier calls in load_query_samples), therefore we just need to stack them
        ls_t = list(zip(*ls))

        X = torch.cat(ls_t[0])
        (num_s, len_ls) = torch.cat(ls_t[1], dim=1).size()
        lS_o = torch.stack([torch.tensor(range(len_ls)) for _ in range(num_s)])
        lS_i = torch.cat(ls_t[2], dim=1)
        T = torch.cat(ls_t[3])
        # debug prints
        # print('get_samples', (X, lS_o, lS_i, T))
        # print('get_samples', X.shape)
        return (X, lS_o, lS_i, T)


# Pre  processing
def pre_process_criteo_dlrm(x):
    return x


# Post processing
class DlrmPostProcess:
    def __init__(self):
        self.good = 0
        self.total = 0
        self.roc_auc = 0
        self.results = []

    def __call__(self, results, expected=None, result_dict=None):
        processed_results = []
        n = len(results)
        for idx in range(0, n):
            # NOTE: copy from GPU to CPU while post processing, if needed. Alternatively,
            # we could do this on the output of predict function in backend_pytorch_native.py
            result = results[idx].detach().cpu()
            target = expected[idx]
            processed_results.append([result, target])
            # debug prints
            # print(result)
            # print(expected[idx])

            # accuracy metric
            if result.round() == target:
                self.good += 1
        self.total += n
        return processed_results

    def add_results(self, results):
        self.results = self.results + results

    def start(self):
        self.good = 0
        self.total = 0
        self.roc_auc = 0
        self.results = []

    def finalize(self, result_dict, ds=False,  output_dir=None):
        # AUC metric
        results, targets = zip(*self.results)
        results = np.concatenate(results, axis=0)
        targets = np.concatenate(targets, axis=0)
        self.roc_auc = sklearn.metrics.roc_auc_score(targets, results)

        result_dict["good"] = self.good
        result_dict["total"] = self.total
        result_dict["roc_auc"] = self.roc_auc
