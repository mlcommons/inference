"""
implementation of imagenet dataset
"""

# pylint: disable=unused-argument,missing-docstring

import logging
import os
import re
import time

import numpy as np
import inspect
# pytorch
import torch
from torch.utils.data import Dataset, RandomSampler

#import dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("imagenet")



# dlrm
import sys
sys.path.append('/root/mnaumov/github/dlrm')

#import data_loader_terabyte as dltb
import dlrm_data_pytorch as dp

class CriteoTerabyte(Dataset):

    def __init__(self, data_path, image_list, name, image_format, pre_process, use_cache, count, max_ind_range, sub_sample_rate, randomize, memory_map=False):
        # debug print
        print('CriteoTerabyte __init__', data_path, image_list, name, image_format, pre_process, use_cache, count, max_ind_range, sub_sample_rate, randomize, memory_map)
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
            num_workers=0, #FIGURE THIS OUT args.test_num_workers,
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

