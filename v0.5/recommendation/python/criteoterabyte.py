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
        

        '''
        data_path, image_list, name, use_cache=0, image_size=None,
        image_format="NHWC", pre_process=None, count=None, cache_dir=None):
        super(Imagenet, self).__init__()

        if image_size is None:
            self.image_size = [224, 224, 3]
        else:
            self.image_size = image_size
        if not cache_dir:
            cache_dir = os.getcwd()
        self.image_list = []
        self.label_list = []
        self.count = count
        self.use_cache = use_cache
        self.cache_dir = os.path.join(cache_dir, "preprocessed", name, image_format)
        self.data_path = data_path
        self.pre_process = pre_process
        # input images are in HWC
        self.need_transpose = True if image_format == "NCHW" else False

        not_found = 0
        if image_list is None:
            # by default look for val_map.txt
            image_list = os.path.join(data_path, "val_map.txt")

        os.makedirs(self.cache_dir, exist_ok=True)

        start = time.time()unload_query_samples
        with open(image_list, 'r') as f:
            for s in f:
                image_name, label = re.split(r"\s+", s.strip())
                src = os.path.join(data_path, image_name)
                if not os.path.exists(src):
                    # if the image does not exists ignore it
                    not_found += 1
                    continue
                os.makedirs(os.path.dirname(os.path.join(self.cache_dir, image_name)), exist_ok=True)
                dst = os.path.join(self.cache_dir, image_name)
                if not os.path.exists(dst + ".npy"):
                    # cache a preprocessed version of the image
                    # TODO: make this multi threaded ?
                    img_org = cv2.imread(src)
                    processed = self.pre_process(img_org, need_transpose=self.need_transpose, dims=self.image_size)
                    np.save(dst, processed)
                
                self.image_list.append(image_name)
                self.label_list.append(int(label))

                # limit the dataset if requested
                if self.count and len(self.image_list) >= self.count:
                    break

        time_taken = time.time() - start
        if not self.image_list:
            log.error("no images in image list found")
            raise ValueError("no images in image list found")
        if not_found > 0:
            log.info("reduced image list, %d images not found", not_found)

        log.info("loaded {} images, cache={}, took={:.1f}sec".format(
            len(self.image_list), use_cache, time_taken))

        self.label_list = np.array(self.label_list)
        '''

    def get_item_count(self):
        return len(self.test_data.X_int)
    
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

        #print('get_samples', (X, lS_o, lS_i, T))
        return (X, lS_o, lS_i, T)

