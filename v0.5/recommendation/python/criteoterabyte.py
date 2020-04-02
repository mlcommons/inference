"""
implementation of imagenet dataset
"""

# pylint: disable=unused-argument,missing-docstring

import logging
import os
import re
import time

import numpy as np

# pytorch
import torch
from torch.utils.data import Dataset, RandomSampler

import dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("imagenet")



# dlrm
import sys
sys.path.append('/root/mnaumov/github/dlrm')

import data_loader_terabyte
import dlrm_data_pytorch as dp

class CriteoTerabyte(dataset.Dataset):

    def __init__(self, data_path, image_list, name, image_format, pre_process, use_cache, count, max_ind_range, sub_sample_rate, randomize, memory_map=False):
        # debug print
        print(data_path, image_list, name, image_format, pre_process, use_cache, count, max_ind_range, sub_sample_rate, randomize, memory_map)
        

        if True:
            dataset_name = "kaggle"
            #raw_data_file = data_path + "/train.txt"
            raw_data_file = data_path + "/train_tiny2.txt"
            processed_data_file = data_path + "/kaggleAdDisplayChallenge_processed.npz"
        else:
            dataset_name = "terabyte"
            raw_data_file = data_path + "/day"
            processed_data_file = data_path + "/terabyte_processed.npz"
        
        self.test_data = dp.CriteoDataset(
            dataset_name,
            max_ind_range,
            sub_sample_rate,
            randomize,
            "test",
            raw_data_file,
            processed_data_file,
            memory_map
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_data,
            batch_size=1, #FIGURE THIS OUT args.test_mini_batch_size,
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

        start = time.time()
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
        
    def get_item(self, nr):
        """Get image by number in the list."""
        X, lS_o, lS_i, T = self.test_loader[nr]
        return (X, lS_o, lS_i, T)
        '''
        dst = os.path.join(self.cache_dir, self.image_list[nr])
        img = np.load(dst + ".npy")
        return img, self.label_list[nr]
        '''

    '''
    def get_item_loc(self, nr):
        src = os.path.join(self.data_path, self.image_list[nr])
        return src
    '''
