"""
implementation of imagenet dataset
"""

# pylint: disable=unused-argument,missing-docstring

import logging
import os
import re
import time

import cv2
import numpy as np

import dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("imagenet")


class Imagenet(dataset.Dataset):

    def __init__(self, data_path, image_list, name, use_cache=0, image_size=None,
            image_format="NHWC", pre_process=None, count=None, cache_dir=None, preprocessed_dir=None, threads=os.cpu_count()):
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
        self.data_path = data_path
        self.pre_process = pre_process # if None we assume data_path is having preprocessed dataset
        self.use_cache = use_cache

        if preprocessed_dir:
            self.cache_dir = preprocessed_dir
        elif pre_process:
            self.cache_dir = os.path.join(cache_dir, "preprocessed", name, image_format)
        else:
            self.cache_dir = cache_dir

        # input images are in HWC
        self.need_transpose = True if image_format == "NCHW" else False

        self.not_found = 0
        if image_list is None:
            # by default look for val_map.txt
            image_list = os.path.join(data_path, "val_map.txt")
        with open(image_list, 'r') as fp:
            for count, line in enumerate(fp):
                pass
        count = count + 1
        if not self.count:
            CNT = count
        else:
            CNT = count if count <= self.count else self.count

        os.makedirs(self.cache_dir, exist_ok=True)

        start = time.time()
        N = threads
        import concurrent.futures
        if N > CNT:
            N = CNT

        if not pre_process:
            log.info("Loading {} preprocessed images using {} threads".format(CNT, N))
        else:
            log.info("Preprocessing {} images using {} threads".format(CNT, N))

        with open(image_list, 'r') as f:
            lists = []
            image_lists = []
            label_lists = []
            for i in range(N):
                lists.append([ next(f) for x in range(int(CNT/N)) ])
                image_lists.append([])
                label_lists.append([])
            if int(CNT%N) > 0:
                lists.append([ next(f) for x in range(int(CNT%N)) ])
                image_lists.append([])
                label_lists.append([])
        executor = concurrent.futures.ThreadPoolExecutor(N)
        futures = [executor.submit(self.process, data_path, item, image_lists[lists.index(item)],
            label_lists[lists.index(item)]) for item in lists]
        concurrent.futures.wait(futures)
        for i in range (len(image_lists)):
            self.image_list += image_lists[i]
            self.label_list += label_lists[i]
        time_taken = time.time() - start
        if not self.image_list:
            log.error("no images in image list found")
            raise ValueError("no images in image list found")
        if self.not_found > 0:
            log.info("reduced image list, %d images not found", self.not_found)

        log.info("loaded {} images, cache={}, already_preprocessed={}, took={:.1f}sec".format(
            len(self.image_list), use_cache, pre_process is None, time_taken))
        self.label_list = np.array(self.label_list)

    def process(self, data_path, files, image_list, label_list):
        for s in files:
            image_name, label = re.split(r"\s+", s.strip())
            src = os.path.join(data_path, image_name)
            if not self.pre_process:
                if not os.path.exists(os.path.join(data_path, image_name) + ".npy"):
                    # if the image does not exists ignore it
                    self.not_found += 1
                    continue
            else:
                if not os.path.exists(src):
                    # if the image does not exists ignore it
                    self.not_found += 1
                    continue
                os.makedirs(os.path.dirname(os.path.join(self.cache_dir, image_name)), exist_ok=True)
                dst = os.path.join(self.cache_dir, image_name)
                if not os.path.exists(dst + ".npy"):
                    # cache a preprocessed version of the image
                    img_org = cv2.imread(src)
                    processed = self.pre_process(img_org, need_transpose=self.need_transpose, dims=self.image_size)
                    np.save(dst, processed)
            image_list.append(image_name)
            label_list.append(int(label))

            # limit the dataset if requested
            if self.count and len(self.image_list) >= self.count:
                break

    def get_item(self, nr):
        """Get image by number in the list."""
        dst = os.path.join(self.cache_dir, self.image_list[nr])
        img = np.load(dst + ".npy")
        return img, self.label_list[nr]

    def get_item_loc(self, nr):
        src = os.path.join(self.data_path, self.image_list[nr])
        return src

