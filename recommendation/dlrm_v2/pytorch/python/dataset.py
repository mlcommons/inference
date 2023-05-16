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

class Item():
    def __init__(self, label, img, idx):
        self.label = label
        self.img = img
        self.idx = idx
        self.start = time.time()


def usleep(sec):
    if sys.platform == 'win32':
        # on windows time.sleep() doesn't work to well
        import ctypes
        kernel32 = ctypes.windll.kernel32
        timer = kernel32.CreateWaitableTimerA(ctypes.c_void_p(), True, ctypes.c_void_p())
        delay = ctypes.c_longlong(int(-1 * (10 * 1000000 * sec)))
        kernel32.SetWaitableTimer(timer, ctypes.byref(delay), 0, ctypes.c_void_p(), ctypes.c_void_p(), False)
        kernel32.WaitForSingleObject(timer, 0xffffffff)
    else:
        time.sleep(sec)


class Dataset():
    def __init__(self):
        self.arrival = None
        self.image_list = []
        self.label_list = []
        self.image_list_inmemory = {}
        self.last_loaded = -1

    def preprocess(self, use_cache=True):
        raise NotImplementedError("Dataset:preprocess")

    def get_item_count(self):
        return len(self.image_list)

    def get_list(self):
        raise NotImplementedError("Dataset:get_list")

    def load_query_samples(self, sample_list):
        raise NotImplementedError("Dataset:load_query_samples")

    def unload_query_samples(self, sample_list):
        raise NotImplementedError("Dataset:unload_query_samples")

    def get_samples(self, id_list):
        raise NotImplementedError("Dataset:get_samples")

    def get_item_loc(self, id):
        raise NotImplementedError("Dataset:get_item_loc")


#
# Post processing
#
class PostProcessCommon:
    def __init__(self, offset=0):
        self.offset = offset
        self.good = 0
        self.total = 0

    def __call__(self, results, ids, expected=None, result_dict=None):
        processed_results = []
        n = len(results[0])
        for idx in range(0, n):
            result = results[0][idx] + self.offset
            processed_results.append([result])
            if result == expected[idx]:
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


class PostProcessArgMax:
    def __init__(self, offset=0):
        self.offset = offset
        self.good = 0
        self.total = 0

    def __call__(self, results, ids, expected=None, result_dict=None):
        processed_results = []
        results = np.argmax(results[0], axis=1)
        n = results.shape[0]
        for idx in range(0, n):
            result = results[idx] + self.offset
            processed_results.append([result])
            if result == expected[idx]:
                self.good += 1
        self.total += n
        return processed_results

    def add_results(self, results):
        pass

    def start(self):
        self.good = 0
        self.total = 0

    def finalize(self, results, ds=False, output_dir=None):
        results["good"] = self.good
        results["total"] = self.total

