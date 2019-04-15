"""
dataset related classes and methods
"""

# pylint: disable=unused-argument,missing-docstring

import sys
import time

import numpy as np


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

    def preprocess(self, use_cache=True):
        raise NotImplementedError("Dataset:preprocess")

    def get_item_count(self):
        return len(self.image_list)

    def get_list(self):
        raise NotImplementedError("Dataset:get_list")

    def clear_trace(self):
        self.arrival = None

    def load_query_samples(self, sample_list):
        self.image_list_inmemory = {}
        for sample in sample_list:
            self.image_list_inmemory[sample], _ = self.get_item(sample)

    def unload_query_samples(self, sample_list):
        self.image_list_inmemory = {}

    def get_samples(self, id_list):
        data = np.array([self.image_list_inmemory[id] for id in id_list])
        return data, self.label_list[id_list]

#
# Post processing
#

def post_process_none(results):
    return results


def post_process_argmax(results):
    return np.argmax(results)


def post_process_offset1(results):
    return results - 1


def post_process_argmax_offset(results):
    return np.argmax(results) - 1


#
# pre-processing
#

def center_crop(img, out_height, out_width):
    width, height = img.size
    left = (width - out_width) / 2
    right = (width + out_width) / 2
    top = (height - out_height) / 2
    bottom = (height + out_height) / 2
    img = img.crop((left, top, right, bottom))
    return img


def resize_with_aspectratio(img, out_height, out_width, scale=87.5):
    width, height = img.size
    new_height = int(100. * out_height / scale)
    new_width = int(100. * out_width / scale)
    if height > width:
        w = new_width
        h = int(out_height * width / new_width)
    else:
        h = new_height
        w = int(out_width * height / new_height)
    img = img.resize((w, h))
    return img


def pre_process_vgg(img, dims=None, need_transpose=False):
    output_height, output_width, _ = dims

    img = resize_with_aspectratio(img, output_height, output_width)
    img = center_crop(img, output_height, output_width)
    img = np.asarray(img, dtype='float32')
    if len(img.shape) != 3:
        img = np.stack([img] * 3, axis=2)

    # normalize image
    means = np.array([123.68, 116.78, 103.94], dtype=np.float32)
    img -= means

    # transpose if needed
    if need_transpose:
        img = img.transpose([2, 0, 1])
    return img


def pre_process_mobilenet(img, dims=None, need_transpose=False):
    output_height, output_width, _ = dims

    img = resize_with_aspectratio(img, output_height, output_width)
    img = center_crop(img, output_height, output_width)
    img = np.asarray(img, dtype='float32')
    if len(img.shape) != 3:
        img = np.stack([img] * 3, axis=2)

    img = img / 255.0
    img = img - 0.5
    img = img * 2

    # transpose if needed
    if need_transpose:
        img = img.transpose([2, 0, 1])
    return img
