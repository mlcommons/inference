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

    def preprocess(self, use_cache=True):
        raise NotImplementedError("Dataset:preprocess")

    def get_item_count(self):
        return len(self.image_list)

    def get_list(self):
        raise NotImplementedError("Dataset:get_list")

    def clear_trace(self):
        self.arrival = None

    def generate_linear_trace(self, min_queries, min_duration, qps, seed=123):
        timestamp = 0
        arrival = []
        qps = int(qps)
        while timestamp < min_duration and len(arrival) < min_queries:
            timestamp += 1 / qps
            arrival.append(1 / qps)
        self.arrival = arrival

    def generate_exp_trace(self, min_queries, min_duration, qps, seed=123):
        timestamp = 0
        arrival = []
        qps = int(qps)
        np.random.seed(seed)
        dist = np.random.exponential(scale=1, size=qps)
        while timestamp < min_duration and len(arrival) < min_queries:
            val = dist[len(arrival) % qps]
            timestamp += val / qps
            arrival.append(val / qps)
        self.arrival = arrival

    def batch(self, batch_size=1):
        arrival = self.arrival
        timestamp = time.time()
        for idx in range(0, len(self.image_list), batch_size):
            y = self.label_list[idx:idx + batch_size]
            if self.use_cache:
                x = self.image_list[idx:idx + batch_size]
            else:
                x = []
                for i in range(0, y.shape[0]):
                    img, _ = self.get_item(idx + i)
                    x.append(img)
                x = np.array(x)
            if arrival:
                # timestamp += arrival[idx % len(arrival)]
                if idx > len(arrival) - 1:
                    break
                timestamp += arrival[idx]
                now = time.time()
                diff = timestamp - now
                if diff > 0:
                    usleep(diff)
            yield Item(y, x, idx)


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

def pre_process_mobilenet(img, dims=None, need_transpose=False):
    if dims is None:
        dims = [244, 244, 3]
    img = img.resize((dims[0], dims[1]))
    img = np.asarray(img, dtype='float32')
    if len(img.shape) != 3:
        img = np.stack([img] * 3, axis=2)
    input_mean = 0.
    input_std = 255.
    img = (img - input_mean) / input_std
    if need_transpose:
        img = img.transpose([2, 0, 1])
    return img


def pre_process_vgg(img, dims=None, need_transpose=False):
    if dims is None:
        dims = [244, 244, 3]

    width, height = img.size
    output_height, output_width, _ = dims

    # scale down to 87.5% keeping the aspect ratio
    new_height = int(100. * output_height / 87.5)
    new_width = int(100. * output_width / 87.5)
    if height > width:
        w = new_width
        h = int(output_height * width / new_width)
    else:
        h = new_height
        w = int(output_width *  height / new_height)

    img = img.resize((w, h))

    # center crop to output_width, output_height
    left = (w - output_width) / 2
    right = (w + output_width) / 2
    top = (h - output_height) / 2
    bottom = (h + output_height) / 2

    img = img.crop((left, top, right, bottom))

    # normalize image
    img = np.asarray(img, dtype='float32')
    if len(img.shape) != 3:
        img = np.stack([img] * 3, axis=2)

    assert list(img.shape) == dims

    means = np.array([123.68, 116.78, 103.94], dtype=np.float32)
    img -= means

    # transpose if needed
    if need_transpose:
        img = img.transpose([2, 0, 1])
    return img
