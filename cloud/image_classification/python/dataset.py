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

    def generate_linear_trace(self, min_queries, min_duration, qps):
        """ Generates inter-arrival times for queries with a uniform distribution.
        It should satisfy both min_duration and min_queries

        Args:
            min_queries: Int, minimal number of queries in the trace.
            min_duration: Int, minimal time duration in seconds for the entire trace.
            qps: Int, expected queries per sec for the uniform distribution.

        Returns:
            None
        """
        timestamp = 0
        arrival = []
        timestep = 1 / qps
        while timestamp < min_duration and len(arrival) < min_queries:
            timestamp += timestep
            arrival.append(timestep)
        self.arrival = arrival

    def generate_exp_trace(self, min_queries, min_duration, qps, seed=123):
        """ Generates inter-arrival times for queries with a poisson distribution.
        It should satisfy both min_duration and min_queries

        Args:
            min_queries: Int, minimal number of queries in the trace.
            min_duration: Int, minimal time duration in seconds for the entire trace.
            qps: Int, expected queries per sec for the possion distribution.

        Returns:
            None
        """
        timestamp = 0
        arrival = []
        num_samples = int(qps)
        if num_samples == 0:
            num_samples = 1
        np.random.seed(seed)
        samples = np.random.exponential(scale=1.0, size=num_samples)
        while timestamp < min_duration and len(arrival) < min_queries:
            idx = len(arrival)
            val = samples[idx % num_samples]
            # accumulative so we know when to stop
            timestamp += val / num_samples
            # for processing only store the delta
            arrival.append(val / num_samples)
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
