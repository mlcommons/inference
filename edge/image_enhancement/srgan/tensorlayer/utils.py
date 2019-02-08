# Copyright 2018 The MLPerf Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

# This code is modified from https://github.com/tensorlayer/srgan/blob/master/utils.py

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.prepro import *
import scipy


def get_imgs_fn(file_name, path):
    """ Input an image path and name, return an image array """
    return scipy.misc.imread(path + file_name, mode='RGB')


def crop_sub_imgs_fn(x, is_random=True):
    x = crop(x, wrg=384, hrg=384, is_random=is_random)
    x = x / (255. / 2.)
    x = x - 1.
    return x


def downsample_fn(x):
    # We obtained the LR images by downsampling the HR images using
    #  bicubic kernel with downsampling factor r = 4.
    x = imresize(x, size=[96, 96], interp='bicubic', mode=None)
    x = x / (255. / 2.)
    x = x - 1.
    return x


def pre(x):
    x = x / (255. / 2.)
    x = x - 1.
    return x
