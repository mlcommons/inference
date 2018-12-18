# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Blob helper functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import cv2

cv2.ocl.setUseOpenCL(False)


def im_list_to_blob(ims):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    PAD_IMAGE_STRIDE = 1
    # Pad the image so they can be divisible by a stride
    if PAD_IMAGE_STRIDE > 1:
        stride = float(PAD_IMAGE_STRIDE)
        max_shape[0] = int(np.ceil(max_shape[0] / stride) * stride)
        max_shape[1] = int(np.ceil(max_shape[1] / stride) * stride)

    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], max_shape[2]),
                    dtype=np.float32)
    for i in range(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob


def prep_im_for_blob(im, pixel_means, pixel_stds, target_sizes, max_size):
    """Mean subtract and scale an image for use in a blob."""
    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im /= pixel_stds
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    ims = []
    im_scales = []
    for target_size in target_sizes:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        ims.append(im)
        im_scales.append(im_scale)
    return ims, im_scales
