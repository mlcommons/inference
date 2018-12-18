# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np


def xyxy_to_xywh(xyxy):
    """Convert [x1 y1 x2 y2] box format to [x1 y1 w h] format."""
    if isinstance(xyxy, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(xyxy) == 4
        x1, y1 = xyxy[0], xyxy[1]
        w = xyxy[2] - x1 + 1
        h = xyxy[3] - y1 + 1
        return (x1, y1, w, h)
    elif isinstance(xyxy, np.ndarray):
        # Multiple boxes given as a 2D ndarray
        return np.hstack((xyxy[:, 0:2], xyxy[:, 2:4] - xyxy[:, 0:2] + 1))
    else:
        raise TypeError("Argument xyxy must be a list, tuple, or numpy array.")


def expand_boxes(boxes, scale):
    """Expand an array of boxes by a given scale."""
    box_dim = boxes.shape[1]
    if box_dim == 4:
        w_half = (boxes[:, 2] - boxes[:, 0]) * 0.5
        h_half = (boxes[:, 3] - boxes[:, 1]) * 0.5
        x_c = (boxes[:, 2] + boxes[:, 0]) * 0.5
        y_c = (boxes[:, 3] + boxes[:, 1]) * 0.5

        w_half *= scale
        h_half *= scale

        boxes_exp = np.zeros(boxes.shape)
        boxes_exp[:, 0] = x_c - w_half
        boxes_exp[:, 2] = x_c + w_half
        boxes_exp[:, 1] = y_c - h_half
        boxes_exp[:, 3] = y_c + h_half
    elif box_dim == 5:
        boxes_exp = boxes.copy()
        boxes_exp[:, 2:4] *= scale
    else:
        raise Exception("Unsupported box dimension: {}".format(box_dim))

    return boxes_exp
