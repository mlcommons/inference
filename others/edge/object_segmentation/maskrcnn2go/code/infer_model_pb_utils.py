from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import sys

import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import numpy as np
import utils2
from caffe2.python import core, workspace


FORMAT = "%(levelname)s %(filename)s:%(lineno)4d: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


PIXEL_MEANS_DEFAULT = np.array([[[0.0, 0.0, 0.0]]])
PIXEL_STDS_DEFAULT = np.array([[[1.0, 1.0, 1.0]]])


def run_single_segms(
    net,
    image,
    target_size,
    pixel_means=PIXEL_MEANS_DEFAULT,
    pixel_stds=PIXEL_STDS_DEFAULT,
    rle_encode=True,
    max_size=1333,
):
    inputs = utils2.prepare_blobs(
        image,
        target_size=target_size,
        max_size=max_size,
        pixel_means=pixel_means,
        pixel_stds=pixel_stds,
    )
    for k, v in inputs.items():
        workspace.FeedBlob(core.ScopedName(k), v)

    try:
        workspace.RunNet(net.Proto().name)
        classids = workspace.FetchBlob("class_nms")
        scores = workspace.FetchBlob("score_nms")  # bbox scores, (R, )
        boxes = workspace.FetchBlob("bbox_nms")  # i.e., boxes, (R, 4*1)
        masks = workspace.FetchBlob("mask_fcn_probs")  # (R, cls, mask_dim, mask_dim)
    except Exception as e:
        print(e)
        # may not detect anything at all
        R = 0
        scores = np.zeros((R,), dtype=np.float32)
        boxes = np.zeros((R, 4), dtype=np.float32)
        classids = np.zeros((R,), dtype=np.float32)
        masks = np.zeros((R, 1, 1, 1), dtype=np.float32)

    # included in the model
    # scale = inputs["im_info"][0][2]
    # boxes /= scale

    R = boxes.shape[0]
    im_masks = []
    if R > 0:
        im_dims = image.shape
        im_masks = utils2.compute_segm_results(
            masks, boxes, classids, im_dims[0], im_dims[1], rle_encode=rle_encode
        )

    boxes = np.column_stack((boxes, scores))

    ret = {"classids": classids, "boxes": boxes, "masks": masks, "im_masks": im_masks}
    return ret
