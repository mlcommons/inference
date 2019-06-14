from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)

from caffe2.python import workspace
import blob_utils
import box_utils
import pycocotools.mask as mask_util
import numpy as np
import logging
import sys
import copy


FORMAT = '%(levelname)s %(filename)s:%(lineno)4d: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def create_input_blobs(net_def):
    for op in net_def.op:
        for blob_in in op.input:
            if not workspace.HasBlob(blob_in):
                workspace.CreateBlob(blob_in)


def prepare_blobs(
    im,
    pixel_means,
    pixel_stds,
    target_size,
    max_size,
):
    return prepare_image_batch_to_blob(
        ims=[im],
        pixel_means=pixel_means,
        pixel_stds=pixel_stds,
        target_size=target_size,
        max_size=max_size
    )


def prepare_image_batch_to_blob(
    ims,
    pixel_means,
    pixel_stds,
    target_size, max_size,
):
    assert isinstance(ims, list)
    blobs = []
    im_infos = []
    if pixel_means is None:
        pixel_means = np.array([[[0.0, 0.0, 0.0]]])
    if pixel_stds is None:
        pixel_stds = np.array([[[1.0, 1.0, 1.0]]])
    for img in ims:
        blob, scale = blob_utils.prep_im_for_blob(
            copy.deepcopy(img), pixel_means, pixel_stds, [target_size], max_size)
        blobs.append(blob[0])
        im_infos.append(np.array(
            [[blob[0].shape[0], blob[0].shape[1], scale[0]]],
            dtype=np.float32
        ))

    ret = {}
    ret['data'] = blob_utils.im_list_to_blob(blobs)
    ret['im_info'] = np.vstack(im_infos)
    return ret


def compute_segm_results(
    masks, ref_boxes, classids, im_h, im_w,
    thresh_binarize=0.5, rle_encode=True
):
    ''' masks: (#boxes, #classes, mask_dim, mask_dim)
        ref_boxes: (#boxes, 5), where each row is [x1, y1, x2, y2, cls]
        classids: (#boxes, )
        ret: list of im_masks, [im_mask, ...] or [im_mask_rle, ...]
    '''
    assert len(masks.shape) == 4
    assert masks.shape[2] == masks.shape[3]
    assert masks.shape[0] == ref_boxes.shape[0]
    assert ref_boxes.shape[1] == 4
    assert len(classids) == masks.shape[0]

    all_segms = []
    # To work around an issue with cv2.resize (it seems to automatically pad
    # with repeated border values), we manually zero-pad the masks by 1 pixel
    # prior to resizing back to the original image resolution. This prevents
    # "top hat" artifacts. We therefore need to expand the reference boxes by an
    # appropriate factor.
    M = masks.shape[2]
    scale = (M + 2.0) / M
    ref_boxes = box_utils.expand_boxes(ref_boxes, scale)
    ref_boxes = ref_boxes.astype(np.int32)
    padded_mask = np.zeros((M + 2, M + 2), dtype=np.float32)

    for mask_ind in range(masks.shape[0]):
        cur_cls = int(classids[mask_ind])
        padded_mask[1:-1, 1:-1] = masks[mask_ind, cur_cls, :, :]

        ref_box = ref_boxes[mask_ind, :]
        w = ref_box[2] - ref_box[0] + 1
        h = ref_box[3] - ref_box[1] + 1
        w = np.maximum(w, 1)
        h = np.maximum(h, 1)

        mask = cv2.resize(padded_mask, (w, h))
        mask = np.array(mask > thresh_binarize, dtype=np.uint8)
        im_mask = np.zeros((im_h, im_w), dtype=np.uint8)

        x_0 = max(ref_box[0], 0)
        x_1 = min(ref_box[2] + 1, im_w)
        y_0 = max(ref_box[1], 0)
        y_1 = min(ref_box[3] + 1, im_h)

        im_mask[y_0:y_1, x_0:x_1] = mask[
            (y_0 - ref_box[1]):(y_1 - ref_box[1]),
            (x_0 - ref_box[0]):(x_1 - ref_box[0])]

        ret = im_mask
        if rle_encode:
            # Get RLE encoding used by the COCO evaluation API
            rle = mask_util.encode(
                np.array(im_mask[:, :, np.newaxis], order='F'))[0]
            ret = rle

        all_segms.append(ret)

    return all_segms
