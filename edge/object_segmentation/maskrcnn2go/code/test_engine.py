from __future__ import absolute_import, division, print_function, unicode_literals


import numpy as np


def empty_results(num_classes, num_images):
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    ret = {"all_boxes": [[[] for _ in range(num_images)] for _ in range(num_classes)]}
    ret["all_segms"] = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    return ret


def extend_results(index, all_res, im_res):
    for j in range(1, len(im_res)):
        all_res[j][index] = im_res[j]


def extend_results_with_classes(index, all_boxes, box_ids):
    boxes, classids = box_ids
    for j, classid in enumerate(classids):
        classid = int(classid)
        assert classid <= len(
            all_boxes
        ), "{} classid out of range!" "class id: {}, boxes: {}".format(
            j, classid, boxes
        )
        if type(all_boxes[classid][index]) is np.ndarray:
            all_boxes[classid][index] = np.vstack((all_boxes[classid][index], boxes[j]))
        else:
            all_boxes[classid][index] = np.array([boxes[j]])


def extend_seg_results_with_classes(index, all_segms, segs_ids):
    im_masks_rle, classids = segs_ids
    for j, classid in enumerate(classids):
        classid = int(classid)
        assert classid <= len(all_segms), (
            "{} classid out of range!"
            "class id: {}, segms: {}".format(j, classid, im_masks_rle)
        )
        all_segms[classid][index].append(im_masks_rle[j])
