"""
dataset related classes and methods
"""

# pylint: disable=unused-argument,missing-docstring

import logging
import sys
import time

import numpy as np
import torch


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("dataset")


class Dataset:
    def __init__(self):
        self.items_inmemory = {}

    def preprocess(self, use_cache=True):
        raise NotImplementedError("Dataset:preprocess")

    def get_item_count(self):
        raise NotImplementedError("Dataset:get_item_count")

    def get_list(self):
        raise NotImplementedError("Dataset:get_list")

    def load_query_samples(self, sample_list):
        raise NotImplementedError("Dataset:load_query_samples")

    def unload_query_samples(self, sample_list):
        raise NotImplementedError("Dataset:unload_query_samples")

    def get_samples(self, id_list):
        raise NotImplementedError("Dataset:get_samples")

    def get_item(self, id):
        raise NotImplementedError("Dataset:get_item")


def preprocess(list_data):
    batched_pts_list, batched_gt_bboxes_list = [], []
    batched_labels_list, batched_names_list = [], []
    batched_difficulty_list = []
    batched_img_list, batched_calib_list = [], []
    batched_images = []
    for data_dict in list_data:
        pts, gt_bboxes_3d = data_dict['pts'], data_dict['gt_bboxes_3d']
        gt_labels, gt_names = data_dict['gt_labels'], data_dict['gt_names']
        difficulty = data_dict['difficulty']
        image_info, calib_info = data_dict['image_info'], data_dict['calib_info']

        batched_pts_list.append(torch.from_numpy(pts))
        batched_gt_bboxes_list.append(torch.from_numpy(gt_bboxes_3d))
        batched_labels_list.append(torch.from_numpy(gt_labels))
        batched_names_list.append(gt_names)  # List(str)
        batched_difficulty_list.append(torch.from_numpy(difficulty))
        batched_img_list.append(image_info)
        batched_calib_list.append(calib_info)
        batched_images.append(data_dict['images'])
    rt_data_dict = dict(
        batched_pts=batched_pts_list,
        batched_gt_bboxes=batched_gt_bboxes_list,
        batched_labels=batched_labels_list,
        batched_names=batched_names_list,
        batched_difficulty=batched_difficulty_list,
        batched_img_info=batched_img_list,
        batched_calib_info=batched_calib_list,
        batched_images=batched_images
    )

    return rt_data_dict
