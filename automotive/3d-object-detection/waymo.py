"""
implementation of waymo dataset
"""

# pylint: disable=unused-argument,missing-docstring

import json
import logging
import os
import time

from PIL import Image
import numpy as np
import pandas as pd
import dataset
import os
import pickle
import torch
from torchvision import transforms
import tools.process as process


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("waymo")


def read_pickle(file_path, suffix='.pkl'):
    assert os.path.splitext(file_path)[1] == suffix
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def point_range_filter(data_dict, point_range):
    '''
    data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
    point_range: [x1, y1, z1, x2, y2, z2]
    '''
    pts = data_dict['pts']
    flag_x_low = pts[:, 0] > point_range[0]
    flag_y_low = pts[:, 1] > point_range[1]
    flag_z_low = pts[:, 2] > point_range[2]
    flag_x_high = pts[:, 0] < point_range[3]
    flag_y_high = pts[:, 1] < point_range[4]
    flag_z_high = pts[:, 2] < point_range[5]
    keep_mask = flag_x_low & flag_y_low & flag_z_low & flag_x_high & flag_y_high & flag_z_high
    pts = pts[keep_mask]
    data_dict.update({'pts': pts})
    return data_dict


def read_points(file_path, dim=4):
    suffix = os.path.splitext(file_path)[1]
    assert suffix in ['.bin', '.ply', '.npy']
    if suffix == '.bin':
        return np.fromfile(file_path, dtype=np.float32).reshape(-1, dim)
    elif suffix == '.npy':
        return np.load(file_path).astype(np.float32)
    else:
        raise NotImplementedError


class Waymo(dataset.Dataset):
    CLASSES = {
        'Pedestrian': 0,
        'Cyclist': 1,
        'Car': 2
    }

    def __init__(self, data_root, split,
                 pts_prefix='velodyne_reduced', painted=True, cam_sync=False):
        super().__init__()
        assert split in ['train', 'val', 'trainval', 'test']
        self.data_root = data_root
        self.split = split
        self.pts_prefix = pts_prefix
        info_file = f'waymo_infos_val.pkl'
        self.painted = painted
        self.cam_sync = cam_sync
        self.point_range_filter = [-74.88, -74.88, -2, 74.88, 74.88, 4]
        self.data_infos = read_pickle(os.path.join(data_root, info_file))
        self.sorted_ids = range(len(self.data_infos))

    def preprocess(self, input):
        image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.485, 0.456, 0.406], std=[
                    0.229, 0.224, 0.225]),
        ])
        input_images = input['images']
        for i in range(len(input_images)):
            input['images'][i] = image_transform(
                input['images'][i]).unsqueeze(0)
        return input

    def get_list(self):
        raise NotImplementedError("Dataset:get_list")

    def load_query_samples(self, sample_list):
        # TODO: Load queries into memory, if needed
        pass

    def unload_query_samples(self, sample_list):
        # TODO: Unload queries from memory, if needed
        pass

    def get_samples(self, id_list):
        data = []
        labels = []
        for id in id_list:
            item = self.get_item(id)
            data.append({'pts': item['pts'],
                         'images': item['images'],
                         'calib_info': item['calib_info'],
                         'image_info': item['image_info']})
            labels.append({'gt_labels': item['gt_labels'],
                           'calib_info': item['calib_info'],
                           'gt_names': item['gt_names'],
                           })
        return data, labels

    def get_item(self, id):
        data_info = self.data_infos[self.sorted_ids[id]]
        image_info, calib_info, annos_info = \
            data_info['image'], data_info['calib'].copy(), data_info['annos']
        # point cloud input
        velodyne_path = data_info['point_cloud']['velodyne_path']
        pts_path = os.path.join(self.data_root, velodyne_path)
        if self.cam_sync:
            annos_info = data_info['cam_sync_annos']
        pts = read_points(pts_path, 6)
        pts = pts[:, :5]

        # calib input: for bbox coordinates transformation between Camera and Lidar.
        # because
        tr_velo_to_cam = calib_info['Tr_velo_to_cam_0'].astype(np.float32)
        r0_rect = calib_info['R0_rect'].astype(np.float32)
        for key in calib_info.keys():
            calib_info[key] = torch.from_numpy(
                calib_info[key]).type(
                torch.float32)

        # annotations input
        annos_info = self.remove_dont_care(annos_info)
        annos_name = annos_info['name']
        annos_location = annos_info['location']
        annos_dimension = annos_info['dimensions']
        rotation_y = annos_info['rotation_y']
        gt_bboxes = np.concatenate(
            [annos_location, annos_dimension, rotation_y[:, None]], axis=1).astype(np.float32)
        gt_bboxes_3d = process.bbox_camera2lidar(
            gt_bboxes, tr_velo_to_cam, r0_rect)
        gt_labels = [self.CLASSES.get(name, -1) for name in annos_name]
        data_dict = {
            'pts': torch.from_numpy(pts),
            'gt_bboxes_3d': torch.from_numpy(gt_bboxes_3d),
            'gt_labels': torch.from_numpy(np.array(gt_labels)),
            'gt_names': annos_name,
            'difficulty': annos_info['difficulty'],
            'image_info': image_info,
            'calib_info': calib_info
        }
        data_dict = point_range_filter(
            data_dict, point_range=self.point_range_filter)
        images = []
        for i in range(5):
            image = self.get_image(
                image_info['image_idx'],
                'image_' + str(i) + '/')
            images.append(image)
        data_dict['images'] = images

        return data_dict

    def remove_dont_care(self, annos_info):
        keep_ids = [
            i for i, name in enumerate(
                annos_info['name']) if name != 'DontCare']
        for k, v in annos_info.items():
            annos_info[k] = v[keep_ids]
        return annos_info

    def get_image(self, idx, camera):
        filename = os.path.join(
            self.data_root, 'training', camera + ('%s.jpg' % idx))
        input_image = Image.open(filename)
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.485, 0.456, 0.406], std=[
                    0.229, 0.224, 0.225]),
        ])

        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)
        return input_batch

    def get_item_count(self):
        return len(self.data_infos)


class PostProcessWaymo:
    def __init__(
        self,  # Postprocess parameters
    ):
        self.content_ids = []
        # TODO: Init Postprocess parameters
        self.results = []

    def add_results(self, results):
        self.results.extend(results)

    def __call__(self, results, content_id, inputs, result_dict):
        self.content_ids.extend(content_id)
        processed_results = []
        for idx in range(len(content_id)):
            processed_results.append([])
            detection_num = len(results[0][idx])
            if detection_num == 0:
                processed_results[idx].append([
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    results[6][idx]
                ])
            for detection in range(0, detection_num):
                processed_results[idx].append([
                    results[0][idx][detection][0],
                    results[0][idx][detection][1],
                    results[0][idx][detection][2],
                    results[1][idx][detection][0],
                    results[1][idx][detection][1],
                    results[1][idx][detection][2],
                    results[2][idx][detection],
                    results[3][idx][detection][0],
                    results[3][idx][detection][1],
                    results[3][idx][detection][2],
                    results[3][idx][detection][3],
                    results[4][idx][detection],
                    results[5][idx][detection],
                    results[6][idx]
                ])
        return processed_results

    def start(self):
        self.results = []

    def finalize(self, result_dict, ds=None):

        return result_dict
