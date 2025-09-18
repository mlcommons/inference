from typing import Optional, List, Union
import os
import torch
import logging
import backend
from collections import namedtuple
from model.painter import Painter
from model.pointpillars import PointPillars
import numpy as np
from tools.process import keep_bbox_from_image_range
from waymo import Waymo


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("backend-pytorch")


def change_calib_device(calib, cuda):
    result = {}
    if cuda:
        device = 'cuda'
    else:
        device = 'cpu'
    result['R0_rect'] = calib['R0_rect'].to(device=device, dtype=torch.float)
    for i in range(5):
        result['P' + str(i)] = calib['P' + str(i)
                                     ].to(device=device, dtype=torch.float)
        result['Tr_velo_to_cam_' +
               str(i)] = calib['Tr_velo_to_cam_' +
                               str(i)].to(device=device, dtype=torch.float)
    return result


class BackendDeploy(backend.Backend):
    def __init__(
        self,
        segmentor_path,
        lidar_detector_path,
        data_path
    ):
        super(BackendDeploy, self).__init__()
        self.segmentor_path = segmentor_path
        self.lidar_detector_path = lidar_detector_path
        # self.segmentation_classes = 18
        self.detection_classes = 3
        self.data_root = data_path
        CLASSES = Waymo.CLASSES
        self.LABEL2CLASSES = {v: k for k, v in CLASSES.items()}

    def version(self):
        return torch.__version__

    def name(self):
        return "python-SUT"

    def load(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        PaintArgs = namedtuple(
            'PaintArgs', [
                'training_path', 'model_path', 'cam_sync'])
        painting_args = PaintArgs(
            os.path.join(
                self.data_root,
                'training'),
            self.segmentor_path,
            False)
        self.painter = Painter(painting_args)
        self.segmentor = self.painter.model
        model = PointPillars(
            nclasses=self.detection_classes,
            painted=True).to(
            device=device)
        model.eval()
        checkpoint = torch.load(
            self.lidar_detector_path, map_location=torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'))
        model.load_state_dict(checkpoint["model_state_dict"])
        self.lidar_detector = model

        return self

    def predict(self, inputs):
        dimensions, locations, rotation_y, box2d, class_labels, class_scores, ids = [
        ], [], [], [], [], [], []
        with torch.inference_mode():
            device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
            model_input = inputs[0]
            batched_pts = model_input['pts']
            scores_from_cam = []
            for i in range(len(model_input['images'])):
                segmentation_score = self.segmentor(
                    model_input['images'][i].to(device))[0]
                scores_from_cam.append(
                    self.painter.get_score(segmentation_score).cpu())
            points = self.painter.augment_lidar_class_scores_both(
                scores_from_cam, batched_pts, model_input['calib_info'])
            batch_results = self.lidar_detector(
                batched_pts=[points.to(device=device)], mode='val')
            for j, result in enumerate(batch_results):
                format_result = {
                    'class': [],
                    'truncated': [],
                    'occluded': [],
                    'alpha': [],
                    'bbox': [],
                    'dimensions': [],
                    'location': [],
                    'rotation_y': [],
                    'score': [],
                    'idx': -1
                }

                calib_info = model_input['calib_info']
                image_info = model_input['image_info']
                idx = model_input['image_info']['image_idx']
                format_result['idx'] = idx
                calib_info = change_calib_device(calib_info, False)
                result_filter = keep_bbox_from_image_range(
                    result, calib_info, 5, image_info, False)

                lidar_bboxes = result_filter['lidar_bboxes']
                labels, scores = result_filter['labels'], result_filter['scores']
                bboxes2d, camera_bboxes = result_filter['bboxes2d'], result_filter['camera_bboxes']
                for lidar_bbox, label, score, bbox2d, camera_bbox in \
                        zip(lidar_bboxes, labels, scores, bboxes2d, camera_bboxes):
                    format_result['class'].append(label.item())
                    format_result['truncated'].append(0.0)
                    format_result['occluded'].append(0)
                    alpha = camera_bbox[6] - \
                        np.arctan2(camera_bbox[0], camera_bbox[2])
                    format_result['alpha'].append(alpha.item())
                    format_result['bbox'].append(bbox2d.tolist())
                    format_result['dimensions'].append(camera_bbox[3:6])
                    format_result['location'].append(camera_bbox[:3])
                    format_result['rotation_y'].append(camera_bbox[6].item())
                    format_result['score'].append(score.item())

                if len(format_result['dimensions']) > 0:
                    format_result['dimensions'] = torch.stack(
                        format_result['dimensions'])
                    format_result['location'] = torch.stack(
                        format_result['location'])
                dimensions.append(format_result['dimensions'])
                locations.append(format_result['location'])
                rotation_y.append(format_result['rotation_y'])
                class_labels.append(format_result['class'])
                class_scores.append(format_result['score'])
                box2d.append(format_result['bbox'])
                ids.append(format_result['idx'])
        return dimensions, locations, rotation_y, box2d, class_labels, class_scores, ids
