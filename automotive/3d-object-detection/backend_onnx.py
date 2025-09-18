from typing import Optional, List, Union
import os
import torch
import logging
import backend
from collections import namedtuple
from model.painter import Painter
from model.pointpillars_core import PointPillarsPre, PointPillarsPos
import numpy as np
from tools.process import keep_bbox_from_image_range
from waymo import Waymo
import onnxruntime as ort


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("backend-onnx")


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


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


class BackendOnnx(backend.Backend):
    def __init__(
        self,
        segmentor_path,
        lidar_detector_path,
        data_path
    ):
        super(BackendOnnx, self).__init__()
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
        device = torch.device("cpu")
        PaintArgs = namedtuple(
            'PaintArgs', [
                'training_path', 'model_path', 'cam_sync'])
        painting_args = PaintArgs(
            os.path.join(
                self.data_root,
                'training'),
            self.segmentor_path,
            False)
        self.painter = Painter(painting_args, onnx=True)
        self.segmentor = self.painter.model
        model_pre = PointPillarsPre()
        model_post = PointPillarsPos(self.detection_classes)
        model_pre.eval()
        model_post.eval()
        ort_sess = ort.InferenceSession(self.lidar_detector_path)
        self.lidar_detector = ort_sess
        self.model_pre = model_pre
        self.model_post = model_post
        return self

    def predict(self, inputs):
        dimensions, locations, rotation_y, box2d, class_labels, class_scores, ids = [
        ], [], [], [], [], [], []
        with torch.inference_mode():
            model_input = inputs[0]
            batched_pts = model_input['pts']
            scores_from_cam = []
            for i in range(len(model_input['images'])):
                input_image_name = self.segmentor.get_inputs()[0].name
                input_data = {
                    input_image_name: to_numpy(
                        model_input['images'][i])}
                segmentation_score = self.segmentor.run(None, input_data)
                segmentation_score = [
                    torch.from_numpy(item) for item in segmentation_score]
                scores_from_cam.append(
                    self.painter.get_score(
                        segmentation_score[0].squeeze(0)).cpu())
            points = self.painter.augment_lidar_class_scores_both(
                scores_from_cam, batched_pts, model_input['calib_info'])
            pillars, coors_batch, npoints_per_pillar = self.model_pre(batched_pts=[
                                                                      points])
            input_pillars_name = self.lidar_detector.get_inputs()[0].name
            input_coors_batch_name = self.lidar_detector.get_inputs()[1].name
            input_npoints_per_pillar_name = self.lidar_detector.get_inputs()[
                2].name
            input_data = {input_pillars_name: to_numpy(pillars),
                          input_coors_batch_name: to_numpy(coors_batch),
                          input_npoints_per_pillar_name: to_numpy(npoints_per_pillar)}
            result = self.lidar_detector.run(None, input_data)
            result = [torch.from_numpy(item) for item in result]
            batch_results = self.model_post(result)
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
