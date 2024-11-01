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
        device='cpu'
    result['R0_rect'] = calib['R0_rect'].to(device=device, dtype=torch.float)
    for i in range(5):
        result['P' + str(i)] = calib['P' + str(i)].to(device=device, dtype=torch.float)
        result['Tr_velo_to_cam_' + str(i)] = calib['Tr_velo_to_cam_' + str(i)].to(device=device, dtype=torch.float)
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
        #self.segmentation_classes = 18
        self.detection_classes = 3
        self.data_root = data_path
        CLASSES = Waymo.CLASSES
        self.LABEL2CLASSES = {v:k for k, v in CLASSES.items()}

    def version(self):
        return torch.__version__

    def name(self):
        return "python-SUT"


    def load(self):
        PaintArgs = namedtuple('PaintArgs', ['training_path', 'model_path', 'cam_sync'])
        painting_args = PaintArgs(os.path.join(self.data_root, 'training'), self.segmentor_path, False)
        self.painter = Painter(painting_args)
        self.segmentor = self.painter.model
        model = PointPillars(nclasses=self.detection_classes, painted=True)
        checkpoint = torch.load(self.lidar_detector_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        self.lidar_detector = model
        return self

    

    def predict(self, inputs):
        # TODO: implement predict
        Boxes, Classes, Scores = [], [], []
        with torch.inference_mode():
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            format_results = {}
            model_input = inputs[0]
            batched_pts = model_input['pts']
            #batched_images = data_dict['batched_images'][0]
            scores_from_cam = []
            for i in range(len(model_input['images'])):
                segmentation_score = self.segmentor(model_input['images'][i].to(device))[0]
                scores_from_cam.append(self.painter.get_score(segmentation_score).cpu())
            points = self.painter.augment_lidar_class_scores_both(scores_from_cam, batched_pts, model_input['calib_info'])
            batch_results = self.lidar_detector(batched_pts=[points], 
                                    mode='val')
            for j, result in enumerate(batch_results):
                    format_result = {
                        'name': [],
                        'truncated': [],
                        'occluded': [],
                        'alpha': [],
                        'bbox': [],
                        'dimensions': [],
                        'location': [],
                        'rotation_y': [],
                        'score': []
                    }
                    
                    calib_info = model_input['calib_info']
                    image_info = model_input['image_info']
                    idx = model_input['image_info']['image_idx']
                    
                    calib_info = change_calib_device(calib_info, False)
                    result_filter = keep_bbox_from_image_range(result, calib_info, 5, image_info, False)
                    #result_filter = keep_bbox_from_lidar_range(result_filter, pcd_limit_range)
                    
                    lidar_bboxes = result_filter['lidar_bboxes']
                    labels, scores = result_filter['labels'], result_filter['scores']
                    bboxes2d, camera_bboxes = result_filter['bboxes2d'], result_filter['camera_bboxes']
                    for lidar_bbox, label, score, bbox2d, camera_bbox in \
                        zip(lidar_bboxes, labels, scores, bboxes2d, camera_bboxes):
                        format_result['name'].append(self.LABEL2CLASSES[label.item()])
                        format_result['truncated'].append(0.0)
                        format_result['occluded'].append(0)
                        alpha = camera_bbox[6] - np.arctan2(camera_bbox[0], camera_bbox[2])
                        format_result['alpha'].append(alpha.item())
                        format_result['bbox'].append(bbox2d.tolist())
                        format_result['dimensions'].append(camera_bbox[3:6])
                        format_result['location'].append(camera_bbox[:3])
                        format_result['rotation_y'].append(camera_bbox[6].item())
                        format_result['score'].append(score.item())
                    
                    #write_label(format_result, os.path.join(saved_submit_path, f'{idx:06d}.txt'))

                    if len(format_result['dimensions']) > 0:
                        format_result['dimensions'] = torch.stack(format_result['dimensions'])
                        format_result['location'] = torch.stack(format_result['location'])
                    format_results[idx] = {k:np.array(v) for k, v in format_result.items()}
            #return Boxes, Classes, Scores # Change to desired output
        return format_results
        

