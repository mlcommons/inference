# Copyright (c) 2019, Xilinx, Inc.
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import sys
import tensorflow as tf

from distutils.version import StrictVersion
import cv2
import numpy
import json
import math
import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def parse_args():
    arg_parser = argparse.ArgumentParser(description="This is a evaluation script for ssd-resnet34 on coco2017.")
    arg_parser.add_argument('-pb', '--model_pb', type=str, default='./pretrained/ssd_resnet34_cloud_native_tf.pb', help='tensroflow pb model path.')
    arg_parser.add_argument('-cl', '--coco_labelmap_path', type=str, default='./dataset_config/coco_labelmap.txt', help='coco labelmap file path.')
    arg_parser.add_argument('-bs', '--batch_size', type=int, default=4)
    arg_parser.add_argument('-st', '--score_threshold', type=float, default=0.05, help='socre threshold for detection results.')
    arg_parser.add_argument('-g', '--gpu', type=str, choices= ['0', '1', '2', '3'], default='0')
    arg_parser.add_argument('-ir', '--image_root', type=str, default='coco/val2017/', help='image root')
    arg_parser.add_argument('-il', '--image_list_file', type=str, default='./dataset_config/val2017_image_list.txt', help='image list for evalution.')
    arg_parser.add_argument('-gt', '--gt_file', type=str, default='coco/annotations/instances_val2017.json', help='groundtruth of coco dataset file.')
    arg_parser.add_argument('-det', '--det_file', type=str, default='ssd_coco_det.json', help='detection result file.')
    return arg_parser.parse_args()

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

def get_labelmap(labelmap_path):
    lines = open(labelmap_path).readlines()
    lines = list(map(lambda x:int(x.strip()),lines))
    lines.insert(0,0)
    return lines

def load_pb_graph(pb_path):
  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(pb_path, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')
  return detection_graph

def load_batch(niter, image_lines, image_root, batch_size):
  if (niter - 1) * batch_size >= len(image_lines):
     raise Exception("%d bacth exceed the number of images!"%niter)
  image_batch = []
  image_name_batch = []
  image_shape_batch = []
  image_index = (niter - 1) * batch_size
  while image_index < len(image_lines) and image_index < niter * batch_size:
    image_name = image_lines[image_index].strip() 
    image_path = os.path.join(image_root, image_name + ".jpg")
    image = cv2.imread(image_path)
    height, width = image.shape[0:2]
    image = cv2.resize(image, (1200, 1200))
    image = image[:, :, ::-1]
    image = image - [_R_MEAN, _G_MEAN, _B_MEAN]
    image_batch.append(image)
    image_name_batch.append(image_name)
    image_shape_batch.append([height, width])
    image_index += 1
  return np.stack(image_batch, axis=0), image_name_batch, image_shape_batch    

def cocoval(detected_json, eval_json):
    eval_gt = COCO(eval_json)
    eval_dt = eval_gt.loadRes(detected_json)
    cocoEval = COCOeval(eval_gt, eval_dt, iouType='bbox')

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

def run_inference_for_eval(graph, args):
  with graph.as_default():
    with tf.Session() as sess:
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in ['detection_bboxes', 'detection_scores',
          'detection_classes']:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image:0')
      with open(args.image_list_file, 'r') as f_image:
        image_lines = f_image.readlines()
      coco_records = []
      count = 0
      labelmap = get_labelmap(args.coco_labelmap_path)
      num_iter = math.ceil(len(image_lines) / args.batch_size)
      for ind in range(num_iter):
        image_batch, image_name_batch, image_shape_batch = load_batch(ind + 1, image_lines, args.image_root, args.batch_size)
        count += image_batch.shape[0] 
        print("process: %d images"%count)
        output_dict = sess.run(tensor_dict,
                               feed_dict={image_tensor: image_batch})
        cur_batch_size = image_batch.shape[0] 
        for bind in range(cur_batch_size):
          num_detections = int(output_dict['detection_bboxes'][bind].shape[0])
          for ind_bb in range(num_detections):
            record = {}
            height, width = image_shape_batch[bind]
            ymin = output_dict['detection_bboxes'][bind][ind_bb][0] * height
            xmin = output_dict['detection_bboxes'][bind][ind_bb][1] * width
            ymax = output_dict['detection_bboxes'][bind][ind_bb][2] * height
            xmax = output_dict['detection_bboxes'][bind][ind_bb][3] * width
            score = output_dict['detection_scores'][bind][ind_bb]
            class_id = int(output_dict['detection_classes'][bind][ind_bb])
            record['image_id'] = int(image_name_batch[bind].split('_')[-1])
            record['category_id'] = labelmap[class_id]
            record['score'] = score
            #record['bbox'] = [xmin, ymin, xmax - xmin + 1, ymax - ymin + 1]
            record['bbox'] = [xmin, ymin, xmax - xmin, ymax - ymin]
            if score < args.score_threshold:
              break 
            coco_records.append(record)
  return coco_records

class MyEncoder(json.JSONEncoder):
   def default(self, obj):
     if isinstance(obj, (numpy.int_, numpy.intc, numpy.intp, numpy.int8,
       numpy.int16, numpy.int32, numpy.int64, numpy.uint8,
       numpy.uint16,numpy.uint32, numpy.uint64)):
       return int(obj)
     elif isinstance(obj, (numpy.float_, numpy.float16, numpy.float32, 
       numpy.float64)):
       return float(obj)
     elif isinstance(obj, (numpy.ndarray,)): 
       return obj.tolist() 
     return json.JSONEncoder.default(self, obj)

if __name__ == "__main__":
  args = parse_args()
  os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
  detection_graph = load_pb_graph(args.model_pb)  
  coco_records = run_inference_for_eval(detection_graph, args)
  with open(args.det_file, 'w') as f_det:
    f_det.write(json.dumps(coco_records, cls=MyEncoder))
  cocoval(args.det_file, args.gt_file)
