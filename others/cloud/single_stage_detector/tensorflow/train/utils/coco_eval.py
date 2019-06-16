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
import cv2
import numpy
import json
import argparse

from utils.cocoval import cocoval 

arg_parser = argparse.ArgumentParser(description="This is a script for evalution on coco.")
arg_parser.add_argument('-cl', '--coco_labelmap_path', type=str, default='./dataset/coco_labelmap.txt', help='coco labelmap file path.')
arg_parser.add_argument('-st', '--score_threshold', type=float, default=0.005, help='socre threshold for detection results.')
arg_parser.add_argument('-gt', '--gt_file', type=str, default='./dataset/instances_val2017.json', help='groundtruth of coco dataset file.')
arg_parser.add_argument('-det', '--det_file', type=str, default='ssd_coco_det.json', help='detection result file.')
args = arg_parser.parse_args()

def get_labelmap(labelmap_path):
    lines = open(labelmap_path).readlines()
    lines = list(map(lambda x:int(x.strip()),lines))
    lines.insert(0,0)
    return lines

def eval_on_coco(det_res, args=args):
    coco_records = []
    labelmap = get_labelmap(args.coco_labelmap_path)
    for cls_key in det_res.keys():
        for ind, pred in enumerate(det_res[cls_key]):
            record = {}
            image_name = pred[0]
            score = pred[1]
            xmin = pred[2]
            ymin = pred[3]
            xmax = pred[4]
            ymax = pred[5]
            class_id = cls_key
            record['image_id'] = int(image_name.split('_')[-1])
            record['category_id'] = labelmap[class_id]
            record['score'] = score
            record['bbox'] = [xmin, ymin, xmax - xmin + 1, ymax - ymin + 1]
            if score < args.score_threshold:
              continue 
            coco_records.append(record)
    with open(args.det_file, 'w') as f_det:                       
      f_det.write(json.dumps(coco_records, cls=MyEncoder))
    return cocoval(args.det_file, args.gt_file)

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

if __name__ == '__main__':
    cocoval(args.det_file, args.gt_file)
