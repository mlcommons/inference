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
import tensorflow as tf
import os
import json

import torch
from tqdm import tqdm

from PIL import Image
from argparse import ArgumentParser
from utils import DefaultBoxes, Encoder

from coco import COCO
from pycocotools.cocoeval import COCOeval


def parse_args():
    parser = ArgumentParser(description="SSD Resnet-34 evaluation on COCO")
    parser.add_argument('--model', '-m', type=str,
            default='./pretrained/resnet34_tf.22.1.pb',
            help='path to frozen graph')
    parser.add_argument('--data-root', '-d', type=str,
            default='coco/val2017',
            help='path to validation images')
    parser.add_argument('--image-list', '-i', type=str,
            default='./dataset_config/val2017_image_list.txt',
            help='path to image list file')
    parser.add_argument('--gt-json','-g', type=str,
            default='coco/annotations/instances_val2017.json',
            help='path to ground truth json annotations')
    parser.add_argument('--labelmap', '-l', type=str,
            default='dataset_config/coco_labelmap.txt',
            help='path to coco labelmap file')
    parser.add_argument('--score-threshold','-t', type=float,
            default=0.005, help='score threshold')
    return parser.parse_args()

def get_labelmap(labelmap_path):
    lines = open(labelmap_path).readlines()
    lines = list(map(lambda x:int(x.strip()),lines))
    lines.insert(0,0)
    return lines

def preprocess(image):
    image = image.astype('float32') / 255.
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    image = (image - mean) / std
    return image.transpose([2,0,1])

def dboxes_R34_coco():
    figsize = [1200, 1200]
    feat_size = [[50, 50], [25, 25], [13, 13], [7, 7], [3, 3], [3, 3]]
    steps=[(int(figsize[0]/fs[0]),int(figsize[1]/fs[1])) for fs in feat_size]
    # use the scales here: https://github.com/amdegroot/ssd.pytorch/blob/master/data/config.py
    scales = [(int(s*figsize[0]/300),int(s*figsize[1]/300)) for s in [21, 45, 99, 153, 207, 261, 315]]
    aspect_ratios =  [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    return dboxes

def run_inference_for_eval(graph, args):
    image_root = args.data_root
    image_list_file = args.image_list

    dboxes = dboxes_R34_coco()
    encoder = Encoder(dboxes)
    labelmap = get_labelmap(args.labelmap)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    with graph.as_default():
        with tf.Session(config=tf_config) as sess:
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            output_keys = ['ssd1200/py_location_pred','ssd1200/py_cls_pred']
            for key in output_keys:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image:0')

            with open(image_list_file, 'r') as f_image:
                image_lines = f_image.readlines()
            ret = []
            count = 0
            for image_line in tqdm(image_lines):
                count += 1
                image_name = image_line.strip()
                image_path = os.path.join(image_root, image_name + ".jpg")

                image = Image.open(image_path).convert("RGB")
                w_ori, h_ori = image.size
                image_id = int(image_name.split('_')[-1])
                image=np.array(image.resize((1200,1200), Image.BILINEAR))
                image = preprocess(image)

                output_dict = sess.run(tensor_dict,
                                       feed_dict={image_tensor: np.expand_dims(image, 0)})
                ploc = output_dict[output_keys[0]]
                plabel = output_dict[output_keys[1]]

                with torch.no_grad():
                    result = encoder.decode_batch(
                        torch.from_numpy(ploc).to(0),
                        torch.from_numpy(plabel).to(0),
                        0.50, 200, device=0)[0]
                    loc, label, prob = [r.cpu().numpy() for r in result]

                    for loc_, label_, prob_ in zip(loc, label, prob):
                        ret.append([image_id, loc_[0]*w_ori,
                                              loc_[1]*h_ori,
                                              (loc_[2] - loc_[0])*w_ori,
                                              (loc_[3] - loc_[1])*h_ori,
                                              prob_,
                                              labelmap[label_]])
    return ret



if __name__=='__main__':

    args = parse_args()
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(args.model, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    coco_records = run_inference_for_eval(detection_graph, args)

    cocoGt = COCO(args.gt_json)
    cocoDt = cocoGt.loadRes(np.array(coco_records))
    E = COCOeval(cocoGt, cocoDt, iouType='bbox')
    E.evaluate()
    E.accumulate()
    E.summarize()
