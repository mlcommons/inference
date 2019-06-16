# Copyright 2018 Changan Wang
# Modifications Copyright (C) 2019 Xilinx
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile
import argparse

from eval_ssd_large import *
arg_parser = argparse.ArgumentParser(description="This is a script for export graph.")
arg_parser.add_argument('-og', '--output_graph', type=str, default="./ssd_resnet34_large.pbtxt", help='exported graph file.')
global_anchor_info = {}

def ssd_model_fn(features, labels, mode, params):
    filename = features['filename']
    shape = features['shape']
    loc_targets = features['loc_targets']
    cls_targets = features['cls_targets']
    match_scores = features['match_scores']
    features = features['image']
    features = tf.unstack(features, axis=-1, name='split_rgb')
    features = tf.stack([features[2], features[1], features[0]], axis=-1, name='merge_bgr')
    global global_anchor_info
    decode_fn = global_anchor_info['decode_fn']
    num_anchors_per_layer = global_anchor_info['num_anchors_per_layer']
    all_num_anchors_depth = global_anchor_info['all_num_anchors_depth']
    with tf.variable_scope(params['model_scope'], default_name=None, values=[features], reuse=tf.AUTO_REUSE):
        backbone = ssd_net_resnet34_large.Resnet34Backbone(params['data_format'])
        feature_layers = backbone.forward(features, training=(mode == tf.estimator.ModeKeys.TRAIN))
        location_pred, cls_pred = ssd_net_resnet34_large.multibox_head(feature_layers, params['num_classes'], all_num_anchors_depth, data_format=params['data_format'], strides=(3, 3))
        if params['data_format'] == 'channels_first':
            cls_pred = [tf.transpose(pred, [0, 2, 3, 1]) for pred in cls_pred]
            location_pred = [tf.transpose(pred, [0, 2, 3, 1]) for pred in location_pred]
        cls_pred = [tf.reshape(pred, [tf.shape(features)[0], -1, params['num_classes']]) for pred in cls_pred]
        location_pred = [tf.reshape(pred, [tf.shape(features)[0], -1, 4]) for pred in location_pred]
        cls_pred = tf.concat(cls_pred, axis=1)
        location_pred = tf.concat(location_pred, axis=1)

    with tf.device('/cpu:0'):
        bboxes_pred = tf.map_fn(lambda _preds : decode_fn(_preds),
                              tf.reshape(location_pred, [tf.shape(features)[0], -1, 4]),
                              dtype=[tf.float32] * len(num_anchors_per_layer), back_prop=False)
        bboxes_pred = tf.concat(bboxes_pred, axis=1)
        print("bboxes_pred:", bboxes_pred)
        print("cls_pred:", cls_pred)
        parse_bboxes_fn = lambda x: parse_by_class_fixed_bboxes(x[0], x[1], params)
        pred_results = tf.map_fn(parse_bboxes_fn, (cls_pred, bboxes_pred), dtype=(tf.float32, tf.float32, tf.float32), back_prop=False)     
 
    predictions = {'filename': filename, 'shape': shape }
    detection_bboxes = tf.concat(pred_results[0], axis=0)
    detection_scores = tf.concat(pred_results[1], axis=0)
    detection_classes = tf.concat(pred_results[2], axis=0)
    predictions['detection_classes'] = detection_classes
    predictions['detection_scores'] = detection_scores
    predictions['detection_bboxes'] = detection_bboxes
    tf.identity(detection_bboxes, name='detection_bboxes')
    tf.identity(detection_scores, name='detection_scores')
    tf.identity(detection_classes, name='detection_classes')
    tf.identity(tf.shape(features)[0], name='eval_images_per_bacth')  
    tf.summary.scalar('eval_images', FLAGS.batch_size_mine)
    summary_hook = tf.train.SummarySaverHook(save_steps=params['save_summary_steps'],
                                             output_dir=params['summary_dir'],
                                             summary_op=tf.summary.merge_all())

def export_graph(args):
    with tf.Graph().as_default() as graph:
        out_shape = [FLAGS.train_image_size, FLAGS.train_image_size]
        anchor_creator = anchor_manipulator.AnchorCreator(out_shape,
                                                    layers_shapes = [(50, 50), (25, 25), (13, 13), (7, 7), (3, 3), (1, 1)],
                                                    anchor_scales = [(0.1,), (0.2,), (0.375,), (0.55,), (0.725,), (0.9,)],
                                                    extra_anchor_scales = [(0.1414,), (0.2739,), (0.4541,), (0.6315,), (0.8078,), (0.9836,)],
                                                    anchor_ratios = [(1., 2., .5), (1., 2., 3., .5, 0.3333), (1., 2., 3., .5, 0.3333), (1., 2., 3., .5, 0.3333), (1., 2., .5), (1., 2., .5)],
                                                    layer_steps = [24, 48, 92, 171, 400, 1200])
        all_anchors, all_num_anchors_depth, all_num_anchors_spatial = anchor_creator.get_all_anchors()
        num_anchors_per_layer = []
        for ind in range(len(all_anchors)):
            num_anchors_per_layer.append(all_num_anchors_depth[ind] * all_num_anchors_spatial[ind])
        anchor_encoder_decoder = anchor_manipulator.AnchorEncoder(allowed_borders = [1.0] * 6,
                                                                  positive_threshold = FLAGS.match_threshold,
                                                                  ignore_threshold = FLAGS.neg_threshold,
                                                                  prior_scaling=[0.1, 0.1, 0.2, 0.2])
        global global_anchor_info
        global_anchor_info = {'decode_fn': lambda pred : anchor_encoder_decoder.decode_all_anchors(pred, num_anchors_per_layer),
                              'num_anchors_per_layer': num_anchors_per_layer,
                              'all_num_anchors_depth': all_num_anchors_depth}
        anchor_encoder_fn = lambda glabels_, gbboxes_: anchor_encoder_decoder.encode_all_anchors(glabels_, gbboxes_, all_anchors, all_num_anchors_depth, all_num_anchors_spatial)
        glabels = [1]
        glabels = tf.cast(glabels, tf.int64)
        gbboxes = [[10., 10., 200., 200.]] 
        gt_targets, gt_labels, gt_scores = anchor_encoder_fn(glabels, gbboxes)
        image = tf.placeholder(name='image', dtype=tf.float32, shape=[None, FLAGS.train_image_size, FLAGS.train_image_size, 3])
        filename = tf.placeholder(name='filename', dtype=tf.string, shape=[None,])
        shape = tf.placeholder(name='shape', dtype=tf.int32, shape=[None, 3])
        input_ = {'image': image, 'filename': filename, 'shape': shape, 'loc_targets': [gt_targets], 'cls_targets': [gt_labels], 'match_scores': [gt_scores]}
        ssd_model_fn(input_, None, tf.estimator.ModeKeys.PREDICT, { 'select_threshold': FLAGS.select_threshold,
                                                                    'min_size': FLAGS.min_size,
                                                                    'nms_threshold': FLAGS.nms_threshold,
                                                                    'nms_topk': FLAGS.nms_topk,
                                                                    'keep_topk': FLAGS.keep_topk,
                                                                    'data_format': FLAGS.data_format,
                                                                    'batch_size': FLAGS.batch_size_mine,
                                                                    'model_scope': FLAGS.model_scope,
                                                                    'save_summary_steps': FLAGS.save_summary_steps,
                                                                    'summary_dir': None,
                                                                    'num_classes': FLAGS.num_classes,
                                                                    'negative_ratio': FLAGS.negative_ratio,
                                                                    'match_threshold': FLAGS.match_threshold,
                                                                    'neg_threshold': FLAGS.neg_threshold,
                                                                    'weight_decay': FLAGS.weight_decay})
 
        graph_def = graph.as_graph_def()
        with gfile.GFile(args.output_graph, 'w') as f:
            f.write(text_format.MessageToString(graph_def))
        print("Finish export inference graph")

def main(_):
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    args = arg_parser.parse_args() 
    export_graph(args)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
