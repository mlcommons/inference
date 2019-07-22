# Copyright 2018 Changan Wang

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
import math

import tensorflow as tf
import numpy as np
from tensorflow.contrib.image.python.ops import image_ops

def areas(gt_bboxes):
    with tf.name_scope('bboxes_areas', [gt_bboxes]):
        ymin, xmin, ymax, xmax = tf.split(gt_bboxes, 4, axis=1)
        return (xmax - xmin) * (ymax - ymin)

def intersection(gt_bboxes, default_bboxes):
    with tf.name_scope('bboxes_intersection', [gt_bboxes, default_bboxes]):
        ymin, xmin, ymax, xmax = tf.split(gt_bboxes, 4, axis=1)
        gt_ymin, gt_xmin, gt_ymax, gt_xmax = [tf.transpose(b, perm=[1, 0]) for b in tf.split(default_bboxes, 4, axis=1)]
        int_ymin = tf.maximum(ymin, gt_ymin)
        int_xmin = tf.maximum(xmin, gt_xmin)
        int_ymax = tf.minimum(ymax, gt_ymax)
        int_xmax = tf.minimum(xmax, gt_xmax)
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        return h * w

def iou_matrix(gt_bboxes, default_bboxes):
    with tf.name_scope('iou_matrix', [gt_bboxes, default_bboxes]):
        inter_vol = intersection(gt_bboxes, default_bboxes)
        union_vol = areas(gt_bboxes) + tf.transpose(areas(default_bboxes), perm=[1, 0]) - inter_vol
        return tf.where(tf.equal(union_vol, 0.0),
                        tf.zeros_like(inter_vol), tf.truediv(inter_vol, union_vol))

def do_dual_max_match(overlap_matrix, low_thres, high_thres, ignore_between=True, gt_max_first=True):
    '''
    overlap_matrix: num_gt * num_anchors
    '''
    with tf.name_scope('dual_max_match', [overlap_matrix]):
        anchors_to_gt = tf.argmax(overlap_matrix, axis=0)
        match_values = tf.reduce_max(overlap_matrix, axis=0)
        #positive_mask = tf.greater(match_values, high_thres)
        less_mask = tf.less(match_values, low_thres)
        between_mask = tf.logical_and(tf.less(match_values, high_thres), tf.greater_equal(match_values, low_thres))
        negative_mask = less_mask if ignore_between else between_mask
        ignore_mask = between_mask if ignore_between else less_mask
        match_indices = tf.where(negative_mask, -1 * tf.ones_like(anchors_to_gt), anchors_to_gt)
        match_indices = tf.where(ignore_mask, -2 * tf.ones_like(match_indices), match_indices)
        # negtive values has no effect in tf.one_hot, that means all zeros along that axis
        # so all positive match positions in anchors_to_gt_mask is 1, all others are 0
        anchors_to_gt_mask = tf.one_hot(tf.clip_by_value(match_indices, -1, tf.cast(tf.shape(overlap_matrix)[0], tf.int64)),
                                        tf.shape(overlap_matrix)[0], on_value=1, off_value=0, axis=0, dtype=tf.int32)
        gt_to_anchors = tf.argmax(overlap_matrix, axis=1)
        if gt_max_first:
            # the max match from ground truth's side has higher priority
            left_gt_to_anchors_mask = tf.one_hot(gt_to_anchors, tf.shape(overlap_matrix)[1], on_value=1, off_value=0, axis=1, dtype=tf.int32)
        else:
            # the max match from anchors' side has higher priority
            # use match result from ground truth's side only when the the matching degree from anchors' side is lower than position threshold
            left_gt_to_anchors_mask = tf.cast(tf.logical_and(tf.reduce_max(anchors_to_gt_mask, axis=1, keep_dims=True) < 1,
                                                            tf.one_hot(gt_to_anchors, tf.shape(overlap_matrix)[1],
                                                                        on_value=True, off_value=False, axis=1, dtype=tf.bool)
                                                            ), tf.int64)
        # can not use left_gt_to_anchors_mask here, because there are many ground truthes match to one anchor, we should pick the highest one even when we are merging matching from ground truth side
        left_gt_to_anchors_scores = overlap_matrix * tf.to_float(left_gt_to_anchors_mask)
        # merge matching results from ground truth's side with the original matching results from anchors' side
        # then select all the overlap score of those matching pairs
        selected_scores = tf.gather_nd(overlap_matrix,  tf.stack([tf.where(tf.reduce_max(left_gt_to_anchors_mask, axis=0) > 0,
                                                                            tf.argmax(left_gt_to_anchors_scores, axis=0),
                                                                            anchors_to_gt),
                                                                    tf.range(tf.cast(tf.shape(overlap_matrix)[1], tf.int64))], axis=1))
        return tf.where(tf.reduce_max(left_gt_to_anchors_mask, axis=0) > 0,
                        tf.argmax(left_gt_to_anchors_scores, axis=0),
                        match_indices), selected_scores


class AnchorEncoder(object):
    def __init__(self, allowed_borders, positive_threshold, ignore_threshold, prior_scaling, clip=False):
        super(AnchorEncoder, self).__init__()
        self._all_anchors = None
        self._allowed_borders = allowed_borders
        self._positive_threshold = positive_threshold
        self._ignore_threshold = ignore_threshold
        self._prior_scaling = prior_scaling
        self._clip = clip

    def center2point(self, center_y, center_x, height, width):
        return center_y - height / 2., center_x - width / 2., center_y + height / 2., center_x + width / 2.,

    def point2center(self, ymin, xmin, ymax, xmax):
        height, width = (ymax - ymin), (xmax - xmin)
        return ymin + height / 2., xmin + width / 2., height, width

    def encode_all_anchors(self, labels, bboxes, all_anchors, all_num_anchors_depth, all_num_anchors_spatial):
        # y, x, h, w are all in range [0, 1] relative to the original image size
        # shape info:
        # y_on_image, x_on_image: layers_shapes[0] * layers_shapes[1]
        # h_on_image, w_on_image: num_anchors
        assert (len(all_num_anchors_depth)==len(all_num_anchors_spatial)) and (len(all_num_anchors_depth)==len(all_anchors)), 'inconsist num layers for anchors.'
        with tf.name_scope('encode_all_anchors'):
            num_layers = len(all_num_anchors_depth)
            list_anchors_ymin = []
            list_anchors_xmin = []
            list_anchors_ymax = []
            list_anchors_xmax = []
            tiled_allowed_borders = []
            for ind, anchor in enumerate(all_anchors):
                anchors_ymin_, anchors_xmin_, anchors_ymax_, anchors_xmax_ = self.center2point(anchor[0], anchor[1], anchor[2], anchor[3])
                anchors_ymin_ = tf.transpose(anchors_ymin_, perm=[2, 0, 1])
                anchors_xmin_ = tf.transpose(anchors_xmin_, perm=[2, 0, 1])
                anchors_ymax_ = tf.transpose(anchors_ymax_, perm=[2, 0, 1])
                anchors_xmax_ = tf.transpose(anchors_xmax_, perm=[2, 0, 1])
                list_anchors_ymin.append(tf.reshape(anchors_ymin_, [-1]))
                list_anchors_xmin.append(tf.reshape(anchors_xmin_, [-1]))
                list_anchors_ymax.append(tf.reshape(anchors_ymax_, [-1]))
                list_anchors_xmax.append(tf.reshape(anchors_xmax_, [-1]))

                tiled_allowed_borders.extend([self._allowed_borders[ind]] * all_num_anchors_depth[ind] * all_num_anchors_spatial[ind])

            anchors_ymin = tf.concat(list_anchors_ymin, 0, name='concat_ymin')
            anchors_xmin = tf.concat(list_anchors_xmin, 0, name='concat_xmin')
            anchors_ymax = tf.concat(list_anchors_ymax, 0, name='concat_ymax')
            anchors_xmax = tf.concat(list_anchors_xmax, 0, name='concat_xmax')

            if self._clip:
                anchors_ymin = tf.clip_by_value(anchors_ymin, 0., 1.)
                anchors_xmin = tf.clip_by_value(anchors_xmin, 0., 1.)
                anchors_ymax = tf.clip_by_value(anchors_ymax, 0., 1.)
                anchors_xmax = tf.clip_by_value(anchors_xmax, 0., 1.)

            anchor_allowed_borders = tf.stack(tiled_allowed_borders, 0, name='concat_allowed_borders')

            inside_mask = tf.logical_and(tf.logical_and(anchors_ymin > -anchor_allowed_borders * 1.,
                                                        anchors_xmin > -anchor_allowed_borders * 1.),
                                        tf.logical_and(anchors_ymax < (1. + anchor_allowed_borders * 1.),
                                                        anchors_xmax < (1. + anchor_allowed_borders * 1.)))

            anchors_point = tf.stack([anchors_ymin, anchors_xmin, anchors_ymax, anchors_xmax], axis=-1)

            # save_anchors_op = tf.py_func(save_anchors,
            #                 [bboxes,
            #                 labels,
            #                 anchors_point],
            #                 tf.int64, stateful=True)

            # with tf.control_dependencies([save_anchors_op]):
            overlap_matrix = iou_matrix(bboxes, anchors_point) * tf.cast(tf.expand_dims(inside_mask, 0), tf.float32)
            matched_gt, gt_scores = do_dual_max_match(overlap_matrix, self._ignore_threshold, self._positive_threshold)
            #print("bboxes shape: ", bboxes.shape)
            #print("labels shape: ", labels.shape)
            #print("matched_gt shape: ", matched_gt.shape)
            #print("gt_scores shape: ", gt_scores.shape)
            # get all positive matching positions
            matched_gt_mask = matched_gt > -1
            matched_indices = tf.clip_by_value(matched_gt, 0, tf.int64.max)
            # the labels here maybe chaos at those non-positive positions
            gt_labels = tf.gather(labels, matched_indices)
            #print("gt_labels shape: ", gt_labels.shape)
            #gt_labels_arm = tf.gather(tf.ones_like(labels), matched_indices)
            # filter the invalid labels
            gt_labels = gt_labels * tf.cast(matched_gt_mask, tf.int64)
            #gt_labels_arm = gt_labels_arm * tf.cast(matched_gt_mask, tf.int64)
            # set those ignored positions to -1
            gt_labels = gt_labels + (-1 * tf.cast(matched_gt < -1, tf.int64))

            gt_ymin, gt_xmin, gt_ymax, gt_xmax = tf.unstack(tf.gather(bboxes, matched_indices), 4, axis=-1)

            # transform to center / size.
            gt_cy, gt_cx, gt_h, gt_w = self.point2center(gt_ymin, gt_xmin, gt_ymax, gt_xmax)
            anchor_cy, anchor_cx, anchor_h, anchor_w = self.point2center(anchors_ymin, anchors_xmin, anchors_ymax, anchors_xmax)
            # encode features.
            # the prior_scaling (in fact is 5 and 10) is use for balance the regression loss of center and with(or height)
            gt_cy = (gt_cy - anchor_cy) / anchor_h / self._prior_scaling[1]
            gt_cx = (gt_cx - anchor_cx) / anchor_w / self._prior_scaling[0]
            gt_h = tf.log(gt_h / anchor_h) / self._prior_scaling[2]
            gt_w = tf.log(gt_w / anchor_w) / self._prior_scaling[2]
            # now gt_localizations is our regression object, but also maybe chaos at those non-positive positions
            gt_targets = tf.stack([gt_cx, gt_cy, gt_w, gt_h], axis=-1)
            # set all targets of non-positive positions to 0
            gt_targets = tf.expand_dims(tf.cast(matched_gt_mask, tf.float32), -1) * gt_targets
            self._all_anchors = (anchor_cx, anchor_cy, anchor_w, anchor_h)
            return gt_targets, gt_labels, gt_scores

    # return a list, of which each is:
    #   shape: [feature_h, feature_w, num_anchors, 4]
    #   order: ymin, xmin, ymax, xmax
    def decode_all_anchors(self, pred_location, num_anchors_per_layer):
        assert self._all_anchors is not None, 'no anchors to decode.'
        with tf.name_scope('decode_all_anchors', [pred_location]):
            anchor_cx, anchor_cy, anchor_w, anchor_h = self._all_anchors
            pred_h = tf.exp(pred_location[:, -1] * self._prior_scaling[3]) * anchor_h
            pred_w = tf.exp(pred_location[:, -2] * self._prior_scaling[2]) * anchor_w
            pred_cy = pred_location[:, 1] * self._prior_scaling[1] * anchor_h + anchor_cy
            pred_cx = pred_location[:, 0] * self._prior_scaling[0] * anchor_w + anchor_cx

            return tf.split(tf.stack(self.center2point(pred_cy, pred_cx, pred_h, pred_w), axis=-1), num_anchors_per_layer, axis=0)

    def ext_decode_all_anchors(self, pred_location, all_anchors, all_num_anchors_depth, all_num_anchors_spatial):
        assert (len(all_num_anchors_depth)==len(all_num_anchors_spatial)) and (len(all_num_anchors_depth)==len(all_anchors)), 'inconsist num layers for anchors.'
        with tf.name_scope('ext_decode_all_anchors', [pred_location]):
            num_anchors_per_layer = []
            for ind in range(len(all_anchors)):
                num_anchors_per_layer.append(all_num_anchors_depth[ind] * all_num_anchors_spatial[ind])

            num_layers = len(all_num_anchors_depth)
            list_anchors_ymin = []
            list_anchors_xmin = []
            list_anchors_ymax = []
            list_anchors_xmax = []
            tiled_allowed_borders = []
            for ind, anchor in enumerate(all_anchors):
                anchors_ymin_, anchors_xmin_, anchors_ymax_, anchors_xmax_ = self.center2point(anchor[0], anchor[1], anchor[2], anchor[3])
                anchors_ymin_ = tf.transpose(anchors_ymin_, perm=[2, 0, 1])
                anchors_xmin_ = tf.transpose(anchors_xmin_, perm=[2, 0, 1])
                anchors_ymax_ = tf.transpose(anchors_ymax_, perm=[2, 0, 1])
                anchors_xmax_ = tf.transpose(anchors_xmax_, perm=[2, 0, 1])
                list_anchors_ymin.append(tf.reshape(anchors_ymin_, [-1]))
                list_anchors_xmin.append(tf.reshape(anchors_xmin_, [-1]))
                list_anchors_ymax.append(tf.reshape(anchors_ymax_, [-1]))
                list_anchors_xmax.append(tf.reshape(anchors_xmax_, [-1]))

            anchors_ymin = tf.concat(list_anchors_ymin, 0, name='concat_ymin')
            anchors_xmin = tf.concat(list_anchors_xmin, 0, name='concat_xmin')
            anchors_ymax = tf.concat(list_anchors_ymax, 0, name='concat_ymax')
            anchors_xmax = tf.concat(list_anchors_xmax, 0, name='concat_xmax')

            anchor_cy, anchor_cx, anchor_h, anchor_w = self.point2center(anchors_ymin, anchors_xmin, anchors_ymax, anchors_xmax)

            pred_h = tf.exp(pred_location[:,-1] * self._prior_scaling[3]) * anchor_h
            pred_w = tf.exp(pred_location[:, -2] * self._prior_scaling[2]) * anchor_w
            pred_cy = pred_location[:, 1] * self._prior_scaling[1] * anchor_h + anchor_cy
            pred_cx = pred_location[:, 0] * self._prior_scaling[0] * anchor_w + anchor_cx

            return tf.split(tf.stack(self.center2point(pred_cy, pred_cx, pred_h, pred_w), axis=-1), num_anchors_per_layer, axis=0)

class AnchorCreator(object):
    def __init__(self, img_shape, layers_shapes, anchor_scales, extra_anchor_scales, anchor_ratios, layer_steps):
        super(AnchorCreator, self).__init__()
        # img_shape -> (height, width)
        self._img_shape = img_shape
        self._layers_shapes = layers_shapes
        self._anchor_scales = anchor_scales
        self._extra_anchor_scales = extra_anchor_scales
        self._anchor_ratios = anchor_ratios
        self._layer_steps = layer_steps
        self._anchor_offset = [0.5] * len(self._layers_shapes)

    def get_layer_anchors(self, layer_shape, anchor_scale, extra_anchor_scale, anchor_ratio, layer_step, offset = 0.5):
        ''' assume layer_shape[0] = 6, layer_shape[1] = 5
        x_on_layer = [[0, 1, 2, 3, 4],
                       [0, 1, 2, 3, 4],
                       [0, 1, 2, 3, 4],
                       [0, 1, 2, 3, 4],
                       [0, 1, 2, 3, 4],
                       [0, 1, 2, 3, 4]]
        y_on_layer = [[0, 0, 0, 0, 0],
                       [1, 1, 1, 1, 1],
                       [2, 2, 2, 2, 2],
                       [3, 3, 3, 3, 3],
                       [4, 4, 4, 4, 4],
                       [5, 5, 5, 5, 5]]
        '''
        with tf.name_scope('get_layer_anchors'):
            x_on_layer, y_on_layer = tf.meshgrid(tf.range(layer_shape[1]), tf.range(layer_shape[0]))
            y_on_image = (tf.cast(y_on_layer, tf.float32) + offset) * layer_step / self._img_shape[0]
            x_on_image = (tf.cast(x_on_layer, tf.float32) + offset) * layer_step / self._img_shape[1]
            num_anchors_along_depth = 2 * len(anchor_scale) * len(anchor_ratio) + len(extra_anchor_scale) + len(anchor_scale)
            num_anchors_along_spatial = layer_shape[1] * layer_shape[0]

            list_h_on_image = []
            list_w_on_image = []

            # for square anchors
            for _, scales in enumerate(zip(anchor_scale, extra_anchor_scale)):
                list_h_on_image.append(scales[0])
                list_w_on_image.append(scales[0])
                list_h_on_image.append(math.sqrt(scales[0] * scales[1]))
                list_w_on_image.append(math.sqrt(scales[0] * scales[1]))
            # for other aspect ratio anchors
            for scale_index, scale in enumerate(anchor_scale):
                for ratio_index, ratio in enumerate(anchor_ratio):
                    h, w = scale / math.sqrt(ratio), scale * math.sqrt(ratio)
                    list_h_on_image.append(h)
                    list_w_on_image.append(w)
                    list_h_on_image.append(w)
                    list_w_on_image.append(h)
            # shape info:
            # y_on_image, x_on_image: layers_shapes[0] * layers_shapes[1]
            # h_on_image, w_on_image: num_anchors_along_depth
            return tf.expand_dims(y_on_image, axis=-1), tf.expand_dims(x_on_image, axis=-1), \
                    tf.constant(list_h_on_image, dtype=tf.float32), \
                    tf.constant(list_w_on_image, dtype=tf.float32), num_anchors_along_depth, num_anchors_along_spatial

    def get_all_anchors(self):
        all_anchors = []
        all_num_anchors_depth = []
        all_num_anchors_spatial = []
        for layer_index, layer_shape in enumerate(self._layers_shapes):
            anchors_this_layer = self.get_layer_anchors(layer_shape,
                                                        self._anchor_scales[layer_index],
                                                        self._extra_anchor_scales[layer_index],
                                                        self._anchor_ratios[layer_index],
                                                        self._layer_steps[layer_index],
                                                        self._anchor_offset[layer_index])
            all_anchors.append(anchors_this_layer[:-2])
            all_num_anchors_depth.append(anchors_this_layer[-2])
            all_num_anchors_spatial.append(anchors_this_layer[-1])
        return all_anchors, all_num_anchors_depth, all_num_anchors_spatial


    def encode_all_anchors(self, all_anchors, all_num_anchors_depth, all_num_anchors_spatial):
        # y, x, h, w are all in range [0, 1] relative to the original image size
        # shape info:
        # y_on_image, x_on_image: layers_shapes[0] * layers_shapes[1]
        # h_on_image, w_on_image: num_anchors
        assert (len(all_num_anchors_depth)==len(all_num_anchors_spatial)) and (len(all_num_anchors_depth)==len(all_anchors)), 'inconsist num layers for anchors.'
        with tf.name_scope('encode_all_anchors'):
            num_layers = len(all_num_anchors_depth)
            list_anchors_ymin = []
            list_anchors_xmin = []
            list_anchors_ymax = []
            list_anchors_xmax = []
            tiled_allowed_borders = []
            for ind, anchor in enumerate(all_anchors):
                anchors_ymin_, anchors_xmin_, anchors_ymax_, anchors_xmax_ = self.center2point(anchor[0], anchor[1], anchor[2], anchor[3])
                anchors_ymin_ = tf.transpose(anchors_ymin_, perm=[2, 0, 1])
                anchors_xmin_ = tf.transpose(anchors_xmin_, perm=[2, 0, 1])
                anchors_ymax_ = tf.transpose(anchors_ymax_, perm=[2, 0, 1])
                anchors_xmax_ = tf.transpose(anchors_xmax_, perm=[2, 0, 1])
                list_anchors_ymin.append(tf.reshape(anchors_ymin_, [-1]))
                list_anchors_xmin.append(tf.reshape(anchors_xmin_, [-1]))
                list_anchors_ymax.append(tf.reshape(anchors_ymax_, [-1]))
                list_anchors_xmax.append(tf.reshape(anchors_xmax_, [-1]))
            anchors_ymin = tf.concat(list_anchors_ymin, 0, name='concat_ymin')
            anchors_xmin = tf.concat(list_anchors_xmin, 0, name='concat_xmin')
            anchors_ymax = tf.concat(list_anchors_ymax, 0, name='concat_ymax')
            anchors_xmax = tf.concat(list_anchors_xmax, 0, name='concat_xmax')
            #anchors_ymin = tf.clip_by_value(anchors_ymin, 0., 1.)
            #anchors_xmin = tf.clip_by_value(anchors_xmin, 0., 1.)
            #anchors_ymax = tf.clip_by_value(anchors_ymax, 0., 1.)
            #anchors_xmax = tf.clip_by_value(anchors_xmax, 0., 1.)
       
        return tf.stack([anchors_xmin, anchors_ymin, anchors_xmax, anchors_ymax], axis=-1)  

    def center2point(self, center_y, center_x, height, width):
        return center_y - height / 2., center_x - width / 2., center_y + height / 2., center_x + width / 2.,

    def point2center(self, ymin, xmin, ymax, xmax):
        height, width = (ymax - ymin), (xmax - xmin)
        return ymin + height / 2., xmin + width / 2., height, width

        

if __name__ == "__main__":
    import os
    out_shape = [1200] * 2
    anchor_creator = AnchorCreator(out_shape,
                                   layers_shapes = [(50, 50), (25, 25), (13, 13), (7, 7), (3, 3), (3, 3)],
                                   anchor_scales = [(0.07,), (0.15,), (0.33,), (0.51,), (0.69,), (0.87,)],
                                   extra_anchor_scales = [(0.15,), (0.33,), (0.51,), (0.69,), (0.87,), (1.05,)],
                                   anchor_ratios = [(2, ), (2., 3.), (2., 3), (2., 3.), (2.,), (2.,)],
                                   layer_steps = [24, 48, 92, 171, 400, 400])
    all_anchors, all_num_anchors_depth, all_num_anchors_spatial = anchor_creator.get_all_anchors()
    anchor_bboxes = anchor_creator.encode_all_anchors(all_anchors, all_num_anchors_depth, all_num_anchors_spatial)
