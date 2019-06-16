# Copyright 2018 Changan Wang

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
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

from model import ssd_net_resnet34_large 
from dataset import dataset_common
from utils import ssd_preprocessing
from utils import anchor_manipulator
from utils import scaffolds

tf.app.flags.DEFINE_integer(
    'num_readers', 8,
    'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 24,
    'The number of threads used to create the batches.')
tf.app.flags.DEFINE_integer(
    'num_cpu_threads', 0,
    'The number of cpu cores used to train.')
tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 1., 'GPU memory fraction to use.')
tf.app.flags.DEFINE_string(
    'data_dir', './tfrecords/',
    'The directory where the dataset input data is stored.')
tf.app.flags.DEFINE_integer(
    'num_classes', 81, 'Number of classes to use in the dataset.')
tf.app.flags.DEFINE_string(
    'model_dir', './logs_mine_sec.ssd_resnet34_pretrain.20.2/',
    'The directory where the model will be stored.')
tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are printed.')
tf.app.flags.DEFINE_integer(
    'save_summary_steps', 500,
    'The frequency with which summaries are saved, in seconds.')
tf.app.flags.DEFINE_integer(
    'save_checkpoints_secs', 3600,
    'The frequency with which the model is saved, in seconds.')
tf.app.flags.DEFINE_integer(
    'train_image_size', 1200,
    'The size of the input image for the model to use.')
tf.app.flags.DEFINE_integer(
    'train_epochs', None,
    'The number of epochs to use for training.')
tf.app.flags.DEFINE_integer(
    'max_number_of_steps', 580000,
    'The max number of steps to use for training.')
tf.app.flags.DEFINE_integer(
    'batch_size', 48,
    'Batch size for training and evaluation.')
tf.app.flags.DEFINE_string(
    'data_format', 'channels_first',
    'A flag to override the data format used in the model. channels_first '
    'provides a performance boost on GPU but is not always compatible '
    'with CPU. If left unspecified, the data format will be chosen '
    'automatically based on whether TensorFlow was built for CPU or GPU.')
tf.app.flags.DEFINE_float(
    'negative_ratio', 3., 'Negative ratio in the loss function.')
tf.app.flags.DEFINE_float(
    'match_threshold', 0.5, 'Matching threshold in the loss function.')
tf.app.flags.DEFINE_float(
    'neg_threshold', 0.5, 'Matching threshold for the negtive examples in the loss function.')
tf.app.flags.DEFINE_integer(
    'tf_random_seed', 20180503, 'Random seed for TensorFlow initializers.')
tf.app.flags.DEFINE_float(
    'weight_decay', 5e-4, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')
tf.app.flags.DEFINE_float('learning_rate', 4e-3, 'Initial learning rate.')
tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.000001,
    'The minimal end learning rate used by a polynomial decay learning rate.')
tf.app.flags.DEFINE_string(
    'decay_boundaries', '420000, 425000, 480000, 540000, 580000',
    'Learning rate decay boundaries by global_step (comma-separated list).')
tf.app.flags.DEFINE_string(
    'lr_decay_factors', '0.01, 0.1, 1, 0.4, 0.1, 0.01',
    'The values of learning_rate decay factor for each segment between boundaries (comma-separated list).')
tf.app.flags.DEFINE_string(
    'checkpoint_path', './model.ssd300.rename/new_ckpt',
    'The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_string(
    'checkpoint_model_scope', 'ssd300',
    'Model scope in the checkpoint. None if the same as the trained model.')
tf.app.flags.DEFINE_string(
    'model_scope', 'ssd1200',
    'Model scope name used to replace the name_scope in checkpoint.')
tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', '',
    'Comma-separated list of scopes of variables to exclude when restoring from a checkpoint.')
tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', True,
    'When restoring a checkpoint would ignore missing variables.')
tf.app.flags.DEFINE_boolean(
    'multi_gpu', True,
    'Whether there is GPU to use for training.')
FLAGS = tf.app.flags.FLAGS

def validate_batch_size_for_multi_gpu(batch_size):
    """For multi-gpu, batch-size must be a multiple of the number of
    available GPUs.
    Note that this should eventually be handled by replicate_model_fn
    directly. Multi-GPU support is currently experimental, however,
    so doing the work here until that feature is in place.
    """
    if FLAGS.multi_gpu:
        from tensorflow.python.client import device_lib
        local_device_protos = device_lib.list_local_devices()
        num_gpus = sum([1 for d in local_device_protos if d.device_type == 'GPU'])
        if not num_gpus:
            raise ValueError('Multi-GPU mode was specified, but no GPUs '
                             'were found. To use CPU, run --multi_gpu=False.')
        remainder = batch_size % num_gpus
        if remainder:
            err = ('When running with multiple GPUs, batch size '
                   'must be a multiple of the number of available GPUs. '
                   'Found {} GPUs with a batch size of {}; try --batch_size={} instead.'
                  ).format(num_gpus, batch_size, batch_size - remainder)
            raise ValueError(err)
        return num_gpus
    return 0

def get_init_fn():
    return scaffolds.get_init_fn_for_scaffold(FLAGS.model_dir, FLAGS.checkpoint_path,
                                              FLAGS.model_scope, FLAGS.checkpoint_model_scope,
                                              FLAGS.checkpoint_exclude_scopes, FLAGS.ignore_missing_vars,
                                              name_remap={'/kernel': '/weights', '/bias': '/biases'})
global_anchor_info = dict()

def input_pipeline(dataset_pattern='pascalvoc_0712_train_*', is_training=True, batch_size=FLAGS.batch_size):
    def input_fn():
        out_shape = [FLAGS.train_image_size] * 2
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

        image_preprocessing_fn = lambda image_, labels_, bboxes_ : ssd_preprocessing.preprocess_image(image_, labels_, bboxes_, out_shape, is_training=is_training, data_format=FLAGS.data_format, output_rgb=False)
        anchor_encoder_fn = lambda glabels_, gbboxes_: anchor_encoder_decoder.encode_all_anchors(glabels_, gbboxes_, all_anchors, all_num_anchors_depth, all_num_anchors_spatial)
        image, _, shape, loc_targets, cls_targets, match_scores = dataset_common.slim_get_batch(FLAGS.num_classes,
                                                                                batch_size,
                                                                                ('train' if is_training else 'val'),
                                                                                os.path.join(FLAGS.data_dir, dataset_pattern),
                                                                                FLAGS.num_readers,
                                                                                FLAGS.num_preprocessing_threads,
                                                                                image_preprocessing_fn,
                                                                                anchor_encoder_fn,
                                                                                num_epochs=FLAGS.train_epochs,
                                                                                is_training=is_training)
        global global_anchor_info
        global_anchor_info = {'decode_fn': lambda pred : anchor_encoder_decoder.decode_all_anchors(pred, num_anchors_per_layer),
                            'num_anchors_per_layer': num_anchors_per_layer,
                            'all_num_anchors_depth': all_num_anchors_depth }
        return image, {'shape': shape, 'loc_targets': loc_targets, 'cls_targets': cls_targets, 'match_scores': match_scores}
    return input_fn

def modified_smooth_l1(bbox_pred, bbox_targets, bbox_inside_weights=1., bbox_outside_weights=1., sigma=1.):
    with tf.name_scope('smooth_l1', [bbox_pred, bbox_targets]):
        sigma2 = sigma * sigma
        inside_mul = tf.multiply(bbox_inside_weights, tf.subtract(bbox_pred, bbox_targets))
        smooth_l1_sign = tf.cast(tf.less(tf.abs(inside_mul), 1.0 / sigma2), tf.float32)
        smooth_l1_option1 = tf.multiply(tf.multiply(inside_mul, inside_mul), 0.5 * sigma2)
        smooth_l1_option2 = tf.subtract(tf.abs(inside_mul), 0.5 / sigma2)
        smooth_l1_result = tf.add(tf.multiply(smooth_l1_option1, smooth_l1_sign),
                                  tf.multiply(smooth_l1_option2, tf.abs(tf.subtract(smooth_l1_sign, 1.0))))
        outside_mul = tf.multiply(bbox_outside_weights, smooth_l1_result)
        return outside_mul

def ssd_model_fn(features, labels, mode, params):
    shape = labels['shape']
    loc_targets = labels['loc_targets']
    cls_targets = labels['cls_targets']
    match_scores = labels['match_scores']
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
        cls_pred = tf.reshape(cls_pred, [-1, params['num_classes']])
        location_pred = tf.reshape(location_pred, [-1, 4])

    with tf.device('/cpu:0'):
        with tf.control_dependencies([cls_pred, location_pred]):
            with tf.name_scope('post_forward'):
                #bboxes_pred = decode_fn(location_pred)
                bboxes_pred = tf.map_fn(lambda _preds : decode_fn(_preds),
                                        tf.reshape(location_pred, [tf.shape(features)[0], -1, 4]),
                                        dtype=[tf.float32] * len(num_anchors_per_layer), back_prop=False)
                #cls_targets = tf.Print(cls_targets, [tf.shape(bboxes_pred[0]),tf.shape(bboxes_pred[1]),tf.shape(bboxes_pred[2]),tf.shape(bboxes_pred[3])])
                bboxes_pred = [tf.reshape(preds, [-1, 4]) for preds in bboxes_pred]
                bboxes_pred = tf.concat(bboxes_pred, axis=0)

                flaten_cls_targets = tf.reshape(cls_targets, [-1])
                flaten_match_scores = tf.reshape(match_scores, [-1])
                flaten_loc_targets = tf.reshape(loc_targets, [-1, 4])

                # each positive examples has one label
                positive_mask = flaten_cls_targets > 0
                n_positives = tf.count_nonzero(positive_mask)

                batch_n_positives = tf.count_nonzero(cls_targets, -1)

                batch_negtive_mask = tf.equal(cls_targets, 0)#tf.logical_and(tf.equal(cls_targets, 0), match_scores > 0.)
                batch_n_negtives = tf.count_nonzero(batch_negtive_mask, -1)

                batch_n_neg_select = tf.cast(params['negative_ratio'] * tf.cast(batch_n_positives, tf.float32), tf.int32)
                batch_n_neg_select = tf.minimum(batch_n_neg_select, tf.cast(batch_n_negtives, tf.int32))

                # hard negative mining for classification
                predictions_for_bg = tf.nn.softmax(tf.reshape(cls_pred, [tf.shape(features)[0], -1, params['num_classes']]))[:, :, 0]
                prob_for_negtives = tf.where(batch_negtive_mask,
                                       0. - predictions_for_bg,
                                       # ignore all the positives
                                       0. - tf.ones_like(predictions_for_bg))
                topk_prob_for_bg, _ = tf.nn.top_k(prob_for_negtives, k=tf.shape(prob_for_negtives)[1])
                score_at_k = tf.gather_nd(topk_prob_for_bg, tf.stack([tf.range(tf.shape(features)[0]), batch_n_neg_select - 1], axis=-1))

                selected_neg_mask = prob_for_negtives >= tf.expand_dims(score_at_k, axis=-1)

                # include both selected negtive and all positive examples
                final_mask = tf.stop_gradient(tf.logical_or(tf.reshape(tf.logical_and(batch_negtive_mask, selected_neg_mask), [-1]), positive_mask))
                total_examples = tf.count_nonzero(final_mask)

                cls_pred = tf.boolean_mask(cls_pred, final_mask)
                location_pred = tf.boolean_mask(location_pred, tf.stop_gradient(positive_mask))
                flaten_cls_targets = tf.boolean_mask(tf.clip_by_value(flaten_cls_targets, 0, params['num_classes']), final_mask)
                flaten_loc_targets = tf.stop_gradient(tf.boolean_mask(flaten_loc_targets, positive_mask))

                predictions = {
                            'classes': tf.argmax(cls_pred, axis=-1),
                            'probabilities': tf.reduce_max(tf.nn.softmax(cls_pred, name='softmax_tensor'), axis=-1),
                            'loc_predict': bboxes_pred }

                cls_accuracy = tf.metrics.accuracy(flaten_cls_targets, predictions['classes'])
                metrics = {'cls_accuracy': cls_accuracy}

                # Create a tensor named train_accuracy for logging purposes.
                tf.identity(cls_accuracy[1], name='cls_accuracy')
                tf.summary.scalar('cls_accuracy', cls_accuracy[1])

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    #flaten_cls_targets=tf.Print(flaten_cls_targets, [flaten_loc_targets],summarize=50000)
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=flaten_cls_targets, logits=cls_pred) * (params['negative_ratio'] + 1.)
    # Create a tensor named cross_entropy for logging purposes.
    tf.identity(cross_entropy, name='cross_entropy_loss')
    tf.summary.scalar('cross_entropy_loss', cross_entropy)

    #loc_loss = tf.cond(n_positives > 0, lambda: modified_smooth_l1(location_pred, tf.stop_gradient(flaten_loc_targets), sigma=1.), lambda: tf.zeros_like(location_pred))
    loc_loss = modified_smooth_l1(location_pred, flaten_loc_targets, sigma=1.)
    #loc_loss = modified_smooth_l1(location_pred, tf.stop_gradient(gtargets))
    loc_loss = tf.reduce_mean(tf.reduce_sum(loc_loss, axis=-1), name='location_loss')
    tf.summary.scalar('location_loss', loc_loss)
    tf.losses.add_loss(loc_loss)

    l2_loss_vars = []
    for trainable_var in tf.trainable_variables():
        if '_bn' not in trainable_var.name:
            if 'conv4_3_scale' not in trainable_var.name:
                l2_loss_vars.append(tf.nn.l2_loss(trainable_var) * 0.1)
            else:
                l2_loss_vars.append(tf.nn.l2_loss(trainable_var) * 0.1)
    # Add weight decay to the loss. We exclude the batch norm variables because
    # doing so leads to a small improvement in accuracy.
    total_loss = tf.add(cross_entropy + loc_loss, tf.multiply(params['weight_decay'], tf.add_n(l2_loss_vars), name='l2_loss'), name='total_loss')

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()

        lr_values = [params['learning_rate'] * decay for decay in params['lr_decay_factors']]
        learning_rate = tf.train.piecewise_constant(tf.cast(global_step, tf.int32),
                                                    [int(_) for _ in params['decay_boundaries']],
                                                    lr_values)
        truncated_learning_rate = tf.maximum(learning_rate, tf.constant(params['end_learning_rate'], dtype=learning_rate.dtype), name='learning_rate')
        # Create a tensor named learning_rate for logging purposes.
        tf.summary.scalar('learning_rate', truncated_learning_rate)

        optimizer = tf.train.MomentumOptimizer(learning_rate=truncated_learning_rate,
                                                momentum=params['momentum'])
        optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)

        # Batch norm requires update_ops to be added as a train_op dependency.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(total_loss, global_step)
    else:
        train_op = None

    return tf.estimator.EstimatorSpec(
                              mode=mode,
                              predictions=predictions,
                              loss=total_loss,
                              train_op=train_op,
                              eval_metric_ops=metrics,
                              #scaffold=None)
                              scaffold=tf.train.Scaffold(init_fn=get_init_fn()))

def parse_comma_list(args):
    return [float(s.strip()) for s in args.split(',')]

def main(_):
    os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
    #tf.set_pruning_mode()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, intra_op_parallelism_threads=FLAGS.num_cpu_threads, inter_op_parallelism_threads=FLAGS.num_cpu_threads, gpu_options=gpu_options)
    num_gpus = validate_batch_size_for_multi_gpu(FLAGS.batch_size)

    run_config = tf.estimator.RunConfig().replace(
                                        save_checkpoints_secs=FLAGS.save_checkpoints_secs).replace(
                                        save_checkpoints_steps=None).replace(
                                        save_summary_steps=FLAGS.save_summary_steps).replace(
                                        keep_checkpoint_max=5).replace(
                                        tf_random_seed=FLAGS.tf_random_seed).replace(
                                        log_step_count_steps=FLAGS.log_every_n_steps).replace(
                                        session_config=config)

    replicate_ssd_model_fn = tf.contrib.estimator.replicate_model_fn(ssd_model_fn, loss_reduction=tf.losses.Reduction.MEAN)
    ssd_detector = tf.estimator.Estimator(
        model_fn=replicate_ssd_model_fn, model_dir=FLAGS.model_dir, config=run_config,
        params={
            'num_gpus': num_gpus,
            'data_format': FLAGS.data_format,
            'batch_size': FLAGS.batch_size,
            'model_scope': FLAGS.model_scope,
            'num_classes': FLAGS.num_classes,
            'negative_ratio': FLAGS.negative_ratio,
            'match_threshold': FLAGS.match_threshold,
            'neg_threshold': FLAGS.neg_threshold,
            'weight_decay': FLAGS.weight_decay,
            'momentum': FLAGS.momentum,
            'learning_rate': FLAGS.learning_rate,
            'end_learning_rate': FLAGS.end_learning_rate,
            'decay_boundaries': parse_comma_list(FLAGS.decay_boundaries),
            'lr_decay_factors': parse_comma_list(FLAGS.lr_decay_factors),
        })
    tensors_to_log = {
        'lr': 'learning_rate',
        'ce': 'cross_entropy_loss',
        'loc': 'location_loss',
        'loss': 'total_loss',
        'l2': 'l2_loss',
        'acc': 'post_forward/cls_accuracy',
    }
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=FLAGS.log_every_n_steps,
                                            formatter=lambda dicts: (', '.join(['%s=%.6f' % (k, v) for k, v in dicts.items()])))
    print('Starting a training cycle.')
    ssd_detector.train(input_fn=input_pipeline(dataset_pattern='coco_2017_train-*', is_training=True, batch_size=FLAGS.batch_size),
                    hooks=[logging_hook], max_steps=FLAGS.max_number_of_steps)
if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
