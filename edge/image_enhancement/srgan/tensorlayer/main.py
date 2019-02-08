# Copyright 2018 The MLPerf Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

# This code is modified from https://github.com/tensorlayer/srgan/blob/master/main.py

import os, time
from datetime import datetime
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from model import SRGAN_g
from utils import *
from config import config, log_config
import json
from tensorflow.python.client import device_lib
from skimage.measure import compare_psnr
from skimage import io
import glob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    for x in local_device_protos:
        gpu_name = x.physical_device_desc
    return gpu_name


def get_psnr(test_im_dir, pred_im_dir):
    """ Calculates average PSNR for image pairs from the test image
    director and the generated image directory.

    Args:
        test_im_dir: path for the test HR image directory in string
        pred_im_dir: path for the generated HR image directory in
            string.
    Returns: None
    """

    all_psnr = list()
    test_im_files = sorted(glob.glob(test_im_dir + '*.png'))
    pred_im_files = sorted(glob.glob(pred_im_dir + '/*.png'))

    for i in range(len(pred_im_files)):
        im_test = io.imread(test_im_files[i])
        im_pred = io.imread(pred_im_files[i])
        psnr = compare_psnr(im_test, im_pred)
        print("PSNR is %4.2f for image %g" % (psnr, i+1))
        all_psnr.append(psnr)
    average_psnr = sum(all_psnr)/len(all_psnr)
    print("Average PSNR %4.2f" % average_psnr)
    return   


def run_model(batch_size):
    """Use the model to generate HR images from LR images
    and report either throughput or accuracy.

    Args:
        batch_size: batch size in an integer.

    Returns:
        None
    """

    # create folders to save result images
    save_dir = "{}/".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir)
    checkpoint_dir = "checkpoint"

    ###====================== PRE-LOAD DATA ===========================###
    valid_lr_img_list = sorted(tl.files.load_file_list(
        path=config.VALID.lr_img_path, regx='.*.png', printable=False))

    valid_lr_imgs = tl.vis.read_images(valid_lr_img_list,
                                       path=config.VALID.lr_img_path,
                                       n_threads=32)

    ###========================== DEFINE MODEL ============================###

    image_count = len(valid_lr_img_list)
    t_image = tf.placeholder('float32', [None, None, None, 3],
                             name='input_image')
    net_g = SRGAN_g(t_image, is_train=False, reuse=False)

    ###========================== RESTORE G =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                            log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess,
                                 name=checkpoint_dir + '/g_srgan.npz',
                                 network=net_g)

    ###======================= EVALUATION =============================###
    # warm-up
    for i in range(warm_iter):
        curr_image = np.expand_dims(valid_lr_imgs[0], axis=0)
        repeated_image = np.repeat(curr_image, batch_size, axis=0)
        input_data = tl.prepro.threading_data(repeated_image, fn=pre)
        out = sess.run(net_g.outputs, {t_image: input_data})

    total_time = []
    for i in range(image_count):
        print('predict image %g ' % (i + 1))
        curr_image = np.expand_dims(valid_lr_imgs[i], axis=0)
        repeated_image = np.repeat(curr_image, batch_size, axis=0)
        input_data = tl.prepro.threading_data(repeated_image, fn=pre)
        start_time = time.time()
        out = sess.run(net_g.outputs, {t_image: input_data})
        step_time = time.time()
        total_time.append(step_time - start_time)
        # save generated HR images only in the accuracy mode
        if tl.global_flag['mode'] == 'accuracy':
            tl.vis.save_image(out[0], save_dir + format(i + 1, '04') + '.png')

    # record throughput only in the 'performance mode'
    if tl.global_flag['mode'] == 'performance':
        n_gpu = get_available_gpus()
        cuda_flag = 0
        if tf.test.is_built_with_cuda():
            cuda_flag = 1

        if cuda_flag == 1:
            data = {
                "Device": "GPU",
                "Name of GPU": n_gpu,
                "Mean fps": round(1 / (sum(total_time) / (len(
                    total_time) * batch_size)), 2),
                "Time Taken to infer(second)": round(sum(total_time), 2),
                "Batch_Size": batch_size,
                "model": "g_srgan.npz",
                "Framework": "Tensorflow",
                "Input model Precision ": "Fp32",
                "Training Dataset": "Div2k",
                "Input Images": "./DIV2K_valid_LR_bicubic/X4/",
                "Output Images": "./samples/evaluate/",
            }
        else:
            data = {
                "Device": "CPU",
                "Mean fps": round(1 / (sum(total_time) / (len(
                    total_time) * batch_size)), 2),
                "Time Taken to infer(second)": round(sum(total_time), 2),
                "Batch_Size": batch_size,
                "model": "g_srgan.npz",
                "Framework": "Tensorflow",
                "Input model Precision ": "Fp32",
                "Training Dataset": "Div2k",
                "Input Images": "./DIV2K_valid_LR_bicubic/X4/",
                "Output Images": "./samples/evaluate/",
            }
        print('batch size %d, mean fps %4.2f' % (batch_size, 1 / (sum(
            total_time) / (len(total_time) * batch_size))))
        with open('result.json', 'a') as outfile:
            outfile.write(json.dumps(data, indent=4, sort_keys=True))
            outfile.write('\n')

    return


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='evaluate',
                        help='srgan, evaluate')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference.')
    args = parser.parse_args()

    warm_iter = 5

    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))

    tl.global_flag['mode'] = args.mode
    if tl.global_flag['mode'] == 'performance':
        run_model(args.batch_size)
    elif tl.global_flag['mode'] == 'accuracy':
        run_model(1)
        get_psnr(config.VALID.hr_img_path, "{}".format(tl.global_flag['mode']))
    else:
        raise Exception("Unknown --mode")
