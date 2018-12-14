""" Using TensorFlow MTCNN to do face detection and facial landmark detection """
""" Then estimate the transformation matrix to do affine transform to align face """
# MIT License
#
# Copyright (c) 2018 Jimmy Chiang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
from scipy import misc
import cv2
import sys
import os
import argparse
import tensorflow as tf
import numpy as np

import align.detect_face
from facenet import get_dataset


def _cp2tform(inputPoints, coordPoints):
    if inputPoints.shape != coordPoints.shape:
        raise ValueError('The shapes of the two point array are different!')
    if inputPoints.shape[1] != 2:
        raise ValueError('The points must be an (m x 2) np-array')
    x = np.array([coordPoints[:, 0]]).T
    y = np.array([coordPoints[:, 1]]).T
    u = np.array([inputPoints[:, 0]]).T
    v = np.array([inputPoints[:, 1]]).T
    tmp0 = np.zeros((len(u), 1))
    tmp1 = np.ones((len(u), 1))

    # without reflection
    Atop = np.concatenate((u, v, tmp1, tmp0), axis=1)
    Adown = np.concatenate((v, -u, tmp0, tmp1), axis=1)
    A = np.concatenate((Atop, Adown), axis=0)
    B = np.concatenate((x, y), axis=0)
    solveeq = np.linalg.lstsq(A, B)
    tmat1 = solveeq[0]
    resid1 = np.sqrt(solveeq[1])[0]

    # with reflection
    Adown = np.concatenate((-v, u, tmp0, tmp1), axis=1)
    A = np.concatenate((Atop, Adown), axis=0)
    solveeq = np.linalg.lstsq(A, B)
    tmat2 = solveeq[0]
    resid2 = np.sqrt(solveeq[1])[0]

    if (resid2 < resid1):
        tmat = np.array((tmat2[0, 0], tmat2[1, 0], tmat2[1, 0], -tmat2[0, 0], tmat2[2, 0], tmat2[3, 0]))
        tmat = tmat.reshape((3, 2))
    else:
        tmat = np.array((tmat1[0, 0], -tmat1[1, 0], tmat1[1, 0], tmat1[0, 0], tmat1[2, 0], tmat1[3, 0]))
        tmat = tmat.reshape((3, 2))

    return tmat

def _align_face(img, landmarks, height, width):
    coord5point = np.float32(
        [[30.2946, 51.6963], [65.5318, 51.5014], [48.0252, 71.7366], [33.5493, 92.3655], [62.7299, 92.2041]])
    landmarks = np.reshape(landmarks, (2, 5)).T
    tmat = _cp2tform(landmarks, coord5point)
    img_align = cv2.warpAffine(img, tmat.T, (width, height))
    return img_align

def sphereface_preprocess(input_dir, output_dir):
    
    output_dir = os.path.expanduser(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print('Creating MTCNN networks and loading parameters')
    
    with tf.Graph().as_default():
        sess = tf.Session()
        pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    image_width = 96
    image_height = 112
    minsize = 20 # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor
    margin = 44

    dataset = get_dataset(input_dir)
    nrof_images_total = 0
    nrof_successfully_aligned = 0
    for cls in dataset:
        output_class_dir = os.path.join(output_dir, cls.name)
        if not os.path.exists(output_class_dir):
            os.makedirs(output_class_dir)
        for image_path in cls.image_paths:
            nrof_images_total += 1
            filename = os.path.splitext(os.path.split(image_path)[1])[0]
            output_filename = os.path.join(output_class_dir, filename + '.png')
            output_filename_flip = os.path.join(output_class_dir, filename + '_flip.png')
            # print(image_path)
            if not os.path.exists(output_filename):
                try:
                    img = misc.imread(image_path)
                except (IOError, ValueError, IndexError) as e:
                    errorMessage = '{}: {}'.format(image_path, e)
                    print(errorMessage)
                else:
                    img = img[:, :, 0:3]
                    bounding_boxes, landmarks = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                    nrof_faces = bounding_boxes.shape[0]
                    if nrof_faces > 0:
                        det = bounding_boxes[:,0:4]
                        img_size = np.asarray(img.shape)[0:2]
                        if nrof_faces > 1:
                            bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                            img_center = img_size / 2
                            offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                            
                            offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                            index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                            
                            det = det[index, :]
                            landmarks = landmarks[:, index]

                        det = np.squeeze(det)
                        bb = np.zeros(4, dtype=np.int32)
                        bb[0] = np.maximum(det[0] - margin / 2, 0)
                        bb[1] = np.maximum(det[1] - margin / 2, 0)
                        bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                        bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                        aligned = _align_face(img, landmarks, image_height, image_width)

                        nrof_successfully_aligned += 1
                        misc.imsave(output_filename, aligned)
                        misc.imsave(output_filename_flip, np.fliplr(aligned))
                    else:
                        print('Unable to align "%s"' % image_path)
    
    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)
