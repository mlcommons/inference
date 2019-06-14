import pdb
import os
import time
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.python.platform import tf_logging
from preprocess.mtcnn_preprocess_align import sphereface_preprocess
from facenet import get_dataset
from lfw import read_pairs
from postprocess.eval import lfw_metric


flags = tf.app.flags
flags.DEFINE_string("input_model", './model/sphereface_float.tflite', 'Name of input tflite file')
flags.DEFINE_string("lfw_dir", '/tmp/dataset/lfw_set1', 'path of lfw validation set tfrecord')
FLAGS = flags.FLAGS

PREPROCESSED_DIR = '/tmp/dataset/lfw_set1_aligned'
LFW_PAIRS = '/tmp/dataset/pairs.txt'
NUM_EVAL_SAMPLES = 600
IMAGE_HEIGHT = 112
IMAGE_WIDTH = 96
MEAN = 128.0
STD = 127.5


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.ERROR)
    # pdb.set_trace()
    print("=========== Start Pre-processing ===========")
    start_time = time.time()
    sphereface_preprocess(FLAGS.lfw_dir, PREPROCESSED_DIR)
    end_time = time.time()
    preprocess_time = (end_time - start_time) * 1.0
    print("=========== Finish Pre-processing ==========")     
    print('It takes {:.4f} s to finish pre-processing'.format(preprocess_time))

    print("========== Start TFLIte inference ==========")
    start_inf = time.time()
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=FLAGS.input_model)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # prepare buffer for output result
    emb_0 = np.zeros((NUM_EVAL_SAMPLES, output_details[0]['shape'][1]))
    emb_0_flip = np.zeros((NUM_EVAL_SAMPLES, output_details[0]['shape'][1]))
    emb_1 = np.zeros((NUM_EVAL_SAMPLES, output_details[0]['shape'][1]))
    emb_1_flip = np.zeros((NUM_EVAL_SAMPLES, output_details[0]['shape'][1]))
    issame_arr = np.zeros(NUM_EVAL_SAMPLES)
    time0 = 0
    time0_flip = 0
    time1 = 0
    time1_flip = 0
    lfw_pairs = read_pairs(LFW_PAIRS)
    for idx in range(NUM_EVAL_SAMPLES):
        # check input data
        pair = lfw_pairs[idx]
        if len(pair) == 3:
            path0 = os.path.join(PREPROCESSED_DIR, pair[0], pair[0] + '_' + '%04d' % int(pair[1]))
            path1 = os.path.join(PREPROCESSED_DIR, pair[0], pair[0] + '_' + '%04d' % int(pair[2]))
            issame = 1
        elif len(pair) == 4:
            path0 = os.path.join(PREPROCESSED_DIR, pair[0], pair[0] + '_' + '%04d' % int(pair[1]))
            path1 = os.path.join(PREPROCESSED_DIR, pair[2], pair[2] + '_' + '%04d' % int(pair[3]))
            issame = 0
        path0_flip = path0 + '_flip.png'
        path0 = path0 + '.png'
        path1_flip = path1 + '_flip.png'
        path1 = path1 + '.png'
        if not os.path.exists(path0):
            raise ValueError('{} image does NOT exist!'.format(path0))
        if not os.path.exists(path1):
            raise ValueError('{} image does NOT exist!'.format(path1))
        if not os.path.exists(path0_flip):
            raise ValueError('{} image does NOT exist!'.format(path0_flip))
        if not os.path.exists(path1_flip):
            raise ValueError('{} image does NOT exist!'.format(path1_flip))
        issame_arr[idx] = int(issame)

        # get data array
        img0 = np.expand_dims(cv2.imread(path0), 0)
        img0_flip = np.expand_dims(cv2.imread(path0_flip), 0)
        img1 = np.expand_dims(cv2.imread(path1), 0)
        img1_flip = np.expand_dims(cv2.imread(path1_flip), 0)
        img0 = (img0.astype(np.float32) - MEAN) / STD
        img0_flip = (img0_flip.astype(np.float32) - MEAN) / STD
        img1 = (img1.astype(np.float32) - MEAN) / STD
        img1_flip = (img1_flip.astype(np.float32) - MEAN) / STD

        # inference image1
        interpreter.set_tensor(input_details[0]['index'], img0)
        start_time = time.time()
        interpreter.invoke()
        end_time = time.time()
        time0 += (end_time - start_time)
        out0 = interpreter.get_tensor(output_details[0]['index'])
        # inference image1_flip
        interpreter.set_tensor(input_details[0]['index'], img0_flip)
        start_time = time.time()
        interpreter.invoke()
        end_time = time.time()
        time0_flip += (end_time - start_time)
        out0_flip = interpreter.get_tensor(output_details[0]['index'])
        # inference image1
        interpreter.set_tensor(input_details[0]['index'], img1)
        start_time = time.time()
        interpreter.invoke()
        end_time = time.time()
        time1 += (end_time - start_time)
        out1 = interpreter.get_tensor(output_details[0]['index'])
        # inference image1_flip
        interpreter.set_tensor(input_details[0]['index'], img1_flip)
        start_time = time.time()
        interpreter.invoke()
        end_time = time.time()
        time1_flip += (end_time - start_time)
        out1_flip = interpreter.get_tensor(output_details[0]['index'])

        if idx % (NUM_EVAL_SAMPLES // 20) == 0:
            print("sample: [%d/%d]" % (idx, NUM_EVAL_SAMPLES))

        emb_0[idx, :] = out0
        emb_0_flip[idx, :] = out0_flip
        emb_1[idx, :] = out1
        emb_1_flip[idx, :] = out1_flip

    end_inf = time.time()
    inf_time = (end_inf - start_inf) * 1.0
    print("=========== End TFLIte inference ===========")
    print('It takes {:.4f} s to finish TFLite inference for all testing data'.format(inf_time))
    
    print("=========== Start Post-processing ==========")
    start_time = time.time()
    embedding_0 = np.concatenate((emb_0, emb_0_flip),axis=1)
    embedding_1 = np.concatenate((emb_1, emb_1_flip),axis=1)
    accuracy = lfw_metric(embedding_0, embedding_1, issame_arr)
    end_time = time.time()
    postprocess_time = (end_time - start_time) * 1.0
    print("============ End Post-processing ===========")
    print('It takes {:.4f} s to finish post-processing'.format(postprocess_time))

    avg_time = 1000.0 * (time0 + time0_flip + time1 + time1_flip) / float(NUM_EVAL_SAMPLES * 4)
    print("sample: [%d/%d]" % (NUM_EVAL_SAMPLES, NUM_EVAL_SAMPLES))
    print("Average time per inference = [%.4f ms]" % avg_time)
    print('Validation Reuslt, Accuracy = [%.4f]' % accuracy)

    logtime = time.localtime()
    logname = 'results/sphereface_benchmark_{:4d}{:2d}{:2d}_{:2d}{:2d}{:2d}.log'.format(
        logtime.tm_year, logtime.tm_mon, logtime.tm_mday, logtime.tm_hour, logtime.tm_min, logtime.tm_sec)
    with open(logname, 'w') as f:
        f.write('Phase\tseconds\n')
        f.write('Pre-processing time\t{}\n'.format(preprocess_time))
        f.write('Inference time for all testing pairs\t{}\n'.format(inf_time))
        f.write('Post-processing time\t{}\n\n'.format(postprocess_time))
        f.write('Average time for each TFLite model inference\t{} ms\n'.format(avg_time))
        f.write('Accuracy on 600 test pairs\t{}\n'.format(accuracy))


if __name__ == '__main__':
    tf.app.run()
