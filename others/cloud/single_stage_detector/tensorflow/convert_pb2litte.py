import os
import argparse
import tensorflow as tf ## tf.1.12

def parse_args():
    parser = argparse.ArgumentParser(description="Tensorflow pb to tflite convert tools.")
    parser.add_argument('--pb', '-p', type=str, 
            default='resnet34_tf.22.5.nhwc.pb',
            help='path to pb model.')
    parser.add_argument('--tflite', '-t', type=str,
            default='tf-ssd_resnet34_nchw.tflite',
            help='path to tflite model')
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    inputs = ["image"]
    outputs = ["ssd1200/py_cls_pred", "ssd1200/py_location_pred"]

    converter = tf.contrib.lite.TocoConverter.from_frozen_graph(args.pb, inputs, outputs)
    tflite_model=converter.convert()

    open(args.tflite, "wb").write(tflite_model)
