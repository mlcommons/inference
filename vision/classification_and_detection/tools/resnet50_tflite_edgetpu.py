import argparse
import os
import cv2
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, decode_predictions, preprocess_input


def main(path_to_mlperf_calib_dataset):
    # load tf model
    model = ResNet50(weights='imagenet')

    # read jpeg files from mlperf calib dataset
    jpeg_files = []
    for filename in os.listdir(path_to_mlperf_calib_dataset):
        if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
            jpeg_files.append(os.path.join(path_to_mlperf_calib_dataset, filename))
    
    # qunatization
    def representative_data_gen():
        for img_path in jpeg_files:
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            yield [x]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    # Ensure that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    # full quant
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model_quant = converter.convert()
    with open('resnet50_quant_full_mlperf.tflite', 'wb') as f:
        f.write(tflite_model_quant)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', type=str, default=None)
    args = parser.parse_args()
    print(args)
    if args.image_dir is None or not os.path.exists(args.image_dir):
        raise ValueError("Please provide a calibration dataset.")
    main(args.image_dir)

    # compile model for edge tpu
    os.system("edgetpu_compiler resnet50_quant_full_mlperf.tflite")