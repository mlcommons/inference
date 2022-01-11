#! /usr/bin/env python3
# coding=utf-8
# Copyright (c) 2021 NVIDIA CORPORATION. All rights reserved.
# Copyright 2021 The MLPerf Authors. All Rights Reserved.
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


import argparse
import json
import pickle
import numpy as np
import nibabel as nib
import pandas as pd

from multiprocessing import Pool
from pathlib import Path


__doc__ = """
Check accuracy of inference performed on KiTS19 dataset.

This script will compare the segmentation results from inference with ground truth segmentation information.
- Segmentation results from inference are generated from MLPerf-Inference log file $(LOG_DIR)/mlperf_log_accuracy.json
  and are stored into $(POSTPROCESSED_DATA_DIR)/case_XXXXX/prediction.nii.gz
- Ground truth segmentation data are stored in $(PREPROCESSED_DATA_DIR)/nifti/case_XXXXX/segmentation.nii.gz
- DICE scores are calculated for segmentation on 1) kidney and 2) tumor, and the mean of the two (AKA composite score)
- Overall (or mean) DICE scores over all the predicted cases are reported at the end
- Individual scores are stored in $(POSTPROCESSED_DATA_DIR)/summary.csv

Accuracy check from MLPerf-Inference accuracy log file:
    python3 accuracy_kits.py
    or
    python3 accuracy_kits.py --log_file $(LOG_DIR)/$(ACCURACY_LOG_FILENAME) 
                             --output_dtype $(DTYPE)
                             --preprocessed_data_dir $(PREPROCESSED_DATA_DIR) 
                             --postprocessed_data_dir $(POSTPROCESSED_DATA_DIR)
                             --num_proc $(NUMBER_PROCESSES) 
"""

# $(DTYPE) mapping to numpy dtype
dtype_map = {
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64
}


def get_args():
    """
    Args used for postprocessing
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--log_file",
                        default="build/logs/mlperf_log_accuracy.json",
                        help="Path to accuracy log json file")
    parser.add_argument("--output_dtype",
                        default="uint8",
                        choices=dtype_map.keys(),
                        help="Output data type")
    parser.add_argument("--preprocessed_data_dir",
                        default="build/preprocessed_data",
                        help="Path to the directory containing preprocessed data")
    parser.add_argument("--postprocessed_data_dir",
                        default="build/postprocessed_data",
                        help="Path to the directory containing postprocessed data")
    parser.add_argument("--num_proc",
                        type=int,
                        default=4,
                        help="Number of processors running postprocessing")
    args = parser.parse_args()
    return args


def to_one_hot(my_array, channel_axis):
    """
    Changes class information into one-hot encoded information
    Number of classes in KiTS19 is 3: background, kidney segmentation, tumor segmentation
    As a result, 1 channel of class info turns into 3 channels of one-hot info
    """
    my_array = prepare_one_hot(my_array, num_classes=3)
    my_array = np.transpose(my_array, (0, 4, 1, 2, 3)).astype(np.float64)
    return my_array


def prepare_one_hot(my_array, num_classes):
    """
    Reinterprets my_array into one-hot encoded, for classes as many as num_classes
    """
    res = np.eye(num_classes)[np.array(my_array).reshape(-1)]
    return res.reshape(list(my_array.shape)+[num_classes])


def get_dice_score(case, prediction, target):
    """
    Calculates DICE score of prediction against target, for classes as many as case
    One-hot encoded form of case/prediction used for easier handling
    Background case is not important and hence removed
    """
    # constants
    channel_axis = 1
    reduce_axis = (2, 3, 4)
    smooth_nr = 1e-6
    smooth_dr = 1e-6

    # apply one-hot
    prediction = to_one_hot(prediction, channel_axis)
    target = to_one_hot(target, channel_axis)

    # remove background
    target = target[:, 1:]
    prediction = prediction[:, 1:]

    # calculate dice score
    assert target.shape == prediction.shape, \
        f"Different shape -- target: {target.shape}, prediction: {prediction.shape}"
    assert target.dtype == np.float64 and prediction.dtype == np.float64, \
        f"Unexpected dtype -- target: {target.dtype}, prediction: {prediction.dtype}"

    # intersection for numerator; target/prediction sum for denominator
    # easy b/c one-hot encoded format
    intersection = np.sum(target * prediction, axis=reduce_axis)
    target_sum = np.sum(target, axis=reduce_axis)
    prediction_sum = np.sum(prediction, axis=reduce_axis)

    # get DICE score for each class
    dice_val = (2.0 * intersection + smooth_nr) / \
        (target_sum + prediction_sum + smooth_dr)

    # return after removing batch dim
    return (case, dice_val[0])


def evaluate(target_files, preprocessed_data_dir, postprocessed_data_dir, num_proc):
    """
    Collects and summarizes DICE scores of all the predicted files using multi-processes
    """
    bundle = list()

    for case in target_files:
        groundtruth_path = Path(preprocessed_data_dir,
                                "nifti", case, "segmentation.nii.gz").absolute()
        prediction_path = Path(postprocessed_data_dir,
                               case, "prediction.nii.gz").absolute()

        groundtruth = nib.load(groundtruth_path).get_fdata().astype(np.uint8)
        prediction = nib.load(prediction_path).get_fdata().astype(np.uint8)

        groundtruth = np.expand_dims(groundtruth, 0)
        prediction = np.expand_dims(prediction, 0)

        assert groundtruth.shape == prediction.shape,\
            "{} -- groundtruth: {} and prediction: {} have different shapes".format(
                case, groundtruth.shape, prediction.shape)

        bundle.append((case, groundtruth, prediction))

    with Pool(num_proc) as p:
        dice_scores = p.starmap(get_dice_score, bundle)

    save_evaluation_summary(postprocessed_data_dir, dice_scores)


def save_evaluation_summary(postprocessed_data_dir, dice_scores):
    """
    Stores collected DICE scores in CSV format: $(POSTPROCESSED_DATA_DIR)/summary.csv
    """
    sum_path = Path(postprocessed_data_dir, "summary.csv").absolute()
    df = pd.DataFrame()

    for _s in dice_scores:
        case, arr = _s
        kidney = arr[0]
        tumor = arr[1]
        composite = np.mean(arr)
        df = df.append(
            {
                "case": case,
                "kidney": kidney,
                "tumor": tumor,
                "composite": composite
            }, ignore_index=True)

    df.set_index("case", inplace=True)
    # consider NaN as a crash hence zero
    df.loc["mean"] = df.fillna(0).mean()

    df.to_csv(sum_path)


def save_nifti(bundle):
    """
    Saves single segmentation result from inference into NIFTI file
    """
    # Note that affine has to be valid, otherwise NIFTI image will look weird
    image, affine, path_to_file = bundle
    if len(image.shape) != 3:
        assert len(image.shape) == 4 and image.shape[0] == 1,\
            "Unexpected image: {}".format(image.shape)
        image = np.squeeze(image, 0)
    nifti_image = nib.Nifti1Image(image, affine=affine)
    path_to_file.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nifti_image, path_to_file)


def save_predictions(predictions, output_dir, preprocessed_data_dir,
                     preprocessed_files, aux, num_proc):
    """
    Saves all the segmentation result from inference into NIFTI files using affine matrices
    Affine matrices were stored for input images during preprocessing
    NIFTI files stored as $(POSTPROCESSED_DATA)/case_XXXX/prediction.nii.gz
    """
    print("Saving predictions...")
    bundle = list()
    for case, case_d in predictions.items():
        pred_file_path = Path(output_dir, case, "prediction.nii.gz")
        bundle.append((case_d['prediction'], aux[case]
                       ['reshaped_affine'], pred_file_path))

    with Pool(num_proc) as p:
        p.map(save_nifti, bundle)

    p.join()
    p.close()


def load_loadgen_log(log_file, result_dtype, file_list, aux):
    """
    Loads accuracy log produced by LoadGen
    Accuracy log has inference results in bitstream; needs split for each case using shape/dtype
    Format is assumed to be linear
    """
    with open(log_file) as f:
        predictions = json.load(f)

    assert len(predictions) == len(aux.keys()),\
        "Number of predictions does not match number of samples in validation set!"

    results = dict()
    for prediction in predictions:
        qsl_idx = prediction["qsl_idx"]
        case = file_list[qsl_idx]
        assert qsl_idx >= 0 and qsl_idx < len(predictions), "Invalid qsl_idx!"
        result_shape = np.array(list(aux[case]["image_shape"]))
        result = np.frombuffer(bytes.fromhex(
            prediction["data"]), result_dtype).reshape(result_shape)
        results[case] = {
            'qsl_idx': qsl_idx,
            'prediction': result
        }

    assert len(results) == len(predictions), "Missing some results!"

    return results


def main():
    """
    Compares the segmentation results from inference with ground truth segmentation information
    Inference results are obtained by translating accuracy log generated by LoadGen
    Collects/reports DICE scores for Kidney segmentation, Tumor segmentation and composite (mean)
    """
    args = get_args()
    log_file = args.log_file
    preprocessed_data_dir = args.preprocessed_data_dir
    postprocessed_data_dir = args.postprocessed_data_dir
    output_dtype = dtype_map[args.output_dtype]
    num_proc = args.num_proc

    # Load necessary metadata.
    print("Loading necessary metadata...")
    with open(Path(preprocessed_data_dir, "preprocessed_files.pkl"), "rb") as f:
        preprocessed_files_content = pickle.load(f)
    target_files = preprocessed_files_content['file_list']
    aux = preprocessed_files_content['cases']

    # Load predictions from loadgen accuracy log.
    print("Loading loadgen accuracy log...")
    predictions = load_loadgen_log(log_file, output_dtype, target_files, aux)

    # Save predictions
    print("Running postprocessing...")
    save_predictions(predictions, postprocessed_data_dir, preprocessed_data_dir,
                     target_files, aux, num_proc)

    # Run evaluation
    print("Running evaluation...")
    evaluate(target_files, preprocessed_data_dir,
             postprocessed_data_dir, num_proc)

    # Finalize evaluation from evaluation summary
    print("Processing evaluation summary...")
    df = pd.read_csv(Path(postprocessed_data_dir, "summary.csv"))
    final = df.loc[df['case'] == 'mean']
    composite = float(final['composite'])
    kidney = float(final['kidney'])
    tumor = float(final['tumor'])
    print("Accuracy: mean = {:.5f}, kidney = {:.4f}, tumor = {:.4f}".format(
          composite, kidney, tumor))
    print("Done!")


if __name__ == "__main__":
    main()
