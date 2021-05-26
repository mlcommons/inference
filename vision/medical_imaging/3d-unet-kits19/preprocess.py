#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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


import os
import argparse
import hashlib
import json
import pickle
from multiprocessing import Process, Pool

import nibabel
import numpy as np

from scipy.ndimage.interpolation import zoom
from pathlib import Path

from global_vars import *


__doc__ = """
Takes KiTS19 RAW data, returns reshaped data with the same voxel spacing.

Preprocess dataset that is used for inference:
    python3 preprocess.py --raw_data_dir $(RAW_DATA_DIR) --results_dir $(PREPROCESSED_DATA_DIR) --mode preprocess

Preprocess dataset that is used for calibration:
    python3 preprocess.py --raw_data_dir $(RAW_DATA_DIR) --results_dir $(PREPROCESSED_DATA_DIR) --mode preprocess --calibration

(Re)generate MD5 hashes for data integrity check on inference dataset
    python3 preprocess.py --raw_data_dir $(RAW_DATA_DIR) --results_dir $(PREPROCESSED_DATA_DIR) --mode gen-hash

(Re)generate MD5 hashes for data integrity check on calibration dataset
    python3 preprocess.py --raw_data_dir $(RAW_DATA_DIR) --results_dir $(PREPROCESSED_DATA_DIR) --mode gen-hash --calibration

Verify MD5 hashes stored from original run for data integrity check on inference dataset
    python3 preprocess.py --raw_data_dir $(RAW_DATA_DIR) --results_dir $(PREPROCESSED_DATA_DIR) --mode verify

Verify MD5 hashes stored from original run for data integrity check on calibration dataset
    python3 preprocess.py --raw_data_dir $(RAW_DATA_DIR) --results_dir $(PREPROCESSED_DATA_DIR) --mode verify --calibration

Optionally, add -num_proc=$(NUMBER_PROCESSES) to use as many processes as $(NUMBER_PROCESSES) to shorten the turnaround time
"""


class Stats:
    """
    A class collecting value distribution of preprocessed images in KiTS19 dataset

    Attributes
    ----------
    mean: list
        collects average values in the preprocessed images
    std: list
        collects standard deviation of values in the preprocessed images
    d: 
        collects depths of the preprocessed images
    h: 
        collects heights of the preprocessed images
    w: 
        collects widths of the preprocessed images

    Methods
    -------
    __init__():
        initiates all the attributes
    append(mean, std, d, h, w):
        adds datapoint (mean, std, d, h, w) to the class instance
    get_string():
        returns string summarizing collected datapoints
    """

    def __init__(self):
        """
        Initiates all the attributes

        Attributes
        ----------
        mean: list
            collects average values in the preprocessed images
        std: list
            collects standard deviation of values in the preprocessed images
        d: 
            collects depths of the preprocessed images
        h: 
            collects heights of the preprocessed images
        w: 
            collects widths of the preprocessed images
        """
        self.mean = []
        self.std = []
        self.d = []
        self.h = []
        self.w = []

    def append(self, mean, std, d, h, w):
        """
        Adds datapoint (mean, std, d, h, w) to the class instance
        """
        self.mean.append(mean)
        self.std.append(std)
        self.d.append(d)
        self.h.append(h)
        self.w.append(w)

    def get_string(self):
        """
        Returns string summarizing collected datapoints
        """
        self.mean = np.median(np.array(self.mean))
        self.std = np.median(np.array(self.std))
        self.d = np.median(np.array(self.d))
        self.h = np.median(np.array(self.h))
        self.w = np.median(np.array(self.w))

        return f"Mean value: {self.mean}, std: {self.std}, d: {self.d}, h: {self.h}, w: {self.w}"


class Preprocessor:
    """
    A class processing images in KiTS19 dataset
    Pre-processing includes below steps (128x128x128 window with 50% overlap as an example)
        1. Get a pair of CT-imaging/segmentation data
        2. Resample to the same, predetermined common voxel spacing (1.6, 1.2, 1.2)[mm]
        3. Pad every volume so it is equal or larger than 128
        4. Pad/crop volumes so they are divisible by 64
    Preprocessed data are saved as pickle format for easy consumption
    Reshaped imaging/segmentation will be saved as NIFTI as well for easy comparison with prediction

    Attributes
    ----------
    results_dir: str
        directory preprocessed data will be stored into
    data_dir: str
        directory containing KiTS19 RAW data
    calibration: bool
        flag for processing calibration set, if true, instead of inference set
    mean, std, min_val, max_val: float
        used for normalizing intensity of the input image
    padding_val: float
        padding with this value
    target_spacing: [z, y, x]
        common voxel spacing that all the CT images reshaped for
    target_shape: [d, h, w]
        ROI (Region of Interest) shape; training done on the sub-volume of this shape
    slide_overlap_factor: float
        sliding window inference will follow this overlapping factor
    stats: obj
        Stats instance that collects image statistics

    Methods
    -------
    __init__():
        initiates all the attributes
    collect_cases():
        populates cases to preprocess from attribute target_cases
    print_stats():
        prints stats of the preprocessed imaging data
    preprocess_dataset():
        performs preprocess of all the cases collected and then prints summary stats of them
    preprocess_case(case):
        picks up the case from KiTS19 RAW data and perform preprocessing:
        1. Get a pair of CT-imaging/segmentation data for the case
        2. Resample to the same, predetermined common voxel spacing
        3. Pad every volume so it is equal or larger than ROI shape
        4. Pad/Crop volumes so they are friendly to sliding window inference
        then save the preprocessed data
    pad_to_min_shape(image, label):
        pads image/label so that the shape is equal or larger than ROI shape
    load_and_resample(case):
        gets a pair of CT-imaging/segmentation data for the case, then, 
        resample to the same, predetermined common voxel spacing
    normalize_intensity(image):
        normalize intensity for a given target stats
    adjust_shape_for_sliding_window(image, label):
        pads/crops image/label volumes so that sliding window inference can easily be done
    constant_pad_volume(volume, roi_shape, strides, padding_val, dim):
        helper padding volume symmetrically with value of padding_val
        padded volume becomes ROI shape friendly
    save(image, label, aux):
        Save preprocessed imaging/segmentation data in pickle format for easy consumption
        auxiliary information also saved together that holds:
    """

    def __init__(self, args):
        """
        Initiates all the attributes

        Attributes
        ----------
        results_dir: str
            directory preprocessed data will be stored into
        data_dir: str
            directory containing KiTS19 RAW data
        calibration: bool
            flag for processing calibration set, if true, instead of inference set
        mean, std, min_val, max_val: float
            used for normalizing intensity of the input image
        padding_val: float
            padding with this value
        target_spacing: [z, y, x]
            One common voxel spacing that all the CT images reshaped for
        target_shape: [d, h, w]
            ROI (Region of Interest) shape; training done on the sub-volume of this shape
        slide_overlap_factor: float
            sliding window inference will follow this overlapping factor
        stats: obj
            Stats instance that collects image statistics

        """
        self.results_dir = args.results_dir
        self.data_dir = args.data_dir
        self.calibration = args.calibration
        self.target_cases = CALIB_CASES if self.calibration else TARGET_CASES
        self.mean = MEAN_VAL
        self.std = STDDEV_VAL
        self.min_val = MIN_CLIP_VAL
        self.max_val = MAX_CLIP_VAL
        self.padding_val = PADDING_VAL
        self.target_spacing = TARGET_SPACING
        self.target_shape = ROI_SHAPE
        self.slide_overlap_factor = SLIDE_OVERLAP_FACTOR
        self.stats = Stats()
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)

    def collect_cases(self):
        """
        Populates cases to preprocess from attribute target_cases
        """
        print(f"Preprocessing {self.data_dir}...")
        all_set = set([f for f in os.listdir(self.data_dir) if "case" in f])
        target_set = set(self.target_cases)
        collected_set = all_set & target_set
        assert collected_set == target_set,\
            "Some of the target inference cases were found: {}".format(
                target_set - collected_set)
        return sorted(list(collected_set))

    def print_stats(self):
        """
        Prints stats of the preprocessed imaging data
        """
        print(self.stats.get_string())

    def preprocess_dataset(self):
        """
        Performs preprocess of all the cases collected and then prints summary stats of them
        """
        for case in self.collect_cases():
            self.preprocess_case(case)
        self.print_stats()

    def preprocess_case(self, case):
        """
        Picks up the case from KiTS19 RAW data and perform preprocessing:
        1. Get a pair of CT-imaging/segmentation data for the case
        2. Resample to the same, predetermined common voxel spacing (1.6, 1.2, 1.2)[mm]
        3. Pad every volume so it is equal or larger than 128
        4. Pad/Crop volumes so they are divisible by 64
        Then save the preprocessed data in pickle format for easy consumption
        Reshaped imaging/segmentation will be saved as NIFTI as well for easy comparison with prediction
        """
        image, label, aux = self.load_and_resample(case)
        image = self.normalize_intensity(image.copy())
        image, label = self.pad_to_min_shape(image, label)
        image, label = self.adjust_shape_for_sliding_window(image, label)
        self.save(image, label, aux)
        aux['image_shape'] = image.shape
        return aux

    @staticmethod
    def pad_to_min_shape(image, label):
        """
        Pads every volume so it is equal or larger than ROI shape
        """
        current_shape = image.shape[1:]
        bounds = [max(0, ROI_SHAPE[i] - current_shape[i]) for i in range(3)]
        paddings = [(0, 0)]
        paddings.extend([(bounds[i] // 2, bounds[i] - bounds[i] // 2)
                         for i in range(3)])

        image = np.pad(image, paddings, mode="edge")
        label = np.pad(label, paddings, mode="edge")

        return image, label

    def load_and_resample(self, case: str):
        """
        Gets a pair of CT-imaging/segmentation data for the case
        Then, resample to the same, predetermined common voxel spacing (1.6, 1.2, 1.2)[mm]
        Also store auxiliary info for future use
        """
        aux = dict()

        image = nibabel.load(
            Path(self.data_dir, case, "imaging.nii.gz").absolute())
        label = nibabel.load(
            Path(self.data_dir, case, "segmentation.nii.gz").absolute())

        image_spacings = image.header["pixdim"][1:4].tolist()
        original_affine = image.affine

        image = image.get_fdata().astype(np.float32)
        label = label.get_fdata().astype(np.uint8)

        spc_arr = np.array(image_spacings)
        targ_arr = np.array(self.target_spacing)
        zoom_factor = spc_arr / targ_arr

        # build reshaped affine
        reshaped_affine = original_affine.copy()
        for i in range(3):
            idx = np.where(original_affine[i][:-1] != 0)
            sign = -1 if original_affine[i][idx] < 0 else 1
            reshaped_affine[i][idx] = targ_arr[idx] * sign

        if image_spacings != self.target_spacing:
            image = zoom(image, zoom_factor, order=1,
                         mode='constant', cval=image.min(), grid_mode=False)
            label = zoom(label, zoom_factor, order=0,
                         mode='constant', cval=label.min(), grid_mode=False)

        aux['original_affine'] = original_affine
        aux['reshaped_affine'] = reshaped_affine
        aux['zoom_factor'] = zoom_factor
        aux['case'] = case

        image = np.expand_dims(image, 0)
        label = np.expand_dims(label, 0)

        return image, label, aux

    def normalize_intensity(self, image: np.array):
        """
        Normalizes intensity for a given target stats
        """
        image = np.clip(image, self.min_val, self.max_val)
        image = (image - self.mean) / self.std
        return image

    def adjust_shape_for_sliding_window(self, image, label):
        """
        Pads/crops volumes so that sliding window inference can easily be done

        Sliding window of 128x128x128 to move smoothly, with overlap factor of 0.5
        then pads/crops volumes so that they are divisible by 64
        This padding or cropping is done as below:
            - if a given edge length modulo 64 is larger than 32 it is constant padded
            - if a given edge length modulo 64 is less than 32 it will be cropped
        """
        image_shape = list(image.shape[1:])
        dim = len(image_shape)
        roi_shape = self.target_shape
        overlap = self.slide_overlap_factor
        strides = [int(roi_shape[i] * (1 - overlap)) for i in range(dim)]

        bounds = [image_shape[i] % strides[i] for i in range(dim)]
        bounds = [bounds[i] if bounds[i] <
                  strides[i] // 2 else 0 for i in range(dim)]
        image = image[...,
                      bounds[0] // 2: image_shape[0] - (bounds[0] - bounds[0] // 2),
                      bounds[1] // 2: image_shape[1] - (bounds[1] - bounds[1] // 2),
                      bounds[2] // 2: image_shape[2] - (bounds[2] - bounds[2] // 2)]
        label = label[...,
                      bounds[0] // 2: image_shape[0] - (bounds[0] - bounds[0] // 2),
                      bounds[1] // 2: image_shape[1] - (bounds[1] - bounds[1] // 2),
                      bounds[2] // 2: image_shape[2] - (bounds[2] - bounds[2] // 2)]
        image, paddings = self.constant_pad_volume(
            image, roi_shape, strides, self.padding_val)
        label, paddings = self.constant_pad_volume(
            label, roi_shape, strides, 0)

        return image, label

    def constant_pad_volume(self, volume, roi_shape, strides, padding_val, dim=3):
        """
        Helper padding volume symmetrically with value of padding_val
        Padded volume becomes ROI shape friendly
        """
        bounds = [(strides[i] - volume.shape[1:][i] % strides[i]) %
                  strides[i] for i in range(dim)]
        bounds = [bounds[i] if (volume.shape[1:][i] + bounds[i]) >= roi_shape[i] else
                  bounds[i] + strides[i]
                  for i in range(dim)]
        paddings = [(0, 0),
                    (bounds[0] // 2, bounds[0] - bounds[0] // 2),
                    (bounds[1] // 2, bounds[1] - bounds[1] // 2),
                    (bounds[2] // 2, bounds[2] - bounds[2] // 2)]

        padded_volume = np.pad(
            volume, paddings, mode='constant', constant_values=[padding_val])
        return padded_volume, paddings

    def save(self, image, label, aux):
        """
        Saves preprocessed imaging/segmentation data in pickle format for easy consumption
        Auxiliary information also saved together that holds:
            - preprocessed image/segmentation shape
            - original affine matrix
            - affine matrix for reshaped imaging/segmentation upon common voxel spacing
            - zoom factor used in transform from original voxel spacing to common voxel spacing
            - case name
        Preprocessed imaging/segmentation data saved as NIFTI
        """
        case = aux['case']
        reshaped_affine = aux['reshaped_affine']
        image = image.astype(np.float32)
        label = label.astype(np.uint8)
        mean, std = np.round(np.mean(image, (1, 2, 3)), 2), np.round(
            np.std(image, (1, 2, 3)), 2)
        self.stats.append(
            mean, std, image.shape[1], image.shape[2], image.shape[3])
        pickle_file_path = Path(self.results_dir, f"{case}.pkl").absolute()
        with open(pickle_file_path, "wb") as f:
            pickle.dump([image, label], f)
        f.close()
        print(
            f"Saved {str(pickle_file_path)} -- shape {image.shape} mean {mean} std {std}")
        if not self.calibration:
            path_to_nifti_dir = Path(
                self.results_dir, "nifti", case).absolute()
            path_to_nifti_dir.mkdir(parents=True, exist_ok=True)
            nifti_image = nibabel.Nifti1Image(
                np.squeeze(image, 0), affine=reshaped_affine)
            nifti_label = nibabel.Nifti1Image(
                np.squeeze(label, 0), affine=reshaped_affine)
            nibabel.save(nifti_image, Path(
                path_to_nifti_dir / "imaging.nii.gz"))
            nibabel.save(nifti_label, Path(
                path_to_nifti_dir / "segmentation.nii.gz"))
            assert nifti_image.shape == nifti_label.shape, \
                "While saving NIfTI files to {}, image: {} and label: {} have different shape".format(
                    path_to_nifti_dir, nifti_image.shape, nifti_label.shape)


def preprocess_multiproc_helper(preproc, case):
    """
    Helps preprocessing with multi-processes
    """
    aux = preproc.preprocess_case(case)
    return aux


def save_preprocessed_info(preproc_dir, aux, targets):
    """
    Saves list of preprocessed files and the associated aux info into preprocessed_files.pkl
    """
    assert len(targets) == len(aux['cases']),\
        "Error in number of preprocessed files:\nExpected:{}\nProcessed:{}".format(
            targets, list(aux['cases'].keys()))
    with open(os.path.join(preproc_dir, 'preprocessed_files.pkl'), 'wb') as f:
        pickle.dump(aux, f)
    f.close()


def preprocess_with_multiproc(args):
    """
    Performs preprocess on KiTS19 imaging/segmentation data using multiprocesses
    """
    preproc = Preprocessor(args)
    cases = preproc.collect_cases()
    aux = {
        'file_list': preproc.target_cases,
        'cases': dict()
    }
    with Pool(args.num_proc) as p:
        pool_out = p.starmap(preprocess_multiproc_helper,
                             zip([preproc]*len(cases), cases))

    for _d in pool_out:
        aux['cases'][_d['case']] = _d
    save_preprocessed_info(preproc.results_dir, aux, preproc.target_cases)
    p.join()
    p.close()


def generate_hash_from_volume(vol_path):
    """
    Generates MD5 hash from a single preprocessed file
    """
    with open(vol_path, 'rb') as f:
        data = f.read()
        md5_hash = hashlib.md5(data).hexdigest()
    f.close()
    return (os.path.basename(vol_path), md5_hash)


def generate_hash_from_dataset(args):
    """
    Generates MD5 hash from all the preprocessed files and store them for future verification
    """
    results_dir = args.results_dir
    num_proc = args.num_proc
    checksum = dict()
    CHECKSUM_FILE = CHECKSUM_CALIB_FILE if args.calibration else CHECKSUM_INFER_FILE
    results = [f for f in os.listdir(results_dir) if f.startswith(
        'case') and f.endswith('pkl')]
    vol_path = [os.path.join(results_dir, v) for v in results]

    print(
        f"Generating checksum file checksum.json from preprocessed data in {results_dir}...")
    with Pool(num_proc) as p:
        pool_out = p.map(generate_hash_from_volume, vol_path)

    for vol, md5 in pool_out:
        checksum[vol] = md5

    with open(CHECKSUM_FILE, 'w') as f:
        json.dump(dict(sorted(checksum.items())), f, indent=4, sort_keys=True)
    f.close()

    p.join()
    p.close()
    print(f"{CHECKSUM_FILE} has been successfully created")


def verify_dataset(args):
    """
    Verifies preprocessed data's integrity by comparing MD5 hashes stored from original run
    """
    results_dir = args.results_dir
    num_proc = args.num_proc
    CHECKSUM_FILE = CHECKSUM_CALIB_FILE if args.calibration else CHECKSUM_INFER_FILE
    results = [f for f in os.listdir(results_dir) if f.startswith(
        'case') and f.endswith('pkl')]
    vol_path = [os.path.join(results_dir, v) for v in results]
    violations = dict()

    print(f"Verifying checksums of preprocessed data in {results_dir}...")
    with open(CHECKSUM_FILE) as f:
        source = json.load(f)
    f.close()
    assert len(source) == len(results),\
        "checksum.json has {} entries while {} volumes found".format(
            len(source), len(results))

    with Pool(num_proc) as p:
        pool_out = p.map(generate_hash_from_volume, vol_path)

    for vol, md5 in pool_out:
        if md5 != source[vol]:
            violations[vol] = (md5, source[vol])

    if any(violations):
        for vol, (res, ref) in violations.items():
            print(f"{vol} -- Invalid hash, got {res} while expecting {ref}")
        assert False,\
            "Verification failed, {}/{} mismatch(es) found".format(
                len(violations), len(results))

    p.join()
    p.close()
    print("Verification completed. All files' checksums match")


def parse_args():
    """
    Args used for preprocessing
    """
    PARSER = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)

    PARSER.add_argument('--raw_data_dir',
                        dest='data_dir',
                        required=True,
                        help="Dir where KiTS19 GIT repo is cloned")
    PARSER.add_argument('--results_dir',
                        dest='results_dir',
                        required=True,
                        help="Dir to store preprocessed data")
    PARSER.add_argument('--mode',
                        dest='mode',
                        choices=["preprocess", "verify", "gen_hash"],
                        default="preprocess",
                        help="""preprocess for generating inference dataset, 
                                gen_hash for generating new checksum file, 
                                verify for verifying the checksums against stored checksum file""")
    PARSER.add_argument('--calibration',
                        dest='calibration',
                        action='store_true',
                        help="Preprocess calibration dataset instead of inference dataset")
    PARSER.add_argument('--num_proc',
                        dest='num_proc',
                        type=int,
                        choices=list(range(1, 17)),
                        default=4,
                        help="Number of processes to be used")

    args = PARSER.parse_args()

    return args


def main():
    """
    Runs preprocess, verify integrity or regenerate MD5 hashes
    """
    args = parse_args()

    if args.mode == "preprocess":
        preprocess_with_multiproc(args)
        verify_dataset(args)

    if args.mode == "gen_hash":
        generate_hash_from_dataset(args)

    if args.mode == "verify":
        verify_dataset(args)


if __name__ == '__main__':
    main()
