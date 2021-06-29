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

import json

from pathlib import Path

__doc__ = """
Define global variables used throughout the 3D UNet reference model.
"""

__all__ = [
    'CHECKSUM_INFER_FILE',
    'CHECKSUM_CALIB_FILE',
    'TARGET_CASES',
    'CALIB_CASES',
    'MEAN_VAL',
    'STDDEV_VAL',
    'MIN_CLIP_VAL',
    'MAX_CLIP_VAL',
    'PADDING_VAL',
    'TARGET_SPACING',
    'ROI_SHAPE',
    'SLIDE_OVERLAP_FACTOR',
]

# file pointers and sanity checks
INFERENCE_CASE_FILE = Path(
    Path.cwd(), 'meta', 'inference_cases.json').absolute()
CALIBRATION_CASE_FILE = Path(
    Path.cwd(), 'meta', 'calibration_cases.json').absolute()
CHECKSUM_INFER_FILE = Path(
    Path.cwd(), 'meta', 'checksum_inference.json').absolute()
CHECKSUM_CALIB_FILE = Path(
    Path.cwd(), 'meta', 'checksum_calibration.json').absolute()
assert INFERENCE_CASE_FILE.is_file(), 'inference_cases.json is not found'
assert CALIBRATION_CASE_FILE.is_file(), 'calibration_cases.json is not found'
assert CHECKSUM_INFER_FILE.is_file(), 'checksum_inference.json is not found'
assert CHECKSUM_CALIB_FILE.is_file(), 'checksum_calibration.json is not found'

# cases used for inference and calibration
TARGET_CASES = json.load(open(INFERENCE_CASE_FILE))
CALIB_CASES = json.load(open(CALIBRATION_CASE_FILE))

# constants used preprocessing images as well as sliding window inference
MEAN_VAL = 101.0
STDDEV_VAL = 76.9
MIN_CLIP_VAL = -79.0
MAX_CLIP_VAL = 304.0
PADDING_VAL = -2.2
TARGET_SPACING = [1.6, 1.2, 1.2]
ROI_SHAPE = [128, 128, 128]
SLIDE_OVERLAP_FACTOR = 0.5
assert isinstance(TARGET_SPACING, list) and \
    len(TARGET_SPACING) == 3 and any(TARGET_SPACING), \
    f"Need proper target spacing: {TARGET_SPACING}"
assert isinstance(ROI_SHAPE, list) and len(ROI_SHAPE) == 3 and any(ROI_SHAPE), \
    f"Need proper ROI shape: {ROI_SHAPE}"
assert isinstance(SLIDE_OVERLAP_FACTOR, float) and \
    SLIDE_OVERLAP_FACTOR > 0 and SLIDE_OVERLAP_FACTOR < 1, \
    f"Need sliding window overlap factor in (0,1): {SLIDE_OVERLAP_FACTOR}"
