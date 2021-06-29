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


from pathlib import Path
import onnx_tf
import onnx
import argparse
import os
import sys
sys.path.insert(0, os.getcwd())


__doc__ = """
Converts ONNX file to TensorFlow saved_model bundle.

Command:
    python3 unet_onnx_to_tensorflow.py
    or
    python3 unet_onnx_to_tensorflow.py --model $(ONNX_MODEL)
                                       --output_dir $(DIR_TO_STORE_TF_SAVED_MODEL)
                                       --output_name $(TF_SAVED_MODEL_DIR_NAME)

Ex) --output_dir build/model --output_name 3dunet_kits19_128x128x128.tf produces:
./build/model/3dunet_kits19_128x128x128.tf/
├── assets
├── saved_model.pb
└── variables
    ├── variables.data-00000-of-00001
    └── variables.index
"""


def get_args():
    """
    Args used for converting ONNX to TensorFlow model
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--model",
                        default="build/model/3dunet_kits19_128x128x128.onnx",
                        help="Path to the ONNX model")
    parser.add_argument("--output_name",
                        default="3dunet_kits19_128x128x128.tf",
                        help="Name of output model")
    parser.add_argument("--output_dir",
                        default="build/model",
                        help="Directory to save output model")

    args = parser.parse_args()

    return args


def main():
    """
    Converts ONNX file to TensorFlow saved_model bundle.
    """
    args = get_args()

    print("Loading ONNX model...")
    onnx_model = onnx.load(args.model)

    print("Converting ONNX model to TF...")
    tf_model = onnx_tf.backend.prepare(onnx_model)

    output_dir_path = Path(args.output_dir).absolute()
    output_dir_path.mkdir(parents=True, exist_ok=True)

    output_path = Path(output_dir_path / args.output_name).absolute()

    tf_model.export_graph(str(output_path))

    print("Successfully exported model {}".format(output_path))


if __name__ == "__main__":
    main()
