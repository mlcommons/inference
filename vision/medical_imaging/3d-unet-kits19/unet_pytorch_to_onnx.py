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


import os
import sys
sys.path.insert(0, os.getcwd())

import argparse
import torch

from pathlib import Path

from global_vars import *



__doc__ = """
Converts PyTorch/TorchScript file to ONNX files, for dynamic batchsize and explicit batchsize

Command:
    python3 unet_pytorch_to_onnx.py
    or
    python3 unet_pytorch_to_onnx.py --model $(PYTORCH_MODEL_FILE)
                                    --output_dir $(DIR_TO_STORE_ONNX)
                                    --output_name $(ONNX_FILENAME)
                                    --dynamic_bs_output_name $(ONNX_DYN_BS_FILENAME)

Ex) --output_dir build/model --output_name 3dunet_kits19_128x128x128.onnx produces:
./build/model
└── 3dunet_kits19_128x128x128.onnx
"""


def get_args():
    """
    Args used for converting PyTorch/TorchScript to ONNX model
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--model",
                        default="build/model/3dunet_kits19_pytorch.ptc",
                        help="Path to the PyTorch model")
    parser.add_argument("--output_name",
                        default="3dunet_kits19_128x128x128.onnx",
                        help="Name of output model")
    parser.add_argument("--dynamic_bs_output_name",
                        default="3dunet_kits19_128x128x128_dynbatch.onnx",
                        help="Name of output model")
    parser.add_argument("--output_dir",
                        default="build/model",
                        help="Directory to save output model")

    args = parser.parse_args()

    return args


def main():
    """
    Converts PyTorch/TorchScript file to ONNX files, for dynamic batchsize and explicit batchsize
    """
    args = get_args()

    print("Converting PyTorch model to ONNX...")

    output_dir_path = Path(args.output_dir).absolute()
    output_dir_path.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model).absolute()

    output_path = Path(args.output_dir, args.output_name).absolute()
    dynamic_bs_output_path = Path(
        args.output_dir, args.dynamic_bs_output_name).absolute()

    print("Loading PyTorch model...")
    assert Path(model_path).is_file(
    ), "Cannot find the model file {:}!".format(model_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = torch.jit.load(model_path, map_location=device)
    model.eval()

    batchsize = 1
    input_channels = 1
    output_channels = 3
    depth, height, width = ROI_SHAPE

    dummy_input = torch.rand(
        [batchsize, input_channels, height, width, depth]).float().to(device)
    dummy_output = torch.rand(
        [batchsize, output_channels, height, width, depth]).float().to(device)

    # using opset version 12
    torch.onnx.export(model, dummy_input, output_path, opset_version=12,
                      do_constant_folding=False, input_names=['input'], output_names=['output'],
                      example_outputs=dummy_output)

    torch.onnx.export(model, dummy_input, dynamic_bs_output_path, opset_version=12,
                      do_constant_folding=False, input_names=['input'], output_names=['output'],
                      dynamic_axes={"input": {0: "batch_size"},
                                    "output": {0: "batch_size"}},
                      example_outputs=dummy_output)

    print("Successfully exported model:\n  {}\nand\n  {}".format(
        output_path, dynamic_bs_output_path))


if __name__ == "__main__":
    main()
