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


import numpy as np
from pathlib import Path

from base_SUT import BASE_3DUNET_SUT

import torch


class _3DUNET_PyTorch_SUT(BASE_3DUNET_SUT):
    """
    A class to represent SUT (System Under Test) for MLPerf.
    This inherits BASE_3DUNET_SUT and builds functionality for PyTorch/TorchScript.

    Attributes
    ----------
    sut: object
        SUT in the context of LoadGen
    qsl: object
        QSL in the context of LoadGen
    model_path: str or PosixPath object
        path to the model for PyTorch/TorchScript
    model: object
        PyTorch/TorchScript model instance that does inference
    device: str
        string describing the device PyTorch/TorchScript backend will run on

    Methods
    -------
    to_tensor(my_array):
        Transform numpy array into Torch tensor
    from_tensor(my_tensor):
        Transform Torch tensor into numpy array
    do_infer(input_tensor):
        Perform inference upon input_tensor with PyTorch/TorchScript
    """

    def __init__(self, model_path, preprocessed_data_dir, performance_count):
        """
        Constructs all the necessary attributes for PyTorch/TorchScript specific 3D UNet SUT

        Parameters
        ----------
            model_path: str or PosixPath object
                path to the model for PyTorch/TorchScript
            preprocessed_data_dir: str or PosixPath
                path to directory containing preprocessed data
            performance_count: int
                number of query samples guaranteed to fit in memory                
        """
        super().__init__(preprocessed_data_dir, performance_count)
        print("Loading PyTorch model...")
        assert Path(model_path).is_file(
        ), "Cannot find the model file {:}!".format(model_path)
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

    def do_infer(self, input_tensor):
        """
        Perform inference upon input_tensor with PyTorch/TorchScript
        """
        with torch.no_grad():
            return self.model(input_tensor)

    def to_tensor(self, my_array):
        """
        Transform numpy array into Torch tensor
        """
        return torch.from_numpy(my_array).float().to(self.device)

    def from_tensor(self, my_tensor):
        """
        Transform Torch tensor into numpy array
        """
        return my_tensor.cpu().numpy().astype(np.float)


def get_sut(model_path, preprocessed_data_dir, performance_count):
    """
    Redirect the call for instantiating SUT to PyTorch/TorchScript specific SUT
    """
    return _3DUNET_PyTorch_SUT(model_path, preprocessed_data_dir, performance_count)
