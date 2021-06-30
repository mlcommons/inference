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

from base_SUT import BASE_3DUNET_SUT

import onnxruntime


class _3DUNET_ONNXRuntime_SUT(BASE_3DUNET_SUT):
    """
    A class to represent SUT (System Under Test) for MLPerf.
    This inherits BASE_3DUNET_SUT and builds functionality for ONNX Runtime.

    Attributes
    ----------
    sut: object
        SUT in the context of LoadGen
    qsl: object
        QSL in the context of LoadGen
    model_path: str or PosixPath object
        path to the model for ONNX Runtime
    sess: object
        ONNX runtime session instance that does inference

    Methods
    -------
    do_infer(input_tensor):
        Perform inference upon input_tensor with ONNX Runtime
    """

    def __init__(self, model_path, preprocessed_data_dir, performance_count):
        """
        Constructs all the necessary attributes for ONNX Runtime specific 3D UNet SUT

        Parameters
        ----------
            model_path: str or PosixPath object
                path to the model for ONNX Runtime
            preprocessed_data_dir: str or PosixPath
                path to directory containing preprocessed data
            performance_count: int
                number of query samples guaranteed to fit in memory            
        """
        super().__init__(preprocessed_data_dir, performance_count)
        print("Loading ONNX model...")
        assert Path(model_path).is_file(
        ), "Cannot find the model file {:}!".format(model_path)
        self.sess = onnxruntime.InferenceSession(model_path)

    def do_infer(self, input_tensor):
        """
        Perform inference upon input_tensor with ONNX Runtime
        """
        return self.sess.run(["output"], {"input": input_tensor})[0].squeeze(0)


def get_sut(model_path, preprocessed_data_dir, performance_count):
    """
    Redirect the call for instantiating SUT to ONNX Runtime specific SUT
    """
    return _3DUNET_ONNXRuntime_SUT(model_path, preprocessed_data_dir, performance_count)
