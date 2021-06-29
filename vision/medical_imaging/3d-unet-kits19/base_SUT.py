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


import array
import numpy as np

import mlperf_loadgen as lg
import inference_utils as infu

from kits_QSL import get_kits_QSL
from global_vars import *


class BASE_3DUNET_SUT:
    """
    A class to represent baseline SUT (System Under Test) for MLPerf.
    SUT works with QSL (Query Sample Library) to locate queries from LoadGen.
    SUT works with backend (or framework such as PyTorch, ONNX-runtime, TensorFlow),
    to perform inference of the given queries and reports back to LoadGen.

    It is required to implement backend specific SUT; inheriting this baseline SUT
    enables minimal implementation of the backend specific SUT.

    Attributes
    ----------
    sut: object
        SUT in the context of LoadGen
    qsl: object
        QSL in the context of LoadGen
    model_path: str or PosixPath object
        path to the model for backend

    Methods
    -------
    to_tensor(my_array):
        transforms my_array into tensor form backend understands
    from_tensor(my_tensor):
        transforms my_tensor backend worked on into numpy friendly array
    do_infer(input_tensor):
        performs backend specific inference upon input_tensor
    infer_single_query(data, mystr):
        performs inference upon data and summarize work in mystr
    issue_queries(query_samples):
        LoadGen calls this with query_samples, a vector containing series of queries
        SUT is to perform each query and calls back to LoadGen with QuerySamplesComplete()
    flush_queries():
        not used
    process_latencies(latencies_ns):
        not used
    """

    def __init__(self, preprocessed_data_dir, performance_count):
        """
        Constructs all the necessary attributes for ONNX Runtime specific 3D UNet SUT
        Baseline SUT doesn't instantiate any 3D UNet model; backend specific SUT needs
        to properly instantiate model

        Parameters
        ----------
            preprocessed_data_dir: str or PosixPath
                path to directory containing preprocessed data
            performance_count: int
                number of query samples guaranteed to fit in memory
        """
        print("Constructing SUT...")
        self.sut = lg.ConstructSUT(
            self.issue_queries, self.flush_queries, self.process_latencies)
        print("Finished constructing SUT.")
        self.qsl = get_kits_QSL(preprocessed_data_dir, performance_count)

    def to_tensor(self, my_array):
        """
        Transforms my_array into tensor form backend understands
        Implementation may differ as backend's need
        """
        return my_array

    def from_tensor(self, my_tensor):
        """
        Transforms my_tensor backend worked on into numpy friendly array
        Implementation may differ as backend's need
        """
        return my_tensor

    def do_infer(self, input_tensor):
        """
        Performs backend specific inference upon input_tensor
        Need specific implementation for the backend
        """
        assert False, "Please implement!"

    @infu.runtime_measure
    def infer_single_query(self, query, mystr):
        """
        Performs inference upon data and summarize work with mystr
        Naive implementation of sliding window inference on sub-volume for predetermined
        ROI (Region of Interest) shape is handled here

        Parameters
        ----------
            query: object
                Query sent by LoadGen
            mystr: str
                String that summarizes the work done; can be updated
        """
        # prepare arrays
        image = query[np.newaxis, ...]
        result, norm_map, norm_patch = infu.prepare_arrays(image, ROI_SHAPE)
        t_image, t_result, t_norm_map, t_norm_patch =\
            self.to_tensor(image), self.to_tensor(result), self.to_tensor(
                norm_map), self.to_tensor(norm_patch)

        # sliding window inference
        subvol_cnt = 0
        for i, j, k in infu.get_slice_for_sliding_window(t_image, ROI_SHAPE, SLIDE_OVERLAP_FACTOR):
            subvol_cnt += 1
            result_slice = t_result[
                ...,
                i:(ROI_SHAPE[0] + i),
                j:(ROI_SHAPE[1] + j),
                k:(ROI_SHAPE[2] + k)]

            input_slice = t_image[
                ...,
                i:(ROI_SHAPE[0] + i),
                j:(ROI_SHAPE[1] + j),
                k:(ROI_SHAPE[2] + k)]

            norm_map_slice = t_norm_map[
                ...,
                i:(ROI_SHAPE[0] + i),
                j:(ROI_SHAPE[1] + j),
                k:(ROI_SHAPE[2] + k)]

            result_slice += self.do_infer(input_slice) * t_norm_patch

            norm_map_slice += t_norm_patch

        result, norm_map = self.from_tensor(
            t_result), self.from_tensor(t_norm_map)

        final_result = infu.finalize(result, norm_map)
        mystr += ", {:3} sub-volumes".format(subvol_cnt)
        return final_result, mystr

    def issue_queries(self, query_samples):
        """
        LoadGen calls this with query_samples, a vector containing series of queries
        SUT is to perform each query and calls back to LoadGen with QuerySamplesComplete()

        Parameters
        ----------
            query_samples: object
                vector object that holds queries from LoadGen
        """
        total = len(query_samples)
        for qsi in range(total):
            query = self.qsl.get_features(query_samples[qsi].index)
            mystr = "{:5d}/{:5d} -- Processing sample id {:2d} with shape = {:}".format(
                    qsi+1, total, query_samples[qsi].index, query.shape)
            final_result, mystr = self.infer_single_query(query, mystr)
            if mystr:
                print(mystr)
            response_array = array.array("B", final_result.tobytes())
            bi = response_array.buffer_info()
            response = lg.QuerySampleResponse(
                query_samples[qsi].id, bi[0], bi[1])
            lg.QuerySamplesComplete([response])

    def flush_queries(self):
        """
        Unused; LoadGen requires override
        """
        pass

    def process_latencies(self, latencies_ns):
        """
        Unused; LoadGen requires override
        """
        pass
