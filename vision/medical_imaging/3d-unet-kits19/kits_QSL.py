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

import pickle

from pathlib import Path

import mlperf_loadgen as lg


class KiTS_2019_QSL:
    """
    A class to represent QSL (Query Sample Library) for MLPerf.

    This populates preprocessed KiTS19 inference data set into LoadGen compatible QSL 

    Attributes
    ----------
    preprocessed_data_dir: str
        path to directory containing preprocessed data 
    preprocessed_files: list of str
        list of KiTS19 cases that are used in inference
    count: int
        total number of KiTS19 cases used in inference
    perf_count: int
        number of KiTS19 cases (or query samples) guaranteed to fit in memory

    Methods
    -------
    load_query_samples(sample_list):
        opens preprocessed files (or query samples) and loads them into memory
    unload_query_samples(self, sample_list):
        deletes loaded query samples from memory
    get_features(self, sample_id):
        picks one sample with sample_id from memory and returns it
    """

    def __init__(self, preprocessed_data_dir, perf_count):
        """
        Constructs all the necessary attributes for QSL

        Parameters
        ----------
            preprocessed_data_dir: str or PosixPath
                path to directory containing preprocessed data
            perf_count: int
                number of query samples guaranteed to fit in memory
        """
        print("Constructing QSL...")
        self.preprocessed_data_dir = preprocessed_data_dir
        with open(Path(self.preprocessed_data_dir, "preprocessed_files.pkl"), "rb") as f:
            self.preprocess_files = pickle.load(f)['file_list']

        self.count = len(self.preprocess_files)
        self.perf_count = perf_count if perf_count is not None else self.count
        print("Found {:d} preprocessed files".format(self.count))
        print("Using performance count = {:d}".format(self.perf_count))

        self.loaded_files = {}
        self.qsl = lg.ConstructQSL(
            self.count, self.perf_count, self.load_query_samples, self.unload_query_samples)
        print("Finished constructing QSL.")

    def load_query_samples(self, sample_list):
        """
        Opens preprocessed files (or query samples) and loads them into memory
        """
        for sample_id in sample_list:
            file_name = self.preprocess_files[sample_id]
            print("Loading file {:}".format(file_name))
            with open(Path(self.preprocessed_data_dir, "{:}.pkl".format(file_name)), "rb") as f:
                self.loaded_files[sample_id] = pickle.load(f)[0]

    def unload_query_samples(self, sample_list):
        """
        Deletes loaded query samples from memory
        """
        for sample_id in sample_list:
            del self.loaded_files[sample_id]

    def get_features(self, sample_id):
        """
        Picks one sample with sample_id from memory and returns it
        """
        return self.loaded_files[sample_id]


def get_kits_QSL(preprocessed_data_dir="build/preprocessed_data", perf_count=None):
    return KiTS_2019_QSL(preprocessed_data_dir, perf_count)
