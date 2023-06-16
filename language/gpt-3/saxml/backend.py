import abc
import os
import time

import numpy as np
import sax

import mlperf_loadgen as lg
from dataset import Dataset
import threading

class SUT_base(abc.ABC):

    def __init__(
        self,
        model_path: str,
        dataset_path: str,
        spm_path: str,
        add_eos: bool = False,
        max_examples: int = None,
        perf_examples: int = None,
    ):

        self._model_path = model_path
        self._dataset_path = dataset_path
        self._spm_path = spm_path
        self._add_eos = add_eos
        self._max_examples = max_examples
        self._perf_examples = perf_examples

        print("Loading Dataset ... ")
        self.dataset = Dataset(
            dataset_path=self._dataset_path,
            spm_path=self._spm_path,
            add_eos=self._add_eos,
            total_count_override=self._max_examples,
            perf_count_override=self._perf_examples,
        )

        print("Loading model ...")
        self._model = sax.Model(self._model_path)
        self._language_model = self._model.LM()

        self.qsl = lg.ConstructQSL(
            self.dataset.count,
            self.dataset.perf_count,
            self.dataset.LoadSamplesToRam,
            self.dataset.UnloadSamplesFromRam
        )

        self.sut = lg.ConstructSUT(
            self.issue_queries,
            self.flush_queries
        )

    @abc.abstractmethod
    def issue_queries(self, query_samples):
        pass

    @abc.abstractmethod
    def inference_call(self):
        pass

    def flush_queries(self):
        pass

    def __del__(self):
        print("Finished destroying SUT.")


class SUT_Offline(SUT_base):

    def issue_queries(self, query_samples):

        print("Number of Samples in query_samples : ", len(query_samples))

        for i in range(len(query_samples)):

            index = query_samples[i].index
            input_sample = self.dataset.inputs[index]

            pred_output = self.inference_call(input_sample)

            response_array = array.array("B", pred_output)
            buffer_info = response_array.buffer_info()
            response = [lg.QuerySampleResponse(
                query_samples[i].id, buffer_info[0], buffer_info[1])]
            lg.QuerySamplesComplete(response)
            if i % 5 == 0:
                print("Completed : ", i)

    def inference_call(self, input_sample):

        print("input_sample: ", input_sample)
        pred_output_response = self._language_model.Generate(input_sample)
        return pred_output_response[0][0]


class SUT_Server(SUT_base):

    def issue_queries(self, query_samples):

        print("Number of Samples in query_samples : ", len(query_samples))

        index = query_samples[0].index
        input_sample = self.dataset.inputs[index]

        pred_output = self.inference_call(input_sample)

        response_array = array.array("B", pred_output)
        buffer_info = response_array.buffer_info()
        response = [lg.QuerySampleResponse(
            query_samples[i].id, buffer_info[0], buffer_info[1])]
        lg.QuerySamplesComplete(response)

    def inference_call(self, input_sample):

        print("input_sample: ", input_sample)
        pred_output_response = self._language_model.Generate(input_sample)
        return pred_output_response[0][0]


def get_SUT(scenario, model_path, dataset_path, spm_path):
    if scenario == "Offline":
        return SUT_Offline(model_path, dataset_path, spm_path)
    elif scenario == "Server":
        return SUT_Server(model_path, dataset_path, spm_path)
