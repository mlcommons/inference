import abc
import os
import time
import array
import random
import threading

import numpy as np

import status
import sax

import mlperf_loadgen as lg
from dataset import Dataset


class SUT_base(abc.ABC):

    def __init__(
        self,
        model_path: str,
        dataset_path: str,
        batch_size: int = 1,
        max_examples: int = None,
        perf_examples: int = None,
        log_interval: int = 100,
    ):

        self._model_path = model_path
        self._dataset_path = dataset_path
        self._batch_size = batch_size
        self._max_examples = max_examples
        self._perf_examples = perf_examples
        self._log_interval = log_interval

        self._batch_pred_outputs = []

        self._completed_issue_queries = 0
        self._completed_query_samples = 0

        print("Loading Dataset ... ")
        self.dataset = Dataset(
            dataset_path=self._dataset_path,
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

    def flush_queries(self):
        pass

    def __del__(self):
        print("Finished destroying SUT.")

    def inference_call(self, input_sample):

        try:
            pred_output_response = self._language_model.Generate(input_sample)
            pred_output_str = pred_output_response[0][0]
            pred_output = np.fromstring(pred_output_str, dtype=int, sep=',')
            return pred_output

        except status.StatusNotOk as e:
            logging.info(e)


class SUT_Offline(SUT_base):

    def _batch_inference_sync(self, batch_query_samples, i):

        sample = batch_query_samples[i]
        input_sample = self.dataset.inputs_str[sample.index]
        pred_output = self.inference_call(input_sample)

        self._batch_pred_outputs.append((sample.id, pred_output))

    def _process_batch_queries(self, batch_query_samples):
        threads = []
        for i in range(len(batch_query_samples)):
            t = threading.Thread(target=self._batch_inference_sync, args=(batch_query_samples, i))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()


    def issue_queries(self, query_samples):

        num_issue_queries = self._completed_issue_queries + 1
        num_query_samples = len(query_samples)
        print(f"The issued query {num_issue_queries} has {num_query_samples} query samples.")

        for i_batch in range(0, num_query_samples, self._batch_size):

            batch_query_samples = query_samples[i_batch:i_batch+self._batch_size]
            self._batch_pred_outputs = []
            self._process_batch_queries(batch_query_samples)
            for sample_id, pred_output in self._batch_pred_outputs:
                response_array = array.array("B", pred_output.tobytes())
                buffer_info = response_array.buffer_info()
                response = lg.QuerySampleResponse(
                    sample_id, buffer_info[0], buffer_info[1])
                lg.QuerySamplesComplete([response])

        self._completed_issue_queries = num_issue_queries
        self._completed_query_samples += num_query_samples

        if self._completed_query_samples % self._log_interval == 0:
            print(f"Total completed {self._completed_issue_queries} issue queries and {self._completed_query_samples} query samples.")


class SUT_Server(SUT_base):

    def issue_queries(self, query_samples):

        num_issue_queries = self._completed_issue_queries + 1
        num_query_samples = len(query_samples)

        sample = query_samples[0]

        index = sample.index
        input_sample = self.dataset.inputs_str[index]
        pred_output = self.inference_call(input_sample)
        pred_output_bytes = pred_output.tobytes()
        response_array = array.array("B", pred_output_bytes)
        buffer_info = response_array.buffer_info()
        response = lg.QuerySampleResponse(
            sample.id, buffer_info[0], buffer_info[1])

        lg.QuerySamplesComplete([response])

        self._completed_issue_queries = num_issue_queries
        self._completed_query_samples += num_query_samples

        if self._completed_query_samples % self._log_interval == 0:
            print(f"Total completed {self._completed_issue_queries} issue queries and {self._completed_query_samples} query samples.")


def get_SUT(
    scenario: str,
    model_path: str,
    dataset_path: str,
    batch_size: int,
    max_examples: int,
    perf_examples: int,
    log_interval: int
):
    if scenario == "Offline":
        return SUT_Offline(
            model_path=model_path,
            dataset_path=dataset_path,
            batch_size=batch_size,
            max_examples=max_examples,
            perf_examples=perf_examples,
            log_interval=log_interval,
        )
    elif scenario == "Server":
        return SUT_Server(
            model_path=model_path,
            dataset_path=dataset_path,
            max_examples=max_examples,
            perf_examples=perf_examples,
            log_interval=log_interval,
        )
