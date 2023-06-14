import array
import torch
import requests
import json
import os
import sys

import mlperf_loadgen as lg
from dataset import Dataset



class SUT_base:
    def __init__(
        self,
        model_path,
        dtype,
        dataset_path,
        max_examples,
        args,
        megatron_args,
        use_gpu=False,
        gen_kwargs={},
    ):
        # TODO : Pass model file name to init instead of args
        print("Loading PyTorch model...")
        self.model_name = "Megatron-LM"
        self.dataset_path = dataset_path
        self.url = 'http://localhost:5000/api'
        self.headers = {'Content-Type': 'application/json'}
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.gen_kwargs = gen_kwargs
        # dtype
        if dtype == "bfloat16":
            self.amp_enabled = True
            self.amp_dtype = torch.bfloat16
            print("BF16 autocast")
        elif dtype == "float16":
            self.amp_enabled = True
            self.amp_dtype = torch.float16
        else:
            self.amp_enabled = False
            self.amp_dtype = torch.float32

        self.data_object = Dataset(
            self.dataset_path, total_count_override=max_examples, args=args
        )

        self.qsl = lg.ConstructQSL(
            self.data_object.count,
            self.data_object.perf_count,
            self.data_object.LoadSamplesToRam,
            self.data_object.UnloadSamplesFromRam,
        )

        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)

    def issue_queries(self, query_samples):
        print("Number of Samples in query_samples : ", len(query_samples))

        for i in range(len(query_samples)):
            index = query_samples[i].index
            input_ids_tensor = self.data_object.source_encoded_input_ids[index]
            input_masks_tensor = self.data_object.source_encoded_attn_masks[index]
            input_length_tensor = self.data_object.source_encoded_input_id_lengths[index]

            pred_output_batch = (
                self.inference_call(
                    input_ids_tensor, input_masks_tensor, input_length_tensor
                )
                .cpu()
                .numpy()
            )

            response_array = array.array("B", pred_output_batch[0].tobytes())
            bi = response_array.buffer_info()
            response = [lg.QuerySampleResponse(query_samples[i].id, bi[0], bi[1])]
            lg.QuerySamplesComplete(response)
            if i % 5 == 0:
                print("Completed : ", i)

    def inference_call(self, input_ids_tensor, input_masks_tensor, input_length_tensor):
        """Common for all scenarios"""
        # TODO: Encode the tensors
        data = {"input_ids_tensor": input_ids_tensor, "input_length_tensor": input_length_tensor}
        response = requests.put(self.url, data=json.dumps(data), headers=self.headers)
        if response.status_code != 200:
            # TODO: Manage exeption
            return None
        else:
            output_batch_truncated = []
            for t in response.json()['output'][0]:
                # TODO: Decode the tensors
                output_batch_truncated.append(t)
        output_batch_truncated = torch.stack(output_batch_truncated)
        return output_batch_truncated

    def flush_queries(self):
        pass

    def __del__(self):
        print("Finished destroying SUT.")


class SUT_Offline(SUT_base):
    def __init__(
        self,
        model_path,
        dtype,
        dataset_path,
        max_examples,
        args,
        megatron_args,
        use_gpu,
        gen_kwargs,
    ):
        SUT_base.__init__(
            self,
            model_path,
            dtype,
            dataset_path,
            max_examples,
            args,
            megatron_args,
            use_gpu,
            gen_kwargs,
        )

    """IssueQuery and inference methods implemented in Base class"""


class SUT_Server(SUT_base):
    def __init__(
        self,
        model_path,
        dtype,
        dataset_path,
        max_examples,
        args,
        megatron_args,
        use_gpu,
        gen_kwargs,
    ):

        SUT_base.__init__(
            self,
            model_path,
            dtype,
            dataset_path,
            max_examples,
            args,
            megatron_args,
            use_gpu,
            gen_kwargs,
        )
        self.total_samples_done = 0
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        print("SUT Server")

    def issue_queries(self, query_samples):

        index = query_samples[0].index
        input_ids_tensor = self.data_object.source_encoded_input_ids[index]
        input_masks_tensor = self.data_object.source_encoded_attn_masks[index]
        input_length_tensor = self.data_object.source_encoded_input_id_leghts[index]

        if self.use_gpu:
            input_ids_tensor = input_ids_tensor.to(self.device)
            input_masks_tensor = input_masks_tensor.to(self.device)

        pred_output_batch = (
            self.inference_call(
                input_ids_tensor, input_masks_tensor, input_length_tensor
            )
            .cpu()
            .numpy()
        )

        response_array = array.array("B", pred_output_batch.tobytes())
        bi = response_array.buffer_info()
        responses = [lg.QuerySampleResponse(query_samples[0].id, bi[0], bi[1])]
        lg.QuerySamplesComplete(responses)
        self.total_samples_done += 1
        if self.total_samples_done % 5 == 0:
            print("Completed : ", self.total_samples_done)


class SUT_SingleStream(SUT_base):
    def __init__(
        self,
        model_path,
        dtype,
        dataset_path,
        max_examples,
        args,
        megatron_args,
        use_gpu,
        gen_kwargs,
    ):
        SUT_base.__init__(
            self,
            model_path,
            dtype,
            dataset_path,
            max_examples,
            args,
            megatron_args,
            use_gpu,
            gen_kwargs,
        )
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        self.total_samples_done = 0

    def issue_queries(self, query_samples):

        index = query_samples[0].index
        input_ids_tensor = self.data_object.source_encoded_input_ids[index]
        input_masks_tensor = self.data_object.source_encoded_attn_masks[index]
        input_length_tensor = self.data_object.source_encoded_input_id_leghts[index]

        if self.use_gpu:
            input_ids_tensor = input_ids_tensor.to(self.device)
            input_masks_tensor = input_masks_tensor.to(self.device)

        pred_output_batch = (
            self.inference_call(
                input_ids_tensor, input_masks_tensor, input_length_tensor
            )
            .cpu()
            .numpy()
        )

        response_array = array.array("B", pred_output_batch.tobytes())
        bi = response_array.buffer_info()
        responses = [lg.QuerySampleResponse(query_samples[0].id, bi[0], bi[1])]
        lg.QuerySamplesComplete(responses)
        self.total_samples_done += 1
        if self.total_samples_done % 5 == 0:
            print("Completed : ", self.total_samples_done)


def get_SUT(
    model_path,
    scenario,
    dtype,
    dataset_path,
    max_examples,
    args,
    megatron_args,
    use_gpu=False,
    gen_kwargs={},
):
    if scenario == "Offline":
        return SUT_Offline(
            model_path,
            dtype,
            dataset_path,
            max_examples,
            args,
            megatron_args,
            use_gpu,
            gen_kwargs=gen_kwargs,
        )
    elif scenario == "Server":
        return SUT_Server(
            model_path,
            dtype,
            dataset_path,
            max_examples,
            args,
            use_gpu,
            megatron_args,
            gen_kwargs=gen_kwargs,
        )
    elif scenario == "SingleStream":
        return SUT_SingleStream(
            model_path,
            dtype,
            dataset_path,
            max_examples,
            args,
            megatron_args,
            use_gpu,
            gen_kwargs=gen_kwargs,
        )
