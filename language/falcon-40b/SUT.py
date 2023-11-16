import os
import time
import numpy as np
import array
import torch
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import mlperf_loadgen as lg
from dataset import Dataset

import threading
import queue

gen_kwargs = {
    "early_stopping": True,
    "max_new_tokens": 1024,
    "min_new_tokens": 1,
    "num_beams": 4
}


class SUT():
    def __init__(self, model_path=None, dtype="bfloat16", device="cpu", total_sample_count=24576, dataset_path=None, workers=1):
        # TODO : dataset_path should be used when dataset is already available on disk

        self.model_path = model_path or "tiiuae/falcon-40b-instruct"
        self.device = device

        # dtype
        if dtype == 'bfloat16':
            self.amp_enabled = True
            self.amp_dtype = torch.bfloat16
        elif dtype == 'float16':
            self.amp_enabled = True
            self.amp_dtype = torch.float16
        else:
            self.amp_enabled = False
            self.amp_dtype = torch.float32

        if 'cuda' in self.device:
            assert torch.cuda.is_available(), "torch gpu is not available, exiting..."

        self.dataset_path = dataset_path
        self.data_object = Dataset(dataset_path=self.dataset_path, total_sample_count=total_sample_count)
        self.qsl = lg.ConstructQSL(self.data_object.total_sample_count, self.data_object.perf_count,
                                   self.data_object.LoadSamplesToRam, self.data_object.UnloadSamplesFromRam)

        self.load_model()

        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()

    def start(self):
        for j in range(self.num_workers):
            worker = threading.Thread(target=self.process_queries)
            worker.start()
            self.worker_threads[j] = worker

        print("Worker threads created")

    def stop(self):
        for worker in self.worker_threads:
            worker.join()

        for _ in range(self.num_workers):
            self.query_queue.put(None)


    def process_queries(self):
        """Processor of the queued queries. User may choose to add batching logic """

        while True:
            q_tem = self.query_queue.get()
            if qitem is None:
                break

            input_ids_tensor = self.data_object.input_ids[qitem.index]
            input_masks_tensor = self.data_object.attention_masks[qitem.index]

            pred_output_tokens = self.model.generate(
                                                input_ids=input_ids_tensor,
                                                attention_mask=input_masks_tensor,
                                                pad_token_id=self.tokenizer.pad_token_id,
                                                **gen_kwargs
                                                )

            processed_output = self.data_object.postProcess(pred_output_tokens)

            response_array = array.array("B", processed_output[0].tobytes())
            bi = response_array.buffer_info()
            response = [lg.QuerySampleResponse(
                qitem.id, bi[0], bi[1])]
            lg.QuerySamplesComplete(response)


    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map= "auto" if self.device=="cpu" else None,
            low_cpu_mem_usage=True if self.device=="cpu" else False,
            torch_dtype=self.amp_dtype
        )

        self.device = torch.device(self.device)
        self.model.to(self.device)

        self.model.eval()
        self.model = self.model.to(memory_format=torch.channels_last)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            model_max_length=1024,
            padding_side="left",
            use_fast=False,)

        self.tokenizer.pad_token = self.tokenizer.eos_token

    def get_sut(self):
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        return self.sut

    def get_qsl(self):
        return self.qsl


    def predict(self,**kwargs):
        raise NotImplementedError


    def issue_queries(self, query_samples):
        """ Receives samples from loadgen and adds them to queue. Users may choose to batch here"""

        list_prompts_tokens = []
        list_prompts_attn_masks = []

        for i in range(len(query_samples)):
            self.query_queue.put(query_samples[i])


    def flush_queries(self):
        pass

    def __del__(self):
        for worker in self.worker_threads:
            worker.join()

        for _ in range(self.num_workers):
            self.query_queue.put(None)

class SUTServer(SUT):
    def __init__(self, model_path=None, dtype="bfloat16", device="cpu", total_sample_count=24576, dataset_path=None, workers=1):

        super().__init__(model_path=model_path, dtype=dtype, device=device, total_sample_count=total_sample_count, dataset_path=dataset_path, workers=workers)


    def process_queries(self):
        """Processor of the queued queries. User may choose to add batching logic """

        while True:

            qitem = self.query_queue.get()
            if qitem is None:
                break
            
            input_ids_tensor = self.data_object.input_ids[qitem.index]
            input_masks_tensor = self.data_object.attention_masks[qitem.index]
            
            #TODO Generate and send 1st output token to loadgen, then generate remainder tokens
            pred_output_tokens = self.model.generate(
                                                input_ids=input_ids_tensor,
                                                attention_mask=input_masks_tensor,
                                                pad_token_id=self.tokenizer.pad_token_id,
                                                **gen_kwargs
                                                )

            processed_output = self.data_object.postProcess(pred_output_tokens)

            response_array = array.array("B", processed_output[0].tobytes())
            bi = response_array.buffer_info()
            response = [lg.QuerySampleResponse(
                qitem.id, bi[0], bi[1])]
            lg.QuerySamplesComplete(response)

    

    def issue_queries(self, query_samples):

        self.query_queue.put(query_samples[0])


