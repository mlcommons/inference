import os
import time
import numpy as np
import array
import torch
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
from transformers.generation.streamers import BaseStreamer

import threading
import queue

import logging
from typing import TYPE_CHECKING, Optional, List

import mlperf_loadgen as lg
from dataset import Dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Llama-70B-SUT")

gen_kwargs = {
    "early_stopping": True,
    "max_new_tokens": 1024,
    "min_new_tokens": 1,
    "num_beams": 1,
    "do_sample": False
}



class FirstTokenStreamer(BaseStreamer):
    """ Streams first tokens to a 'holder' """

    def __init__(self, first_token, tokens_cache=[], is_first_token=True, response_ids=[] ):
        """ Response ids added to 'sign' the first token"""

        self.first_token = first_token # Queue for first token
        self.is_first_token = is_first_token

        # Cache for subsequent generated tokens
        self.tokens_cache = tokens_cache

        self.response_ids = response_ids

        self.is_prompt = True # The first tokens sent to the streamer are actually the input prompts

    def put(self, value):
        """ Caches the tokens as they're generated. Assumes bs=1 """

        # Prompts are streamed first so we need to skip the first time value that arrives
        if self.is_prompt:
            self.is_prompt = False
            return

        value = value.item()
        if self.is_first_token:

            # Add generated first token together with its query response_id to first tokens queue
            self.first_token.put((value, self.response_ids[0]))

            self.is_first_token = False
            return

        self.tokens_cache.append(value)


    def end(self):
        pass

    def get_out_tokens(self):
        return self.tokens_cache


class SUT():
    def __init__(self, model_path=None, dtype="bfloat16", device="cpu", total_sample_count=24576, dataset_path=None, workers=1):

        self.model_path = model_path or "meta-llama/Llama-2-70b-chat-hf"
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
        self.data_object = Dataset(self.model_path, dataset_path=self.dataset_path, total_sample_count=total_sample_count)
        self.qsl = lg.ConstructQSL(self.data_object.total_sample_count, self.data_object.perf_count,
                                   self.data_object.LoadSamplesToRam, self.data_object.UnloadSamplesFromRam)

        self.load_model()

        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()


    def start(self):
        
        # Create worker threads
        for j in range(self.num_workers):
            worker = threading.Thread(target=self.process_queries)
            worker.start()
            self.worker_threads[j] = worker

    def stop(self):
        for _ in range(self.num_workers):
            self.query_queue.put(None)

        for worker in self.worker_threads:
            worker.join()


    def process_queries(self):
        """Processor of the queued queries. User may choose to add batching logic """

        while True:
            qitem = self.query_queue.get()
            if qitem is None:
                break

            # TODO: If batching, call collator to batch the inputs here
            input_ids_tensor = self.data_object.input_ids[qitem.index]
            input_masks_tensor = self.data_object.attention_masks[qitem.index]
            input_len = [self.data_object.input_lens[qitem.index]]


            pred_output_tokens = self.model.generate(
                                                input_ids=input_ids_tensor,
                                                attention_mask=input_masks_tensor,
                                                pad_token_id=self.tokenizer.pad_token_id,
                                                **gen_kwargs
                                                )

            processed_output = self.data_object.postProcess(pred_output_tokens, input_len)

            response_array = array.array("B", processed_output[0].tobytes())
            bi = response_array.buffer_info()
            response = [lg.QuerySampleResponse(
                qitem.id, bi[0], bi[1])]
            lg.QuerySamplesComplete(response)


    def load_model(self):
        self.model = LlamaForCausalLM.from_pretrained(
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
        pass


class SUTServer(SUT):
    def __init__(self, model_path=None, dtype="bfloat16", device="cpu", total_sample_count=24576, dataset_path=None, workers=1):

        super().__init__(model_path=model_path, dtype=dtype, device=device, total_sample_count=total_sample_count, dataset_path=dataset_path, workers=workers)

        self.first_token_queue = queue.Queue()

    def start(self):

        # Create worker threads
        for j in range(self.num_workers):
            worker = threading.Thread(target=self.process_queries)
            worker.start()
            self.worker_threads[j] = worker

        # Create first token response thread
        self.ft_response_thread = threading.Thread(target=self.process_first_tokens)
        self.ft_response_thread.start()


    def process_first_tokens(self):

        while True:
            first_token_item = self.first_token_queue.get()

            if first_token_item is None:
                log.info("Exiting First token response thread")
                break

            first_tokens, response_id = first_token_item

            response_data = array.array("B", np.array(first_tokens, np.float32).tobytes())
            bi = response_data.buffer_info()
            response = [lg.QuerySampleResponse(response_id, bi[0], bi[1])]
            lg.FirstTokenComplete(response)

    def process_queries(self):
        """Processor of the queued queries. User may choose to add batching logic """
        while True:

            qitem = self.query_queue.get()
            if qitem is None:
                break
            
            input_ids_tensor = self.data_object.input_ids[qitem.index]
            input_masks_tensor = self.data_object.attention_masks[qitem.index]

            #TODO: This PoC is super slow with significant overhead. Best to create a patch to `generate`
            tokens_cache = []
            tokens_streamer = FirstTokenStreamer(self.first_token_queue, tokens_cache=tokens_cache, is_first_token=True, response_ids=[qitem.id])

            _ = self.model.generate(    input_ids=input_ids_tensor,
                                        attention_mask=input_masks_tensor,
                                        pad_token_id=self.tokenizer.pad_token_id,
                                        streamer = tokens_streamer,
                                        **gen_kwargs
                                        )

            output_tokens = tokens_streamer.get_out_tokens()

            response_array = array.array("B", np.array(output_tokens, np.int32).tobytes())
            bi = response_array.buffer_info()
            response = [lg.QuerySampleResponse(
                qitem.id, bi[0], bi[1])]
            lg.QuerySamplesComplete(response)
    

    def issue_queries(self, query_samples):

        self.query_queue.put(query_samples[0])


    def stop(self):
        for _ in range(self.num_workers):
            self.query_queue.put(None)

        for worker in self.worker_threads:
            worker.join()

        self.first_token_queue.put(None)
        self.ft_response_thread.join()
