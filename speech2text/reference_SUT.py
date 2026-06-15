# Copyright (c) 2025 Intel Corporation
# Copyright (c) 2020, Cerebras Systems, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard packages
import sys
import os
import array
import subprocess
import math
import queue
import time
import logging
import threading

# Common math packages
import numpy as np
from tqdm import tqdm

# Framework packages
import torch
import torch.multiprocessing as mp
from vllm import LLM, SamplingParams

# Optimization packages
from numa import schedule, memory

# Local python packages
from QSL import AudioQSL, AudioQSLInMemory
import mlperf_loadgen as lg

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("SUT")


def get_start_cores(start_cores="0"):
    start_cores = start_cores.split(",")
    start_cores = list(map(int, start_cores))
    return start_cores


cores_per_inst = int(os.environ.get("CORES_PER_INST", "1"))
num_numa_nodes = int(os.environ.get("NUM_NUMA_NODES", "1"))
nodes_per_inst = int(os.environ["NUM_NUMA_NODES"]
                     ) / int(os.environ["NUM_INSTS"])
insts_per_node = int(os.environ["INSTS_PER_NODE"])
start_cores = os.environ["START_CORES"]

precision = torch.float32
n_mels = 128
sample_rate = 16000
model_path = "openai/whisper-large-v3"

labels = [
    " ",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "'",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "$",
    "%",
    "+",
    "-"]

labels_dict = {}
for i in range(len(labels)):
    labels_dict[labels[i]] = i


class Instance(mp.Process):
    def __init__(
        self,
        model_path=None,
        dataset_path=None,
        manifest_filepath=None,
        device="cpu",
        batch_size=-1,
        total_sample_count=-1,
        rank=-1,
        dtype=precision,
        core_list=(),
        node_list=(),
        input_queue=None,
        output_queue=None,
        cond_var=None,
        alive_counter=None,
        sample_counter=None
    ):
        mp.Process.__init__(self)
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.manifest_filepath = manifest_filepath
        self.device = device
        self.batch_size = batch_size
        self.total_sample_count = total_sample_count
        self.rank = rank
        self.dtype = dtype
        self.core_list = core_list
        self.node_list = node_list
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.cond_var = cond_var
        self.alive_counter = alive_counter
        self.sample_counter = sample_counter
        self.num_samples = 0
        self.total_time = 0
        self.query_idx_mapping = []
        self.qid_mapping = []
        self.req_counter = 0
        self.finished = False

    def run(self):
        node_list = tuple([math.floor(node) for node in self.node_list])
        memory.set_membind_nodes(*node_list)
        schedule.run_on_cpus(os.getpid(), *self.core_list)
        print(f"Binding rank {self.rank} to nodes {node_list}")
        print(f"Binding rank {self.rank} to cores {self.core_list}")

        dataset_vocab = labels

        self.qsl = AudioQSLInMemory(
            self.dataset_path,
            self.manifest_filepath,
            dataset_vocab,
            sample_rate,
            self.total_sample_count
        )

        dtype = "bfloat16"
        print(f"Precision: {dtype}")
        model = LLM(
            model=self.model_path,
            dtype=dtype,
            skip_tokenizer_init=False,
            trust_remote_code=True,
            tensor_parallel_size=1,
            max_num_seqs=64,
            max_model_len=448,
            max_num_batched_tokens=800,
            gpu_memory_utilization=0.95,
            num_scheduler_steps=1,
            limit_mm_per_prompt={"audio": 1},
        )
        sampling_params = SamplingParams(
            temperature=0,
            top_p=1.0,
            max_tokens=200,
        )

        self.model = model
        self.sampling_params = sampling_params
        with self.cond_var:
            self.alive_counter.value += 1
            self.cond_var.notify()

        keep_alive = True
        while keep_alive:
            keep_alive = self.process_queries()

    def process_queries(self):
        try:
            qitem_list = self.input_queue.get(timeout=1)
        except queue.Empty:
            qitem_list = None

        if qitem_list is None:
            return False

        prompt_list = []
        for qitem in qitem_list:
            prompt = self.qsl[qitem.index]
            prompt_list.append(prompt)
            self.query_idx_mapping.append(qitem.index)
            self.qid_mapping.append(qitem.id)
            self.req_counter += 1

        results = []
        query_ids = []
        qid = []

        start_time = time.time()
        outputs = self.model.generate(prompt_list, self.sampling_params)
        print(
            f"Sample number: {self.num_samples} | Step time {time.time()-start_time:.3f}s")

        for output in outputs:
            request_id = int(output.request_id)
            vllm_text = output.outputs[0].text
            results.append((vllm_text, len(output.outputs[0].token_ids)))
            query_ids.append(self.query_idx_mapping[request_id])
            qid.append(self.qid_mapping[request_id])

        self.num_samples += len(results)

        for i, result_tuple in enumerate(results):
            # Whisper outputs space in the front and capitalizes things
            result, n_tokens = result_tuple
            result = result.lower().strip()
            transcript = []
            for s in result:
                if s in labels_dict:
                    transcript.append(labels_dict[s])
            transcript = [transcript]

            assert len(transcript) == 1
            response_array = array.array('q', transcript[0])

            self.output_queue.put((qid[i], n_tokens, response_array))
            print(f"Finished {qid[i]}")
        return True


class vllmSUT:
    def __init__(self, dataset_dir,
                 manifest_filepath, perf_count, model_path="openai/whisper-large-v3", num_workers=1, device="cpu"):
        self.model_path = model_path
        self.dataset_path = dataset_dir
        self.manifest_filepath = manifest_filepath
        self.device = device
        self.batch_size = 16
        self.total_sample_count = perf_count
        self.num_workers = num_workers
        self.worker_threads = [None] * self.num_workers

        dataset_vocab = labels

        # self.dev = torch.device("cuda:0") if torch.cuda.is_available() and os.environ.get("USE_GPU", "").lower() not in  [ "no", "false" ]  else torch.device("cpu")

        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        self.qsl = AudioQSL(dataset_dir,
                            manifest_filepath,
                            dataset_vocab,
                            sample_rate,
                            perf_count)
        self.query_queue = mp.JoinableQueue()
        self.output_queue = mp.Queue()
        self.alive_counter = mp.Value("i", 0)
        self.cond_var = mp.Condition(lock=mp.Lock())
        self.sample_counter = mp.Value("i", 0)

    def start(self):
        node_start_cores = get_start_cores(start_cores)
        core_lists = []
        if insts_per_node > 0:
            for i in range(num_numa_nodes):
                for j in range(insts_per_node):
                    core_lists.append(
                        list(
                            range(
                                node_start_cores[i] +
                                j *
                                cores_per_inst,
                                node_start_cores[i] +
                                (
                                    j +
                                    1) *
                                cores_per_inst)))

        for j in range(self.num_workers):
            core_list = core_lists[j]

            worker = Instance(
                model_path=self.model_path,
                dataset_path=self.dataset_path,
                manifest_filepath=self.manifest_filepath,
                device=self.device,
                batch_size=self.batch_size,
                total_sample_count=self.total_sample_count,
                rank=j,
                dtype=precision,
                core_list=tuple(core_list),
                node_list=tuple([math.floor(j * nodes_per_inst)]),
                input_queue=self.query_queue,
                output_queue=self.output_queue,
                cond_var=self.cond_var,
                alive_counter=self.alive_counter,
                sample_counter=self.sample_counter
            )
            worker.start()
            self.worker_threads[j] = worker

        with self.cond_var:
            self.cond_var.wait_for(
                lambda: self.alive_counter.value == self.num_workers)

        log.info(f"Starting Loadgen response thread")
        response_thread = threading.Thread(target=self.response_loadgen)
        response_thread.daemon = True
        response_thread.start()

    def issue_queries(self, query_samples):
        query_sample_list = []
        for query_sample in query_samples:
            # Continuous batching
            self.query_queue.put([query_sample])
        if len(query_sample_list) > 0:
            self.query_queue.put(query_sample_list)

    def flush_queries(self):
        pass

    def response_loadgen(self):
        keep_alive = True
        while keep_alive:
            qid, n_tokens, response_array = self.output_queue.get()
            if qid is None:
                keep_alive = False
            else:
                bi = response_array.buffer_info()
                response = lg.QuerySampleResponse(qid, bi[0],
                                                  bi[1] * response_array.itemsize, n_tokens)
                lg.QuerySamplesComplete([response])

    def stop(self):
        for i in range(self.num_workers):
            self.query_queue.put(None)
        for worker in self.worker_threads:
            worker.kill()

    def __del__(self):
        lg.DestroySUT(self.sut)
        print("Finished destroying SUT.")
