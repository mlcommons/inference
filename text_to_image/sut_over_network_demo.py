"""
MLPerf Inference Benchmarking Tool - SUT Node
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import array
import collections
import json
import logging
import os
import sys
import threading
import time
import socket
import struct

import numpy as np
import torch

from flask import Flask, request, jsonify, Response
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess

import dataset
import coco

from queue import Queue

import mlperf_loadgen as lg  # Only needed if you plan to run LoadGen locally

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")

NANO_SEC = 1e9
MILLI_SEC = 1000

SUPPORTED_DATASETS = {
    "coco-1024": (
        coco.Coco,
        dataset.preprocess,
        coco.PostProcessCoco(),
        {"image_size": [3, 1024, 1024]},
    )
}

SUPPORTED_PROFILES = {
    "defaults": {
        "dataset": "coco-1024",
        "backend": "pytorch",
        "model-name": "stable-diffusion-xl",
    },
    "debug": {
        "dataset": "coco-1024",
        "backend": "debug",
        "model-name": "stable-diffusion-xl",
    },
    "stable-diffusion-xl-pytorch": {
        "dataset": "coco-1024",
        "backend": "pytorch",
        "model-name": "stable-diffusion-xl",
    },
    "stable-diffusion-xl-pytorch-dist": {
        "dataset": "coco-1024",
        "backend": "pytorch-dist",
        "model-name": "stable-diffusion-xl",
    },
    "stable-diffusion-xl-migraphx": {
        "dataset": "coco-1024",
        "backend": "migraphx",
        "model-name": "stable-diffusion-xl",
    }
}

SCENARIO_MAP = {
    "SingleStream": lg.TestScenario.SingleStream,
    "MultiStream": lg.TestScenario.MultiStream,
    "Server": lg.TestScenario.Server,
    "Offline": lg.TestScenario.Offline,
}

app = Flask(__name__)

# Global variables to hold models and runners
backends = []
models = []
runners = []
ds = None
args = None

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS.keys(), help="dataset")
    parser.add_argument("--dataset-path", required=True, help="path to the dataset")
    parser.add_argument(
        "--profile", choices=SUPPORTED_PROFILES.keys(), help="standard profiles"
    )
    parser.add_argument(
        "--scenario",
        default="SingleStream",
        help="mlperf benchmark scenario, one of " + str(list(SCENARIO_MAP.keys())),
    )
    parser.add_argument(
        "--max-batchsize",
        type=int,
        default=1,
        help="max batch size in a single inference",
    )
    parser.add_argument("--threads", default=1, type=int, help="threads")
    parser.add_argument("--accuracy", action="store_true", help="enable accuracy pass")
    parser.add_argument(
        "--find-peak-performance",
        action="store_true",
        help="enable finding peak performance pass",
    )
    parser.add_argument("--backend", help="Name of the backend", default="migraphx")
    parser.add_argument("--model-name", help="Name of the model")
    parser.add_argument("--output", default="output", help="test results")
    parser.add_argument("--qps", type=int, help="target qps")
    parser.add_argument("--model-path", help="Path to model weights")

    parser.add_argument(
        "--dtype",
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="dtype of the model",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu", "rocm"],
        help="device to run the benchmark",
    )
    parser.add_argument(
        "--latent-framework",
        default="torch",
        choices=["torch", "numpy"],
        help="framework to load the latents",
    )

    # file to use mlperf rules compliant parameters
    parser.add_argument(
        "--mlperf_conf", default="mlperf.conf", help="mlperf rules config"
    )
    # file for user LoadGen settings such as target QPS
    parser.add_argument(
        "--user_conf",
        default="user.conf",
        help="user config for user LoadGen settings such as target QPS",
    )
    # file for LoadGen audit settings
    parser.add_argument(
        "--audit_conf", default="audit.config", help="config for LoadGen audit settings"
    )
    # arguments to save images
    parser.add_argument("--ids-path", help="Path to caption ids", default="tools/sample_ids.txt")

    # below will override mlperf rules compliant settings - don't use for official submission
    parser.add_argument("--time", type=int, help="time to scan in seconds")
    parser.add_argument("--count", type=int, help="dataset items to use")
    parser.add_argument("--debug", action="store_true", help="debug")
    parser.add_argument(
        "--performance-sample-count", type=int, help="performance sample count", default=5000
    )
    parser.add_argument(
        "--max-latency", type=float, help="mlperf max latency in pct tile"
    )
    parser.add_argument(
        "--samples-per-query",
        default=8,
        type=int,
        help="mlperf multi-stream samples per query",
    )
    args = parser.parse_args()

    # don't use defaults in argparser. Instead we default to a dict, override that with a profile
    # and take this as default unless command line give
    defaults = SUPPORTED_PROFILES["defaults"]

    if args.profile:
        profile = SUPPORTED_PROFILES[args.profile]
        defaults.update(profile)
    for k, v in defaults.items():
        kc = k.replace("-", "_")
        if getattr(args, kc) is None:
            setattr(args, kc, v)

    if args.scenario not in SCENARIO_MAP:
        parser.error("valid scenarios:" + str(list(SCENARIO_MAP.keys())))
    return args

def get_backend(backend, **kwargs):
    if backend == "pytorch":
        from backend_pytorch import BackendPytorch

        backend = BackendPytorch(**kwargs)
        
    elif backend == "migraphx":
        from backend_migraphx import BackendMIGraphX

        backend = BackendMIGraphX(**kwargs)

    elif backend == "debug":
        from backend_debug import BackendDebug

        backend = BackendDebug()
    else:
        raise ValueError("unknown backend: " + backend)
    return backend

class Item:
    """An item that we queue for processing by the thread pool."""

    def __init__(self, query_id, content_id, inputs, img=None):
        self.query_id = query_id
        self.content_id = content_id
        self.img = img
        self.inputs = inputs
        self.start = time.time()

class RunnerBase:
    def __init__(self, model, ds, threads, post_proc=None, max_batchsize=128):
        self.take_accuracy = False
        self.ds = ds
        self.model = model
        self.post_process = post_proc
        self.threads = threads
        self.take_accuracy = False
        self.max_batchsize = max_batchsize
        self.result_timing = []
        self.result_dict = {}

    def handle_tasks(self, tasks_queue):
        pass

    def start_run(self, result_dict, take_accuracy):
        self.result_dict = result_dict
        self.result_timing = []
        self.take_accuracy = take_accuracy
        self.post_process.start()

    def run_one_item(self, qitem: Item):
        # print("in run_one_item")
        # run the prediction
        processed_results = []
        
        # preprocess the prompts:
        qitem.inputs = [
            {
                # "input_tokens": ds.preprocess(input['input_tokens'], ds.pipe_tokenizer),
                # "input_tokens_2": ds.preprocess(input['input_tokens_2'], ds.pipe_tokenizer_2),
                "caption": input['caption'],
                "latents": torch.tensor(input['latents']).half(),  #.half()
            }
            for input in qitem.inputs
        ]
        # 
        try:
            # log.info(f"[Yalu] qitem.inputs[0]['caption'] -> {qitem.inputs[0].get('caption')}")
            # log.info(f"[Yalu] qitem.inputs[0]['latents'] -> {qitem.inputs[0].get('latents')}")
            # log.info(f"[Yalu] qitem.inputs length -> {len(qitem.inputs)}")
            results = self.model.predict(qitem.inputs)
            processed_results = self.post_process(
                results, qitem.content_id, qitem.inputs, self.result_dict
            )
            if self.take_accuracy:
                self.post_process.add_results(processed_results)
            self.result_timing.append(time.time() - qitem.start)
        except Exception as ex:  # pylint: disable=broad-except
            src = [self.ds.get_item_loc(i) for i in qitem.content_id]
            log.error("thread: failed on contentid=%s, %s", src, ex)
            print("thread: failed on contentid=%s, %s", src, ex)
            # since post_process will not run, fake empty responses
            processed_results = [[]] * len(qitem.query_id)
        finally:
            response_array_refs = []
            response = []
            for idx, query_id in enumerate(qitem.query_id):
                response_array = array.array(
                    "B", np.array(processed_results[idx], np.uint8).tobytes()
                )
                # response_array_refs.append(response_array)
                # bi = response_array.buffer_info()
                # response.append({'query_id': query_id, 'data': bi[0], 'size': bi[1]})
                response.append({'query_id': query_id, 'data': response_array.tolist()})
            return response  # Return the response instead of calling QuerySamplesComplete

    def enqueue(self, query_samples):
        try:
            idx = [q['index'] for q in query_samples]
            query_id = [q['id'] for q in query_samples]
            data = [q['data'] for q in query_samples]
            label = None # label is never used in any functions
            
            responses = []
            if len(idx) < self.max_batchsize:
                responses.extend(self.run_one_item(Item(query_id, idx, data, label)))
            else:
                bs = self.max_batchsize
                for i in range(0, len(idx), bs):
                    # print("samples obtained")
                    responses.extend(
                        self.run_one_item(
                            Item(query_id[i : i + bs], idx[i : i + bs], data[i : i + bs], label)
                        )
                    )
        except Exception as e:
            print(f'An error occured in enqueue: {e}')
        return responses

    def finish(self):
        pass

def initialize():
    global backends, models, runners, ds, args, post_proc
    args = get_args()

    log.info(args)

    # Initialize backends and models
    backends = [get_backend(
                    args.backend,
                    precision=args.dtype,
                    device=f'cuda:{i}',
                    model_path=args.model_path,
                    batch_size=args.max_batchsize
                ) 
                for i in [0,1,2,3]]  # Adjust GPU indices as needed

    models = [backend.load() for backend in backends]

    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    # Load dataset
    dataset_class, pre_proc, post_proc, kwargs = SUPPORTED_DATASETS[args.dataset]
    ds = dataset_class(
        data_path=args.dataset_path,
        name=args.dataset,
        pre_process=pre_proc,
        count=args.count,
        threads=args.threads,
        pipe_tokenizer=models[0].pipe.tokenizer,
        pipe_tokenizer_2=models[0].pipe.tokenizer_2,
        latent_dtype=dtype,
        latent_device=args.device,
        latent_framework=args.latent_framework,
        pipe_type=args.backend,
        **kwargs,
    )

    scenario = SCENARIO_MAP[args.scenario]
    runner_map = {
        lg.TestScenario.SingleStream: RunnerBase,
        lg.TestScenario.MultiStream: RunnerBase,
        lg.TestScenario.Server: RunnerBase,
        lg.TestScenario.Offline: RunnerBase,
    }

    runners = [runner_map[scenario](
                    model, ds, args.threads, post_proc=post_proc, max_batchsize=args.max_batchsize
                )
                for model in models]
    
    # added because we need to pass result_dict to the runner class
    log.info("starting {}".format(scenario))
    result_dict = {"scenario": str(scenario)}
    for runner in runners: 
        runner.start_run(result_dict, args.accuracy)

@app.route('/predict/', methods=['POST'])
def predict():
    query_data = request.get_json(force=True)
    query_samples = query_data['query_samples']

    # Distribute queries among runners
    query_samples_len = len(query_samples)
    num_runners = len(runners)
    query_samples_seg_len = int(query_samples_len / num_runners)
    splitted_query_samples = []
    for idx in range(num_runners):
        if idx == num_runners -1:
            splitted_query_samples.append(query_samples[idx*query_samples_seg_len:])
        else:
            splitted_query_samples.append(query_samples[idx*query_samples_seg_len : (idx+1)*query_samples_seg_len])

    # Use ThreadPoolExecutor to run queries concurrently
    responses = []
    with ThreadPoolExecutor(max_workers=num_runners) as executor:
        futures = {
            executor.submit(runner.enqueue, queries): runner 
            for runner, queries in zip(runners, splitted_query_samples)
        }

        for future in as_completed(futures):
            runner = futures[future]
            try:
                result = future.result()
                responses.extend(result)
            except Exception as exc:
                log.error(f'Runner {runner} generated an exception: {exc}')

    print(f'response of len {len(responses)} returned')
    print (f'RETURNING from predict')
    
    s = time.time() 
    # output = jsonify(result=responses)
    response_bytes = bytearray()
    for resp in responses:
        query_id = resp['query_id']
        data_array = np.array(resp['data'], dtype=np.uint8)
        data_bytes = data_array.tobytes()

        # Pack the query_id (8 bytes) and the length of data (4 bytes), then the data
        packed_data = struct.pack('Q', query_id)
        packed_data += struct.pack('I', len(data_bytes))
        packed_data += data_bytes
        response_bytes.extend(packed_data)
    e = time.time()
    
    print (f'\n Time to jsonify output is: \t {e-s} \n')
    print (f'\n Mark Time to return: \t {e} \n')
    # Todo: send samples back
    # return output 
    print(f'Type of response_bytes: {type(response_bytes)}') 
    return Response(bytes(response_bytes), mimetype='application/octet-stream')

@app.route('/getname/', methods=['POST', 'GET'])
def getname():
    return jsonify(name=f"SUT Node running on {socket.gethostname()}")

def issue_queries(query_samples):
    # This function is not used in the networked version
    pass

def flush_queries():
    pass

if __name__ == "__main__":
    initialize()
    
    # get public ip addr of current node
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip_address = s.getsockname()[0]
    
    # Change host ip addr and port number 
    app.run(host=ip_address, port=8008)
