"""
Python demo showing how to use the MLPerf Inference LoadGen over the Network bindings.
This program runs on the LON Node side.
It runs the demo in MLPerf server mode over the network.
It communicates over the network with Network SUT nodes,
which are running the networked SUT code.
"""

import argparse
import threading
import requests
import array
import time
import json
import array
import collections
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from absl import app
# from absl import flags
import mlperf_loadgen as lg
import numpy as np
import torch

import struct

import dataset
import coco

from queue import Queue

# FLAGS = flags.FLAGS

# flags.DEFINE_list(
#     "sut_server", "http://localhost:8000", "Address of the server(s) under test."
# )

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
    "stable-diffusion-migraphx": {
        "dataset": "coco-1024",
        "backend": "migraphx",
        "model-name": "stable-diffusion-xl",
    },
}

SCENARIO_MAP = {
    "SingleStream": lg.TestScenario.SingleStream,
    "MultiStream": lg.TestScenario.MultiStream,
    "Server": lg.TestScenario.Server,
    "Offline": lg.TestScenario.Offline,
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sut-server', required=True, nargs='+', help='A list of server address & port')
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
    # pass this argument for official submission
    # parser.add_argument("--output-images", action="store_true", help="Store a subset of the generated images")
    # do not modify this argument for official submission
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
        parser.error("valid scanarios:" + str(list(SCENARIO_MAP.keys())))
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

class QSL:
    def __init__(self, total_sample_count, performance_sample_count, ds=None):
        # self.eval_features = {
        #     i: {"index": i, "id": i} for i in range(total_sample_count)
        # }
        self.qsl = lg.ConstructQSL(
            total_sample_count, 
            performance_sample_count, 
            ds.load_query_samples, 
            ds.unload_query_samples
        )
    
    def __del__(self):
        lg.DestroyQSL(self.qsl)

class QDL:
    """QDL acting as a proxy to the SUT.
    This QDL communicates with the SUT via HTTP.
    It uses two endpoints to communicate with the SUT:
    - /predict/ : Send a query to the SUT and get a response.
    - /getname/ : Get the name of the SUT. Send a getname to the SUT and get a response.
    """

    def __init__(self, qsl: QSL, sut_server_addr: list, ds=None):
        """
        Constructor for the QDL.
        Args:
            qsl: The QSL to use.
            sut_server_addr: A list of addresses of the SUT.
        """
        self.qsl = qsl

        # Construct QDL from the python binding
        self.qdl = lg.ConstructQDL(
            self.issue_query, self.flush_queries, self.client_get_name
        )
        self.sut_server_addr = sut_server_addr
        self.ds = ds
        

    def issue_query(self, query_samples):
        """Process the query to send to the SUT"""
        threading.Thread(
            target=self.process_query_async,
            args=[query_samples],
            daemon=True # remove
            ).start()

    def flush_queries(self):
        """Flush the queries. Dummy implementation."""
        pass

    def process_query_async(self, query_samples):
        """Serialize the query, send it to the SUT in round robin, and return the deserialized response."""
        
        query_samples_len = len (query_samples)
        query_samples_seg_len = int (query_samples_len / len (self.sut_server_addr))
        splitted_query_samples = []
        for idx in range (len (self.sut_server_addr)): 
            if idx == len (self.sut_server_addr) -1: 
                splitted_query_samples.append (query_samples[idx*query_samples_seg_len:])
            else:
                splitted_query_samples.append (query_samples[idx*query_samples_seg_len : (idx+1)*query_samples_seg_len])
        
        responses = []
        with ThreadPoolExecutor(max_workers=len(self.sut_server_addr)) as executor:
            futures = { 
                executor.submit(self.request_validate, '{}/predict/'.format(url), queries): self
                for url, queries in zip(self.sut_server_addr, splitted_query_samples)
            }
        
                

    # Send inference request to one host, receive the inference result
    # then calls loadgen to verify the inference result
    def request_validate(self, url, query_samples, backend="migraphx"):
        # turn query_samples into list of json: 
        indexes = [q.index for q in query_samples]
        ids = [q.id for q in query_samples]
        data, label = self.ds.get_samples(indexes)
        
        if backend == "migraphx":
            data = [
                {
                    'caption': d['caption'],
                    'latents': d['latents'].tolist()  # Convert tensor to a list
                }
                for d in data
            ]
        else:
            data = [
                {
                    'input_tokens': d['input_tokens'],
                    'input_tokens_2': d['input_tokens_2'],
                    'latents': d['latents'].tolist()  # Convert tensor to a list
                }
                for d in data
            ]
        
        '''
        data[0]:
        {
            'input_tokens': <class 'transformers.tokenization_utils_base.BatchEncoding'>, 
            'input_tokens_2': <class 'transformers.tokenization_utils_base.BatchEncoding'>, 
            'latents': <class 'torch.Tensor'>  
        }
        or
        {
            'caption': <class 'str'>
            'latents': <class 'torch.Tensor'>  
        }
        '''
        
        # Todo: The response got None object when we have 2 inference nodes
        # This problem doesn't exist when we just inference on one node
        
        query_samples = [ {'index': q[0], 'id': q[1], 'data': q[2]} 
                         for q in zip(indexes, ids, data) ]
        response = requests.post(url, json={"query_samples": query_samples})
        e = time.time()
        print (f'RETURNED from requests.post on predict at time \t {e}')
        
        
        
        
        # print(response.json()["result"])
        
        # print("result type:", type(result))
        # print("result:", result)
        # result = response.json()["result"]
        # print("result type:", type(type(result)))
        # print("result type:", type(result))
        # print("result:", result)
        # print("result len:", len(result))
        # print("result[0]:", result[0])
        
        
        
        # response_array_refs = []
        # response = []
        # for sample in result:
        #     sample_in_memory = array.array("B", sample['data'])
        #     bi = sample_in_memory.buffer_info()
        #     response_array_refs.append(sample_in_memory)
        #     response.append(lg.QuerySampleResponse(sample['query_id'], bi[0], bi[1]))
            
        response_bytes = response.content
        offset = 0
        responses = []
        response_array_refs = []

        while offset < len(response_bytes):
            # Unpack the query_id
            query_id = struct.unpack_from('Q', response_bytes, offset)[0]
            offset += 8

            # Unpack the data length
            data_length = struct.unpack_from('I', response_bytes, offset)[0]
            offset += 4

            # Extract the data
            data_bytes = response_bytes[offset:offset + data_length]
            offset += data_length

            # Convert bytes to array
            sample_in_memory = array.array("B", data_bytes)
            bi = sample_in_memory.buffer_info()
            response_array_refs.append(sample_in_memory)

            responses.append(lg.QuerySampleResponse(query_id, bi[0], bi[1]))
        
            
        print (f'BEFORE lg.QuerySamplesComplete(response)')
        lg.QuerySamplesComplete(responses)
        print (f'AFTER lg.QuerySamplesComplete(response)')
        
        
        '''
        query_samples[0]:
        {
            'index': 1, 
            'id': 1, 
            'data': {
                'inputs_tokens': "this is a prompt",
                'inputs_tokens_2': "this is a prompt",
                'latents': [list converted from tensor]
            }
        }
        or
        {
            'index': 1, 
            'id': 1, 
            'data': {
                'caption': "this is a prompt",
                'latents': [list converted from tensor]
            }
        }
        '''
        

    def client_get_name(self):
        """Get the name of the SUT from ALL the SUTS."""
        # if len(self.sut_server_addr) == 1:
        #     return requests.post(
        #         f"{self.sut_server_addr[0]}/getname/").json()["name"]

        # sut_names = [
        #     requests.post(f"{addr}/getname/").json()["name"]
        #     for addr in self.sut_server_addr
        # ]
        # return "Multi-node SUT: " + ", ".join(sut_names)
        return "Multi-node SUT: N1, N2"

    def __del__(self):
        lg.DestroyQDL(self.qdl)

def main(args):
    # args = get_args()
    
    backend = get_backend(
                    args.backend,
                    precision=args.dtype,
                    device='cuda:0',
                    model_path=args.model_path,
                    batch_size=args.max_batchsize
                )
    model = backend.load()

    
    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    
    # --count applies to accuracy mode only and can be used to limit the number of images
    # for testing.
    count_override = False
    count = args.count
    if count:
        count_override = True
    
    scenario = SCENARIO_MAP[args.scenario]
    
    dataset_class, pre_proc, post_proc, kwargs = SUPPORTED_DATASETS[args.dataset]
    ds = dataset_class(
        data_path=args.dataset_path,
        name=args.dataset,
        pre_process=pre_proc,
        count=count,
        threads=args.threads,
        # pipe_tokenizer=models[0].pipe.tokenizer,
        # pipe_tokenizer_2=models[0].pipe.tokenizer_2,
        pipe_tokenizer=model.pipe.tokenizer,
        pipe_tokenizer_2=model.pipe.tokenizer_2,
        latent_dtype=dtype,
        latent_device=args.device,
        latent_framework=args.latent_framework,
        pipe_type=args.backend,
        **kwargs,
    )
    count = ds.get_item_count()
    
    
    mlperf_conf = os.path.abspath(args.mlperf_conf)
    if not os.path.exists(mlperf_conf):
        log.error("{} not found".format(mlperf_conf))
        sys.exit(1)

    user_conf = os.path.abspath(args.user_conf)
    if not os.path.exists(user_conf):
        log.error("{} not found".format(user_conf))
        sys.exit(1)

    audit_config = os.path.abspath(args.audit_conf)
    
    if args.accuracy:
        ids_path = os.path.abspath(args.ids_path)
        with open(ids_path) as f:
            saved_images_ids = [int(_) for _ in f.readlines()]

    if args.output:
        output_dir = os.path.abspath(args.output)
        os.makedirs(output_dir, exist_ok=True)
        os.chdir(output_dir)

    performance_sample_count = (
        args.performance_sample_count
        if args.performance_sample_count
        else min(count, 500)
    )
    

    
    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = output_dir
    log_output_settings.copy_summary_to_stdout = False
    log_settings = lg.LogSettings()
    log_settings.enable_trace = args.debug
    log_settings.log_output = log_output_settings

    settings = lg.TestSettings()
    settings.FromConfig(mlperf_conf, args.model_name, args.scenario)
    settings.FromConfig(user_conf, args.model_name, args.scenario)
    if os.path.exists(audit_config):
        settings.FromConfig(audit_config, args.model_name, args.scenario)
    settings.scenario = scenario
    settings.mode = lg.TestMode.PerformanceOnly
    if args.accuracy:
        settings.mode = lg.TestMode.AccuracyOnly
    if args.find_peak_performance:
        settings.mode = lg.TestMode.FindPeakPerformance

    if args.time:
        # override the time we want to run
        settings.min_duration_ms = args.time * MILLI_SEC
        settings.max_duration_ms = args.time * MILLI_SEC

    if args.qps:
        qps = float(args.qps)
        settings.server_target_qps = qps
        settings.offline_expected_qps = qps

    if count_override:
        settings.min_query_count = count
        settings.max_query_count = count

    if args.samples_per_query:
        settings.multi_stream_samples_per_query = args.samples_per_query
    if args.max_latency:
        settings.server_target_latency_ns = int(args.max_latency * NANO_SEC)
        settings.multi_stream_expected_latency_ns = int(args.max_latency * NANO_SEC)

    performance_sample_count = (
        args.performance_sample_count
        if args.performance_sample_count
        else min(count, 500)
    )

    # QDL and QSL
    qsl = QSL(count, performance_sample_count, ds=ds)
    # qsl = QSL(50, performance_sample_count, ds=ds)
    qdl = QDL(qsl, sut_server_addr=args.sut_server, ds=ds)

    lg.StartTest(qdl.qdl, qsl.qsl, settings)
    
    del qsl
    del qdl


if __name__ == "__main__":
    # app.run(main)
    main(None)