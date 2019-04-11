"""
mlperf inference benchmarking tool
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import json
import logging
import os
import threading
import time
from queue import Queue

import mlperf_loadgen as lg
import numpy as np

import dataset
import imagenet

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")

# pylint: disable=missing-docstring

# the datasets we support
SUPPORTED_DATASETS = {
    "imagenet":
        (imagenet.Imagenet, dataset.pre_process_vgg, dataset.post_process_offset1,
         {"image_size": [224, 224, 3]}),
    "imagenet_mobilenet":
        (imagenet.Imagenet, dataset.pre_process_mobilenet, dataset.post_process_argmax_offset,
         {"image_size": [224, 224, 3]}),
}

# pre-defined command line options so simplify things. They are used as defaults and can be
# overwritten from command line
DEFAULT_LATENCY_BUCKETS = "0.010,0.050,0.100,0.200,0.400"

SUPPORTED_PROFILES = {
    "defaults": {
        "dataset": "imagenet",
        "backend": "tensorflow",
        "cache": 0,
        "time": 128,
        "max-latency": DEFAULT_LATENCY_BUCKETS,
    },
    "mobilenet-tf": {
        "inputs": "input:0",
        "outputs": "MobilenetV1/Predictions/Reshape_1:0",
        "dataset": "imagenet_mobilenet",
        "backend": "tensorflow",
    },
    "mobilenet-onnx": {
        "dataset": "imagenet_mobilenet",
        "outputs": "MobilenetV1/Predictions/Reshape_1:0",
        "backend": "onnxruntime",
    },
    "resnet50-tf": {
        "inputs": "input_tensor:0",
        "outputs": "ArgMax:0",
        "dataset": "imagenet",
        "backend": "tensorflow",
    },
    "resnet50-onnxruntime": {
        "dataset": "imagenet",
        "outputs": "ArgMax:0",
        "backend": "onnxruntime",
    }
}


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS.keys(), help="dataset")
    parser.add_argument("--dataset-path", required=True, help="path to the dataset")
    parser.add_argument("--dataset-list", help="path to the dataset list")
    parser.add_argument("--data-format", choices=["NCHW", "NHWC"], help="data format")
    parser.add_argument("--profile", choices=SUPPORTED_PROFILES.keys(), help="standard profiles")
    parser.add_argument("--model", required=True, help="model file")
    parser.add_argument("--inputs", help="model inputs")
    parser.add_argument("--output", help="test results")
    parser.add_argument("--outputs", help="model outputs")
    parser.add_argument("--backend", help="runtime to use")
    parser.add_argument("--threads", default=os.cpu_count(), type=int, help="threads")
    parser.add_argument("--time", type=int, help="time to scan in seconds")
    parser.add_argument("--count", type=int, help="dataset items to use")
    parser.add_argument("--max-latency", type=str, help="max latency in 99pct tile")
    parser.add_argument("--cache", type=int, default=0, help="use cache")
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
    if args.inputs:
        args.inputs = args.inputs.split(",")
    if args.outputs:
        args.outputs = args.outputs.split(",")
    if args.max_latency:
        args.max_latency = [float(i) for i in args.max_latency.split(",")]
    return args


def get_backend(backend):
    if backend == "tensorflow":
        from backend_tf import BackendTensorflow
        backend = BackendTensorflow()
    elif backend == "onnxruntime":
        from backend_onnxruntime import BackendOnnxruntime
        backend = BackendOnnxruntime()
    elif backend == "null":
        from backend_null import BackendNull
        backend = BackendNull()
    elif backend == "pytorch":
        from backend_pytorch import BackendPytorch
        backend = BackendPytorch()
    elif backend == "tflite":
        from backend_tflite import BackendTflite
        backend = BackendTflite()
    else:
        raise ValueError("unknown backend: " + backend)
    return backend


class Item:
    """An item that we queue for processing by the thread pool."""

    def __init__(self, query_id, img, label=None):
        self.id = query_id
        self.img = img
        self.label = label
        self.start = time.time()


class Runner:
    def __init__(self, model, ds, threads, post_process=None):
        self.tasks = Queue(maxsize=threads * 5)
        self.workers = []
        self.model = model
        self.post_process = post_process
        self.threads = threads
        self.result_list = []
        self.result_dict = {}

    def handle_tasks(self, tasks_queue):
        """Worker thread."""
        while True:
            qitem = tasks_queue.get()
            if qitem is None:
                # None in the queue indicates the parent want us to exit
                tasks_queue.task_done()
                break

            try:
                # run the prediction
                results = self.model.predict({self.model.inputs[0]: qitem.img})
                # and keep track of how long it took
                self.result_list.append(time.time() - qitem.start)
                response = []
                for idx, result in enumerate(results[0]):
                    result = self.post_process(result)
                    if qitem.label[idx] == result:
                        self.result_dict["good"] += 1
                    self.result_dict["total"] += 1
                    # FIXME: unclear what to return here
                    # response.append(lg.QuerySampleResponse(result, sys.getsizeof(result)))
                    response.append(lg.QuerySampleResponse(0, 0))
                lg.QueryComplete(qitem.id, response)
            except Exception as ex:  # pylint: disable=broad-except
                log.error("execute_parallel thread: %s", ex)

            tasks_queue.task_done()

    def start_pool(self):
        for _ in range(self.threads):
            worker = threading.Thread(target=self.handle_tasks, args=(self.tasks,))
            worker.daemon = True
            self.workers.append(worker)
            worker.start()

    def start_run(self, result_list, result_dict):
        self.result_list = result_list
        self.result_dict = result_dict

    def enqueue(self, id, data, label):
        item = Item(id, data, label)
        self.tasks.put(item)

    def finish(self):
        # exit all threads
        for _ in self.workers:
            self.tasks.put(None)
        for worker in self.workers:
            worker.join()


def add_results(final_results, name, result_dict, result_list, took):
    percentiles = [50., 80., 90., 95., 99., 99.9]
    buckets = np.percentile(result_list, percentiles).tolist()
    buckets_str = ",".join(["{}:{:.4f}".format(p, b) for p, b in zip(percentiles, buckets)])

    # this is what we record for each run
    result = {
        "mean": np.mean(result_list),
        "took": took,
        "qps": len(result_list) / took,
        "count": len(result_list),
        "percentiles": {str(k): v for k, v in zip(percentiles, buckets)},
        "good_items": result_dict["good"],
        "total_items": result_dict["total"],
        "accuracy": 100. * result_dict["good"] / result_dict["total"],
    }
    # add the result to the result dict
    final_results[name] = result

    # to stdout
    print("{} qps={:.2f}, mean={:.6f}, time={:.2f}, acc={:.2f}, tiles={}".format(
        name, result["qps"], result["mean"], took, result["accuracy"], buckets_str))


def main():
    args = get_args()

    print(args)

    # find backend
    backend = get_backend(args.backend)

    # override image format if given
    image_format = args.data_format if args.data_format else backend.image_format()

    # dataset to use
    wanted_dataset, preprocessor, postprocessor, kwargs = SUPPORTED_DATASETS[args.dataset]
    ds = wanted_dataset(data_path=args.dataset_path,
                        image_list=args.dataset_list,
                        name=args.dataset,
                        image_format=image_format,
                        pre_process=preprocessor,
                        use_cache=args.cache,
                        count=args.count, **kwargs)

    # load model to backend
    model = backend.load(args.model, inputs=args.inputs, outputs=args.outputs)

    final_results = {
        "runtime": model.name(),
        "version": model.version(),
        "time": int(time.time()),
        "cmdline": str(args),
    }

    #
    # make one pass over the dataset to validate accuracy
    #
    count = args.count if args.count else ds.get_item_count()

    runner = Runner(model, ds, args.threads, post_process=postprocessor)
    runner.start_pool()

    # warmup
    log.info("warmup ...")
    ds.load_query_samples([0])
    for _ in range(100):
        img, _ = ds.get_samples([0])
        _ = backend.predict({backend.inputs[0]: img})

    def issue_query(query_id, query_samples):
        data, label = ds.get_samples(query_samples)
        runner.enqueue(query_id, data, label)

    sut = lg.ConstructSUT("mlperf", issue_query)
    qsl = lg.ConstructQSL("mlperf", count, args.time, ds.load_query_samples, ds.unload_query_samples)
    runs = ["edge"]
    for run in runs:
        log.info("starting {}".format(run))
        result_list = []
        result_dict = {"good": 0, "total": 0}
        runner.start_run(result_list, result_dict)
        start = time.time()
        lg.StartTest(sut, qsl, "--mlperf_scenario " + run)
        add_results(final_results, run, result_dict, result_list, time.time() - start)

    runner.finish()
    lg.DestroyQSL(qsl)
    lg.DestroySUT(sut)

    #
    # write final results
    #
    if args.output:
        with open(args.output, "w") as f:
            json.dump(final_results, f, sort_keys=True, indent=4)


if __name__ == "__main__":
    main()
