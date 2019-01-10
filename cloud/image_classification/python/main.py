"""
mlperf inference benchmarking tool
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# pylint: disable=unused-argument,missing-docstring

import argparse
import json
import logging
import os
import threading
import time
from queue import Queue

import numpy as np

import dataset
import imagenet

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")

# the datasets we support
SUPPORTED_DATASETS = {
    "imagenet": (imagenet.Imagenet, dataset.pre_process_vgg, dataset.post_process_offset1)
}

# pre-canned command line options so simplify things. They are used as defaults and can be
# overwritten by command line
SUPPORTED_PROFILES = {
    "defaults": {
        "dataset": "imagenet",
        "data-format": "NHWC",
        "backend": "tensorflow",
        "cache": 0,
        "batch_size": 1,
        "time": 30,
        "max-latency": "0.010,0.050,0.100,0.200",
    },
    "resnet50-tf": {
        "inputs": "input_tensor:0",
        "outputs": "ArgMax:0",
        "dataset": "imagenet",
        "data-format": "NHWC",
        "backend": "tensorflow",
        "cache": 0,
        "max-latency": "0.010,0.050,0.100,0.200",
    },
    "resnet50-onnxruntime": {
        "inputs": "input_tensor:0",
        "outputs": "ArgMax:0",
        "dataset": "imagenet",
        "data-format": "NHWC",
        "backend": "onnxruntime",
        "cache": 0,
        "max-latency": "0.010,0.050,0.100,0.200",
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
    parser.add_argument("--batch_size", type=int, help="batch_size")
    parser.add_argument("--threads", default=os.cpu_count(), type=int, help="threads")
    parser.add_argument("--time", type=int, help="time to scan in seconds")
    parser.add_argument("--count", type=int, help="dataset items to use")
    parser.add_argument("--max-latency", type=str, help="max latency in 99pct tile")
    parser.add_argument("--cache", type=int, default=0, help="use cache")
    parser.add_argument("--result", help="result file")
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


def execute_parallel(model, ds, count, threads, result_list, result_dict,
                     batch_size=1, check_acc=False, post_process=None):
    """Run inference in parallel."""
    tasks = Queue(maxsize=5*threads)
    workers = []

    def handle_tasks(tasks_queue):
        good = 0
        total = 0
        while True:
            item = tasks_queue.get()
            if item is None:
                # done, exit thread
                tasks_queue.task_done()
                break
            if check_acc:
                item.start = time.time()
            results = model.predict({model.inputs[0]: item.img})
            result_list.append(time.time() - item.start)
            if check_acc:
                results = results[0]
                for idx, result in enumerate(results):
                    result = post_process(result)
                    if item.label[idx] == result:
                        good += 1
                    total += 1
            tasks_queue.task_done()
            # TODO: should we yield here to not starve the feeder ?

        if check_acc:
            # TODO: this should be under lock
            result_dict["good"] += good
            result_dict["total"] += total

    # Create and start worker threads
    # TODO: since we start as many threads as we have cores we might starve
    # the parent if we run on cpu only so it can not feed fast enough ?
    for _ in range(threads):
        worker = threading.Thread(target=handle_tasks, args=(tasks,))
        worker.daemon = True
        workers.append(worker)
        worker.start()

    start = time.time()
    try:
        for item in ds.batch(batch_size):
            if item.img.shape[0] < batch_size:
                continue
            tasks.put(item, block=check_acc)
        ret_code = True
    except Exception:
        # we get here when the queue is full and we'd block. This is a hint that
        # the system is not remotely keeping up. The caller might want to abort this run.
        ret_code = False

    # exit all threads
    for _ in workers:
        tasks.put(None)
    for worker in workers:
        worker.join()
    end = time.time()
    result_dict["runtime"] = end - start
    return ret_code


def report_result(name, final_result, result_list, result_dict, check_acc=None):
    r = {}
    percentiles = [50, 80, 90, 95, 99, 99.9]
    buckets = np.percentile(result_list, percentiles).tolist()
    buckets_str = ",".join(["{}:{:.4f}".format(p, b) for p, b in zip(percentiles, buckets)])
    mean = np.mean(result_list)
    r1 = {str(k): v for k, v in zip(percentiles, buckets)}
    r["mean"] = mean
    r["runtime"] = result_dict["runtime"]
    r["qps"] = len(result_list) / result_dict["runtime"]
    r["count"] = len(result_list)
    r["percentiles"] = r1
    print("{} qps={:.2f}, mean={:.6f}, time={:.2f}, tiles={}".format(
        name, r["qps"], r["mean"], r["runtime"], buckets_str))
    if check_acc:
        r["good_items"] = result_dict["good"]
        r["total_items"] = result_dict["total"]
        r["accuacy"] = 100. * result_dict["good"] / result_dict["total"]
        print("{} accuacy={:.2f}, good_items={}, total_items={}".format(
            name, r["accuacy"], r["good_items"], r["total_items"]))
    final_result[name] = r


def find_qps(prefix, model, ds, count, threads, final_results, target_latency,
             batch_size=1, post_process=None, distribution=None, runtime=10):
    """Scan to find latency bound qps."""
    qps_lower = 1
    qps_upper = 100000
    target_qps = threads * 2 / target_latency
    best_match = None
    best_qps = 0

    while qps_upper - qps_lower > 1:
        name = "{}/{}/{}".format(prefix, target_latency, int(target_qps))
        if distribution:
            distribution(ds.get_item_count(), runtime, target_qps)
        result_list = []
        result_dict = {}
        ret_code = execute_parallel(model, ds, count, threads, result_list, result_dict,
                                    batch_size=batch_size, post_process=post_process)
        report_result(name, final_results["scan"][prefix], result_list, result_dict)
        if not ret_code:
            print("^queue is full, early out")
        measured_latency = np.percentile(result_list, [99.]).tolist()[0]
        measured_qps = int(len(result_list) / result_dict["runtime"])
        if not ret_code or measured_latency > target_latency:
            # did not meet target latency
            qps_upper = min(target_qps, qps_upper)
        else:
            # meet target latency
            if measured_qps < target_qps * 0.9:
                # not in 90% of expected latency ... something must be wrong
                print("^latency meet but qps is off, something very wrong")
                qps_upper = min(target_qps, qps_upper)
            else:
                qps_lower = target_qps
                if measured_qps > best_qps:
                    print("^taken")
                    best_match = measured_qps, result_list, result_dict, measured_latency
                    best_qps = measured_qps
        target_qps = int(round((qps_lower + qps_upper) / 2))
    if best_match:
        measured_qps, result_list, result_dict, measured_latency = best_match
    name = str(target_latency)
    report_result(name, final_results["results"][prefix], result_list, result_dict)

    if measured_latency > target_latency:
        # did not meet latency target
        final_results["results"][prefix][name]["qps"] = -1
        final_results["final_results"][prefix][name] = -1
        print("===RESULT: {} target_latency={} qps={}".format(prefix, name, "FAIL"))
    else:
        # latency target reached
        final_results["final_results"][prefix][name] = target_qps
        print("===RESULT: {} target_latency={} measured_latency={} qps={}".format(
            prefix, name, measured_latency, measured_qps))

    return measured_qps


def main():
    args = get_args()

    print(args)

    # find backend
    if args.backend == "tensorflow":
        from backend_tf import BackendTensorflow
        image_format = "NHWC"
        backend = BackendTensorflow()
    elif args.backend == "onnxruntime":
        from backend_onnxruntime import BackendOnnxruntime
        image_format = "NCHW"
        backend = BackendOnnxruntime()
    elif args.backend == "null":
        from backend_null import BackendNull
        image_format = "NCHW"
        backend = BackendNull()
    else:
        raise ValueError("unknown backend: " + args.backend)

    # override image format if given
    image_format = args.data_format if args.data_format else image_format

    # dataset to use
    wanted_dataset, preprocessor, postprocessor = SUPPORTED_DATASETS[args.dataset]
    ds = wanted_dataset(data_path=args.dataset_path,
                                          image_list=args.dataset_list,
                                          image_format=image_format,
                                          pre_process=preprocessor,
                                          use_cache=args.cache,
                                          count=args.count)

    # load model to backend
    model = backend.load(args.model, inputs=args.inputs, outputs=args.outputs)

    final_results = {
        "runtime": model.name(),
        "version": model.version(),
        "time": int(time.time()),
        "cmdline": str(args),
        "scan": {"linear": {}, "exponential": {}},
        "results": {"linear": {}, "exponential": {}},
        "final_results": {"linear": {}, "exponential": {}},
    }

    #
    # make one pass over the dataset to validate accuracy
    #
    count = args.count if args.count else ds.get_item_count()
    result_list = []
    result_dict = {"good": 0, "total": 0}
    execute_parallel(model, ds, count, args.threads, result_list, result_dict,
                     check_acc=True, post_process=postprocessor)
    report_result("check_accuracy", final_results, result_list, result_dict, True)

    #
    # find max qps with equal request distribution
    #
    for latency in args.max_latency:
        find_qps("linear", model, ds, count, args.threads, final_results, latency,
                 batch_size=args.batch_size, post_process=postprocessor,
                 distribution=ds.generate_linear_trace, runtime=args.time)

    #
    # find max qps with exponential request distribution
    #
    for latency in args.max_latency:
        find_qps("exponential", model, ds, count, args.threads, final_results, latency,
                 batch_size=args.batch_size, post_process=postprocessor,
                 distribution=ds.generate_exp_trace, runtime=args.time)

    print("===FINAL_RESULT:", final_results["final_results"])

    #
    # write final results
    #
    if args.output:
        with open(args.output, "w") as f:
            json.dump(final_results, f, sort_keys=True, indent=4)


if __name__ == "__main__":
    main()
