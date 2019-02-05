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
        "batch_size": 1,
        "time": 30,
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
    parser.add_argument("--batch_size", type=int, help="batch_size")
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


def execute_parallel(model, ds, count, threads, result_list, result_dict,
                     batch_size=1, check_acc=False, post_process=None):
    """Run inference in parallel."""

    # We want the queue to be large enough to not ever block the feeder but small enough
    # to about if we can not keep up with the processing.
    tasks = Queue(maxsize=threads*5)
    workers = []

    def handle_tasks(tasks_queue):
        """Worker thread."""
        good = 0
        total = 0
        while True:
            qitem = tasks_queue.get()
            if qitem is None:
                # None in the queue indicates the parent want us to exit
                tasks_queue.task_done()
                break
            try:
                if check_acc:
                    # if check_acc is set we want to not include time spend in the queue
                    qitem.start = time.time()

                # run the prediction
                results = model.predict({model.inputs[0]: qitem.img})
                # and keep track of how long it took
                result_list.append(time.time() - qitem.start)

                if check_acc:
                    # check if the result was correct and count the outcome
                    results = results[0]
                    for idx, result in enumerate(results):
                        result = post_process(result)
                        if qitem.label[idx] == result:
                            good += 1
                        total += 1
            except Exception as ex:  # pylint: disable=broad-except
                log.error("execute_parallel thread: %s", ex)

            tasks_queue.task_done()

            # TODO: should we yield here to not starve the feeder ?

        # thread is done
        if check_acc:
            # TODO: this should be under lock
            result_dict["good"] += good
            result_dict["total"] += total

    # create and start worker threads
    # TODO: since we start as many threads as we have cores we might starve
    # the parent if we run on cpu only so it can not feed fast enough ?
    for _ in range(threads):
        worker = threading.Thread(target=handle_tasks, args=(tasks,))
        worker.daemon = True
        workers.append(worker)
        worker.start()

    start = time.time()
    # feed the queue
    try:
        for item in ds.batch(batch_size):
            if item.img.shape[0] < batch_size:
                continue
            tasks.put(item, block=check_acc)
        ret_code = True
    except Exception:  # pylint: disable=broad-except
        # we get here when the queue is full and we'd block. This is a hint that
        # the system is not remotely keeping up. The caller uses this hint to abort this run to.
        ret_code = False

    # exit all threads
    for _ in workers:
        tasks.put(None)
    for worker in workers:
        worker.join()
    end = time.time()
    result_dict["runtime"] = end - start
    return ret_code


def report_result(name, target_latency, final_result, result_list, result_dict, check_acc=False):
    """Record a result in the final_result dict and write it to stdout."""
    result = {}
    # the percentiles we want to record
    percentiles = [50., 80., 90., 95., 99., 99.9]
    buckets = np.percentile(result_list, percentiles).tolist()
    buckets_str = ",".join(["{}:{:.4f}".format(p, b) for p, b in zip(percentiles, buckets)])
    mean = np.mean(result_list)

    # this is what we record for each run
    result["target_latency"] = target_latency
    result["mean"] = mean
    result["runtime"] = result_dict["runtime"]
    result["qps"] = len(result_list) / result_dict["runtime"]
    result["count"] = len(result_list)
    result["percentiles"] = {str(k): v for k, v in zip(percentiles, buckets)}

    # to stdout
    print("{} qps={:.2f}, mean={:.6f}, time={:.2f}, tiles={}".format(
        name, result["qps"], result["mean"], result["runtime"], buckets_str))

    if check_acc:
        # record accuracy if we have it.
        result["good_items"] = result_dict["good"]
        result["total_items"] = result_dict["total"]
        result["accuracy"] = 100. * result_dict["good"] / result_dict["total"]
        print("{} accuracy={:.2f}, good_items={}, total_items={}".format(
            name, result["accuracy"], result["good_items"], result["total_items"]))

    # add the result to the result dict
    final_result[name] = result


def find_qps(prefix, model, ds, count, threads, final_results, target_latency,
             batch_size=1, post_process=None, distribution=None, runtime=10):
    """Scan to find latency bound qps."""
    qps_lower = 1
    qps_upper = 100000
    target_qps = threads * 2 / target_latency
    best_match = None
    best_qps = 0

    result_list = []
    result_dict = {}
    measured_latency = -1
    measured_qps = 0
    while qps_upper - qps_lower > 1:
        name = "{}/{}/{}".format(prefix, target_latency, int(target_qps))
        if distribution:
            distribution(ds.get_item_count(), runtime, target_qps)
        result_list = []
        result_dict = {}
        ret_code = execute_parallel(model, ds, count, threads, result_list, result_dict,
                                    batch_size=batch_size, post_process=post_process)
        report_result(name, target_latency, final_results["scan"][prefix], result_list, result_dict)
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
                # not in 90% of expected latency
                print("^latency meet but qps is off")
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
    report_result(name, target_latency, final_results["results"][prefix], result_list, result_dict)

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
    report_result("check_accuracy", 0., final_results, result_list, result_dict, check_acc=True)

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
