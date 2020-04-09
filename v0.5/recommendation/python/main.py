"""
mlperf inference benchmarking tool
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
from queue import Queue

import mlperf_loadgen as lg
import numpy as np

import dataset
import criteo

# add dlrm code path
try:
    dlrm_dir_path = os.environ['DLRM_DIR']
    sys.path.append(dlrm_dir_path)
except KeyError:
    print("ERROR: Please set DLRM_DIR environment variable to the dlrm code location")
    sys.exit(0)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")

NANO_SEC = 1e9
MILLI_SEC = 1000

# pylint: disable=missing-docstring

# the datasets we support
SUPPORTED_DATASETS = {
    "kaggle":
        (criteo.Criteo, criteo.pre_process_criteo_dlrm, criteo.DlrmPostProcess(),
         {"randomize": 'total',  "memory_map": True}),
    "terabyte":
        (criteo.Criteo, criteo.pre_process_criteo_dlrm, criteo.DlrmPostProcess(),
         {"max_ind_range": 40000000, "sub_sample_rate": 0.0, "randomize": 'total',  "memory_map": True}),
}

# pre-defined command line options so simplify things. They are used as defaults and can be
# overwritten from command line

SUPPORTED_PROFILES = {
    "defaults": {
        "dataset": "terabyte",
        "inputs": "continuous and categorical features",
        "outputs": "probability",
        "backend": "pytorch-native",
        "model": "dlrm",
        "max-batchsize": 2048,
    },
    "dlrm-kaggle-pytorch": {
        "dataset": "kaggle",
        "inputs": "continuous and categorical features",
        "outputs": "probability",
        "backend": "pytorch-native",
        "model": "dlrm",
        "max-batchsize": 128,
    },
    "dlrm-terabyte-pytorch": {
        "dataset": "terabyte",
        "inputs": "continuous and categorical features",
        "outputs": "probability",
        "backend": "pytorch-native",
        "model": "dlrm",
        "max-batchsize": 2048,
    },
}

SCENARIO_MAP = {
    "SingleStream": lg.TestScenario.SingleStream,
    "MultiStream": lg.TestScenario.MultiStream,
    "Server": lg.TestScenario.Server,
    "Offline": lg.TestScenario.Offline,
}

last_timeing = []


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="name of the mlperf model, ie. dlrm")
    parser.add_argument("--model-path", required=True, help="path to the model file")
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS.keys(), help="dataset")
    parser.add_argument("--dataset-path", required=True, help="path to the dataset")
    parser.add_argument("--profile", choices=SUPPORTED_PROFILES.keys(), help="standard profiles")
    parser.add_argument("--scenario", default="SingleStream",
                        help="mlperf benchmark scenario, one of " + str(list(SCENARIO_MAP.keys())))
    parser.add_argument("--test-num-workers", type=int, default=0, help='# of workers reading the data')
    parser.add_argument("--max-batchsize", type=int, help="max batch size in a single inference")
    parser.add_argument("--output", help="test results")
    parser.add_argument("--inputs", help="model inputs")
    parser.add_argument("--outputs", help="model outputs")
    parser.add_argument("--backend", help="runtime to use")
    parser.add_argument("--threads", default=os.cpu_count(), type=int, help="threads")
    parser.add_argument("--qps", type=int, help="target qps")
    parser.add_argument("--cache", type=int, default=0, help="use cache")
    parser.add_argument("--accuracy", action="store_true", help="enable accuracy pass")
    parser.add_argument("--find-peak-performance", action="store_true", help="enable finding peak performance pass")

    # file to use mlperf rules compliant parameters
    parser.add_argument("--config", default="../mlperf.conf", help="mlperf rules config")

    # below will override mlperf rules compliant settings - don't use for official submission
    parser.add_argument("--time", type=int, help="time to scan in seconds")
    parser.add_argument("--count", type=int, help="dataset items to use")
    parser.add_argument("--max-latency", type=float, help="mlperf max latency in pct tile")
    parser.add_argument("--samples-per-query", type=int, help="mlperf multi-stream sample per query")
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

    if args.scenario not in SCENARIO_MAP:
        parser.error("valid scanarios:" + str(list(SCENARIO_MAP.keys())))
    return args


def get_backend(backend):
    if backend == "tensorflow":
        from backend_tf import BackendTensorflow
        backend = BackendTensorflow()
    elif backend == "tflite":
        from backend_tflite import BackendTflite
        backend = BackendTflite()
    elif backend == "onnxruntime":
        from backend_onnxruntime import BackendOnnxruntime
        backend = BackendOnnxruntime()
    elif backend == "null":
        from backend_null import BackendNull
        backend = BackendNull()
    elif backend == "pytorch":
        from backend_pytorch import BackendPytorch
        backend = BackendPytorch()
    elif backend == "pytorch-native":
        from backend_pytorch_native import BackendPytorchNative
        # NOTE: pass model parameters here, the following options are available
        #
        # 1. Criteo Kaggle Display Advertisement Challenge Dataset (see ./bench/dlrm_s_criteo_kaggle.sh)
        # backend = BackendPytorchNative(
        #    m_spa=16,
        #    ln_emb=np.array([1460,583,10131227,2202608,305,24,12517,633,3,93145,5683,8351593,3194,27,14992,5461306,10,5652,2173,4,7046547,18,15,286181,105,142572]),
        #    ln_bot=np.array([13,512,256,64,16]),
        #    ln_top=np.array([367,512,256,1])
        #  )
        #
        # 2. Criteo Terabyte (see ./bench/dlrm_s_criteo_terabyte.sh [--sub-sample=0.875] --max-in-range=10000000)
        # backend = BackendPytorchNative(
        #    m_spa=64,
        #    ln_emb=np.array([10000000,39060,17295,7424,20265,3,7122,1543,63,10000000,3067956,405282,10,2209,11938,155,4,976,14,10000000,10000000,10000000,590152,12973,108,36]),
        #    with sub-sampling 9980333,36084,17217,7378,20134,3,7112,1442,61, 9758201,1333352,313829,10,2208,11156,122,4,970,14, 9994222, 7267859, 9946608,415421,12420,101, 3
        #    ln_bot=np.array([13,512,256,64]),
        #    ln_top=np.array([415,512,512,256,1])
        # )
        #
        # 3. Criteo Terabyte MLPerf training (see ./bench/run_and_time.sh --max-in-range=40000000)
        # backend = BackendPytorchNative(
        #    m_spa=128,
        #    ln_emb=np.array([40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,36]),
        #    ln_bot=([13,512,256,128]),
        #    ln_top=([479,1024,1024,512,256,1])
        # )
        backend = BackendPytorchNative(
            m_spa=16,
            ln_emb=np.array([1460,583,10131227,2202608,305,24,12517,633,3,93145,5683,8351593,3194,27,14992,5461306,10,5652,2173,4,7046547,18,15,286181,105,142572]),
            ln_bot=np.array([13,512,256,64,16]),
            ln_top=np.array([367,512,256,1])
        )
    else:
        raise ValueError("unknown backend: " + backend)
    return backend


class Item:
    """An item that we queue for processing by the thread pool."""

    def __init__(self, query_id, content_id, batch_dense_X, batch_lS_o, batch_lS_i, batch_T=None):
        self.query_id = query_id
        self.content_id = content_id
        self.batch_dense_X = batch_dense_X
        self.batch_lS_o = batch_lS_o
        self.batch_lS_i = batch_lS_i
        self.batch_T = batch_T
        self.start = time.time()

class RunnerBase:
    def __init__(self, model, ds, threads, post_proc=None, max_batchsize=128):
        self.take_accuracy = False
        self.ds = ds
        self.model = model
        self.post_process = post_proc
        self.threads = threads
        self.max_batchsize = max_batchsize
        self.result_timing = []

    def handle_tasks(self, tasks_queue):
        pass

    def start_run(self, result_dict, take_accuracy):
        self.result_dict = result_dict
        self.result_timing = []
        self.take_accuracy = take_accuracy
        self.post_process.start()

    def run_one_item(self, qitem):
        # run the prediction
        processed_results = []
        try:
            results = self.model.predict({self.model.inputs[0]: qitem.img})
            processed_results = self.post_process(results, qitem.content_id, qitem.label, self.result_dict)
            if self.take_accuracy:
                self.post_process.add_results(processed_results)
                self.result_timing.append(time.time() - qitem.start)
        except Exception as ex:  # pylint: disable=broad-except
            src = [self.ds.get_item_loc(i) for i in qitem.content_id]
            log.error("thread: failed on contentid=%s, %s", src, ex)
            # since post_process will not run, fake empty responses
            processed_results = [[]] * len(qitem.query_id)
        finally:
            response_array_refs = []
            response = []
            for idx, query_id in enumerate(qitem.query_id):
                response_array = array.array("B", np.array(processed_results[idx], np.float32).tobytes())
                response_array_refs.append(response_array)
                bi = response_array.buffer_info()
                response.append(lg.QuerySampleResponse(query_id, bi[0], bi[1]))
            lg.QuerySamplesComplete(response)


    def run_one_item_dlrm(self, qitem):
        # run the prediction
        processed_results = []
        try:
            results = self.model.predict(qitem.batch_dense_X, qitem.batch_lS_o, qitem.batch_lS_i)
            processed_results = self.post_process(results, qitem.batch_T, self.result_dict)
            if self.take_accuracy:
                self.post_process.add_results(processed_results)
                self.result_timing.append(time.time() - qitem.start)
        except Exception as ex:  # pylint: disable=broad-except
            log.error("thread: failed, %s", ex)
            # since post_process will not run, fake empty responses
            processed_results = [[]] * len(qitem.query_id)
        finally:
            response_array_refs = []
            response = []
            for idx, query_id in enumerate(qitem.query_id):
                response_array = array.array("B", np.array(processed_results[idx], np.float32).tobytes())
                response_array_refs.append(response_array)
                bi = response_array.buffer_info()
                response.append(lg.QuerySampleResponse(query_id, bi[0], bi[1]))
            lg.QuerySamplesComplete(response)


    def enqueue(self, query_samples):
        idx = [q.index for q in query_samples]
        query_id = [q.id for q in query_samples]
        print('RunnerBase enqueue idx', idx, query_id)

        if len(query_samples) < self.max_batchsize:

            data, label = self.ds.get_samples(idx)
            self.run_one_item(Item(query_id, idx, data, label))
        else:
            bs = self.max_batchsize
            for i in range(0, len(idx), bs):
                data, label = self.ds.get_samples(idx[i:i+bs])
                self.run_one_item(Item(query_id[i:i+bs], idx[i:i+bs], data, label))

    def enqueue_dlrm(self, query_samples):
        idx = [q.index for q in query_samples]
        query_id = [q.id for q in query_samples]

        if len(query_samples) < self.max_batchsize:

            batch_dense_X, batch_lS_o, batch_lS_i, batch_T = self.ds.get_samples(idx)
            self.run_one_item_dlrm(Item(query_id, idx, batch_dense_X, batch_lS_o, batch_lS_i, batch_T))
        else:
            bs = self.max_batchsize
            for i in range(0, len(idx), bs):
                dbatch_dense_X, batch_lS_o, batch_lS_i, batch_T = self.ds.get_samples(idx[i:i+bs])
                self.run_one_item_dlrm(Item(query_id[i:i+bs], idx[i:i+bs], batch_dense_X, batch_lS_o, batch_lS_i, batch_T))


    def finish(self):
        pass


class QueueRunner(RunnerBase):
    def __init__(self, model, ds, threads, post_proc=None, max_batchsize=128):
        super().__init__(model, ds, threads, post_proc, max_batchsize)
        self.tasks = Queue(maxsize=threads * 4)
        self.workers = []
        self.result_dict = {}

        for _ in range(self.threads):
            worker = threading.Thread(target=self.handle_tasks, args=(self.tasks,))
            worker.daemon = True
            self.workers.append(worker)
            worker.start()

    def handle_tasks(self, tasks_queue):
        """Worker thread."""
        while True:
            qitem = tasks_queue.get()
            if qitem is None:
                # None in the queue indicates the parent want us to exit
                tasks_queue.task_done()
                break
            #self.run_one_item(qitem)
            self.run_one_item_dlrm(qitem)
            tasks_queue.task_done()

    def enqueue_dlrm(self, query_samples):
        idx = [q.index for q in query_samples]
        query_id = [q.id for q in query_samples]

        if len(query_samples) < self.max_batchsize:
            batch_dense_X, batch_lS_o, batch_lS_i, batch_T = self.ds.get_samples(idx)
            self.tasks.put(Item(query_id, idx, batch_dense_X, batch_lS_o, batch_lS_i, batch_T))
        else:
            bs = self.max_batchsize
            for i in range(0, len(idx), bs):
                ie = i + bs
                batch_dense_X, batch_lS_o, batch_lS_i, batch_T = self.ds.get_samples(idx[i:ie])
                self.tasks.put(Item(query_id[i:ie], idx[i:ie], batch_dense_X, batch_lS_o, batch_lS_i, batch_T))

    def finish(self):
        # exit all threads
        for _ in self.workers:
            self.tasks.put(None)
        for worker in self.workers:
            worker.join()



def add_results(final_results, name, result_dict, result_list, took, show_accuracy=False):
    percentiles = [50., 80., 90., 95., 99., 99.9]
    buckets = np.percentile(result_list, percentiles).tolist()
    buckets_str = ",".join(["{}:{:.4f}".format(p, b) for p, b in zip(percentiles, buckets)])

    if result_dict["total"] == 0:
        result_dict["total"] = len(result_list)

    # this is what we record for each run
    result = {
        "took": took,
        "mean": np.mean(result_list),
        "percentiles": {str(k): v for k, v in zip(percentiles, buckets)},
        "qps": len(result_list) / took,
        "count": len(result_list),
        "good_items": result_dict["good"],
        "total_items": result_dict["total"],
    }
    acc_str = ""
    if show_accuracy:
        result["accuracy"] = 100. * result_dict["good"] / result_dict["total"]
        acc_str = ", acc={:.3f}%".format(result["accuracy"])
        if "mAP" in result_dict:
            result["mAP"] = 100. * result_dict["mAP"]
            acc_str += ", mAP={:.3f}%".format(result["mAP"])

    # add the result to the result dict
    final_results[name] = result

    # to stdout
    print("{} qps={:.2f}, mean={:.4f}, time={:.3f}{}, queries={}, tiles={}".format(
        name, result["qps"], result["mean"], took, acc_str,
        len(result_list), buckets_str))


def main():
    global last_timeing
    args = get_args()

    log.info(args)

    # find backend
    backend = get_backend(args.backend)

    # --count applies to accuracy mode only and can be used to limit the number of images
    # for testing. For perf model we always limit count to 200.
    count_override = False
    count = args.count
    if count:
        count_override = True

    # dataset to use
    wanted_dataset, pre_proc, post_proc, kwargs = SUPPORTED_DATASETS[args.dataset]

    ds = wanted_dataset(data_path=args.dataset_path,
                        name=args.dataset,
                        pre_process=pre_proc,  # currently an identity function
                        use_cache=args.cache,  # currently not used
                        count=count,
                        test_num_workers=args.test_num_workers,
                        **kwargs)
    # load model to backend
    model = backend.load(args.model_path, inputs=args.inputs, outputs=args.outputs)
    final_results = {
        "runtime": model.name(),
        "version": model.version(),
        "time": int(time.time()),
        "cmdline": str(args),
    }

    config = os.path.abspath(args.config)
    if not os.path.exists(config):
        log.error("{} not found".format(config))
        sys.exit(1)

    if args.output:
        output_dir = os.path.abspath(args.output)
        os.makedirs(output_dir, exist_ok=True)
        os.chdir(output_dir)

    #
    # make one pass over the dataset to validate accuracy
    #
    count = ds.get_item_count()
    # warmup
    ds.load_query_samples([0])

    for _ in range(5):
        batch_dense_X, batch_lS_o, batch_lS_i, batch_T = ds.get_samples([0])
        _ = backend.predict(batch_dense_X, batch_lS_o, batch_lS_i)

    ds.unload_query_samples(None)

    scenario = SCENARIO_MAP[args.scenario]
    runner_map = {
        lg.TestScenario.SingleStream: RunnerBase,
        lg.TestScenario.MultiStream: QueueRunner,
        lg.TestScenario.Server: QueueRunner,
        lg.TestScenario.Offline: QueueRunner
    }

    runner = runner_map[scenario](model, ds, args.threads, post_proc=post_proc, max_batchsize=args.max_batchsize)

    def issue_queries(query_samples):
        runner.enqueue_dlrm(query_samples)

    def flush_queries():
        pass

    def process_latencies(latencies_ns):
        # called by loadgen to show us the recorded latencies
        global last_timeing
        last_timeing = [t / NANO_SEC for t in latencies_ns]

    settings = lg.TestSettings()
    settings.FromConfig(config, args.model, args.scenario)
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
        settings.multi_stream_target_latency_ns = int(args.max_latency * NANO_SEC)

    sut = lg.ConstructSUT(issue_queries, flush_queries, process_latencies)
    qsl = lg.ConstructQSL(count, min(count, 500), ds.load_query_samples, ds.unload_query_samples)

    log.info("starting {}".format(scenario))
    result_dict = {"good": 0, "total": 0, "scenario": str(scenario)}
    runner.start_run(result_dict, args.accuracy)
    lg.StartTest(sut, qsl, settings)

    if not last_timeing:
        last_timeing = runner.result_timing
    if args.accuracy:
        post_proc.finalize(result_dict, ds, output_dir=args.output)
    add_results(final_results, "{}".format(scenario),
                result_dict, last_timeing, time.time() - ds.last_loaded, args.accuracy)

    runner.finish()
    lg.DestroyQSL(qsl)
    lg.DestroySUT(sut)

    #
    # write final results
    #
    if args.output:
        with open("results.json", "w") as f:
            json.dump(final_results, f, sort_keys=True, indent=4)


if __name__ == "__main__":
    main()
