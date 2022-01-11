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
from multiprocessing import JoinableQueue

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
         {"randomize": 'total',  "memory_map": True}),
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
    "dlrm-kaggle-onnxruntime": {
        "dataset": "kaggle",
        "inputs": "continuous and categorical features",
        "outputs": "probability",
        "backend": "onnxruntime",
        "model": "dlrm",
        "max-batchsize": 128,
    },
    "dlrm-terabyte-onnxruntime": {
        "dataset": "terabyte",
        "inputs": "continuous and categorical features",
        "outputs": "probability",
        "backend": "onnxruntime",
        "model": "dlrm",
        "max-batchsize": 2048,
    },
    "tf_dlrm-kaggle-tensorflow": {
        "dataset": "kaggle",
        "inputs": "continuous and categorical features",
        "outputs": "probability",
        "backend": "tensorflow",
        "model": "tf_dlrm",
        "max-batchsize": 128,
    },
    "tf_dlrm-terabyte-tensorflow": {
        "dataset": "terabyte",
        "inputs": "continuous and categorical features",
        "outputs": "probability",
        "backend": "tensorflow",
        "model": "tf_dlrm",
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
    parser.add_argument("--max-ind-range", type=int, default=-1)
    parser.add_argument("--data-sub-sample-rate", type=float, default=0.0)
    parser.add_argument("--mlperf-bin-loader", action='store_true', default=False)
    parser.add_argument("--max-batchsize", type=int, help="max batch size in a single inference")
    parser.add_argument("--output", help="test results")
    parser.add_argument("--inputs", help="model inputs (currently not used)")
    parser.add_argument("--outputs", help="model outputs (currently not used)")
    parser.add_argument("--backend", help="runtime to use")
    parser.add_argument("--use-gpu", action="store_true", default=False)
    parser.add_argument("--threads", default=os.cpu_count(), type=int, help="threads")
    parser.add_argument("--cache", type=int, default=0, help="use cache (currently not used)")
    parser.add_argument("--accuracy", action="store_true", help="enable accuracy pass")
    parser.add_argument("--find-peak-performance", action="store_true", help="enable finding peak performance pass")

    # file to use mlperf rules compliant parameters
    parser.add_argument("--mlperf_conf", default="mlperf.conf", help="mlperf rules config")
    # file for user LoadGen settings such as target QPS
    parser.add_argument("--user_conf", default="user.conf", help="user config for user LoadGen settings such as target QPS")

    # below will override mlperf rules compliant settings - don't use for official submission
    parser.add_argument("--duration", type=int, help="duration in milliseconds (ms)")
    parser.add_argument("--target-qps", type=int, help="target/expected qps")
    parser.add_argument("--max-latency", type=float, help="mlperf max latency in pct tile")
    parser.add_argument("--count-samples", type=int, help="dataset items to use")
    parser.add_argument("--count-queries", type=int, help="number of queries to use")
    parser.add_argument("--samples-per-query-multistream", default=8, type=int, help="query length for multi-stream scenario (in terms of aggregated samples)")
    # --samples-per-query-offline is equivalent to perf_sample_count
    parser.add_argument("--samples-per-query-offline", type=int, default=2048, help="query length for offline scenario (in terms of aggregated samples)")
    parser.add_argument("--samples-to-aggregate-fix", type=int, help="number of samples to be treated as one")
    parser.add_argument("--samples-to-aggregate-min", type=int, help="min number of samples to be treated as one in random query size")
    parser.add_argument("--samples-to-aggregate-max", type=int, help="max number of samples to be treated as one in random query size")
    parser.add_argument("--samples-to-aggregate-quantile-file", type=str, help="distribution quantile used to generate number of samples to be treated as one in random query size")
    parser.add_argument("--samples-to-aggregate-trace-file", type=str, default="dlrm_trace_of_aggregated_samples.txt")
    parser.add_argument("--numpy-rand-seed", type=int, default=123)
    args = parser.parse_args()

    # set random seed
    np.random.seed(args.numpy_rand_seed)

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


def get_backend(backend, dataset, max_ind_range, data_sub_sample_rate, use_gpu):

    if backend == "pytorch-native":
        from backend_pytorch_native import BackendPytorchNative
        # NOTE: pass model parameters here, the following options are available
        if dataset == "kaggle":
            # 1. Criteo Kaggle Display Advertisement Challenge Dataset (see ./bench/dlrm_s_criteo_kaggle.sh)
            backend = BackendPytorchNative(
                m_spa=16,
                ln_emb=np.array([1460,583,10131227,2202608,305,24,12517,633,3,93145,5683,8351593,3194,27,14992,5461306,10,5652,2173,4,7046547,18,15,286181,105,142572]),
                ln_bot=np.array([13,512,256,64,16]),
                ln_top=np.array([367,512,256,1]),
                use_gpu=use_gpu
            )
        elif dataset == "terabyte":
            if max_ind_range == 10000000:
                # 2. Criteo Terabyte (see ./bench/dlrm_s_criteo_terabyte.sh [--sub-sample=0.875] --max-in-range=10000000)
                backend = BackendPytorchNative(
                    m_spa=64,
                    ln_emb=np.array([9980333,36084,17217,7378,20134,3,7112,1442,61, 9758201,1333352,313829,10,2208,11156,122,4,970,14, 9994222, 7267859, 9946608,415421,12420,101, 36]),
                    ln_bot=np.array([13,512,256,64]),
                    ln_top=np.array([415,512,512,256,1]),
                    use_gpu=use_gpu
                )
            elif max_ind_range == 40000000:
                # 3. Criteo Terabyte MLPerf training (see ./bench/run_and_time.sh --max-in-range=40000000)
                backend = BackendPytorchNative(
                    m_spa=128,
                    ln_emb=np.array([39884406,39043,17289,7420,20263,3,7120,1543,63,38532951,2953546,403346,10,2208,11938,155,4,976,14,39979771,25641295,39664984,585935,12972,108,36]),
                    ln_bot=np.array([13,512,256,128]),
                    ln_top=np.array([479,1024,1024,512,256,1]),
                    use_gpu=use_gpu
                )
            else:
                raise ValueError("only --max-ind-range 10M or 40M is supported")
        else:
            raise ValueError("only kaggle|terabyte dataset options are supported")

    elif backend == "onnxruntime":
        from backend_onnxruntime import BackendOnnxruntime

        # NOTE: pass model parameters here, the following options are available
        if dataset == "kaggle":
            # 1. Criteo Kaggle Display Advertisement Challenge Dataset (see ./bench/dlrm_s_criteo_kaggle.sh)
            backend = BackendOnnxruntime(
                m_spa=16,
                ln_emb=np.array([1460,583,10131227,2202608,305,24,12517,633,3,93145,5683,8351593,3194,27,14992,5461306,10,5652,2173,4,7046547,18,15,286181,105,142572]),
                ln_bot=np.array([13,512,256,64,16]),
                ln_top=np.array([367,512,256,1]),
                use_gpu=use_gpu
            )
        elif dataset == "terabyte":
            if max_ind_range == 10000000:
                # 2. Criteo Terabyte (see ./bench/dlrm_s_criteo_terabyte.sh [--sub-sample=0.875] --max-in-range=10000000)
                backend = BackendOnnxruntime(
                    m_spa=64,
                    ln_emb=np.array([9980333,36084,17217,7378,20134,3,7112,1442,61, 9758201,1333352,313829,10,2208,11156,122,4,970,14, 9994222, 7267859, 9946608,415421,12420,101, 36]),
                    ln_bot=np.array([13,512,256,64]),
                    ln_top=np.array([415,512,512,256,1]),
                    use_gpu=use_gpu
                )
            elif max_ind_range == 40000000:
                # 3. Criteo Terabyte MLPerf training (see ./bench/run_and_time.sh --max-in-range=40000000)
                backend = BackendOnnxruntime(
                    m_spa=128,
                    ln_emb=np.array([39884406,39043,17289,7420,20263,3,7120,1543,63,38532951,2953546,403346,10,2208,11938,155,4,976,14,39979771,25641295,39664984,585935,12972,108,36]),
                    ln_bot=np.array([13,512,256,128]),
                    ln_top=np.array([479,1024,1024,512,256,1]),
                    use_gpu=use_gpu
                )
            else:
                raise ValueError("only --max-in-range 10M or 40M is supported")
        else:
            raise ValueError("only kaggle|terabyte dataset options are supported")

    elif backend == "tensorflow":
        from backend_tf import BackendTF
        # NOTE: pass model parameters here, the following options are available
        if dataset == "kaggle":
            # 1. Criteo Kaggle Display Advertisement Challenge Dataset (see ./bench/dlrm_s_criteo_kaggle.sh)
            backend = BackendTF(
                dim_embed=16,
                vocab_sizes=np.array([1460,583,10131227,2202608,305,24,12517,633,3,93145,5683,8351593,3194,27,14992,5461306,10,5652,2173,4,7046547,18,15,286181,105,142572]),
                mlp_bottom=np.array([13,512,256,64,16]),
                mlp_top=np.array([367,512,256,1]),
            )
        elif dataset == "terabyte":
            if max_ind_range == 10000000:
                # 2. Criteo Terabyte (see ./bench/dlrm_s_criteo_terabyte.sh [--sub-sample=0.875] --max-in-range=10000000)
                backend = BackendTF(
                    dim_embed=64,
                    vocab_sizes=np.array([9980333,36084,17217,7378,20134,3,7112,1442,61, 9758201,1333352,313829,10,2208,11156,122,4,970,14, 9994222, 7267859, 9946608,415421,12420,101, 36]),
                    mlp_bottom=np.array([13,512,256,64]),
                    mlp_top=np.array([415,512,512,256,1]),
                )
            elif max_ind_range == 40000000:
                # 3. Criteo Terabyte MLPerf training (see ./bench/run_and_time.sh --max-in-range=40000000)
                backend = BackendTF(
                    dim_embed=128,
                    vocab_sizes=np.array([39884406,39043,17289,7420,20263,3,7120,1543,63,38532951,2953546,403346,10,2208,11938,155,4,976,14,39979771,25641295,39664984,585935,12972,108,36]),
                    mlp_bottom=np.array([13,512,256,128]),
                    mlp_top=np.array([479,1024,1024,512,256,1]),
                )
            else:
                raise ValueError("only --max-in-range 10M or 40M is supported")
        else:
            raise ValueError("only kaggle|terabyte dataset options are supported")

    else:
        raise ValueError("unknown backend: " + backend)
    return backend


class Item:
    """An item that we queue for processing by the thread pool."""

    def __init__(self, query_id, content_id, batch_dense_X, batch_lS_o, batch_lS_i, batch_T=None, idx_offsets=None):
        self.query_id = query_id
        self.content_id = content_id
        self.batch_dense_X = batch_dense_X
        self.batch_lS_o = batch_lS_o
        self.batch_lS_i = batch_lS_i
        self.batch_T = batch_T
        self.idx_offsets = idx_offsets
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
                # NOTE: processed_results returned by DlrmPostProcess store both
                # result = processed_results[idx][0] and target = processed_results[idx][1]
                # also each idx might be a query of samples, rather than a single sample
                # depending on the --samples-to-aggregate* arguments.
                s_idx = qitem.idx_offsets[idx]
                e_idx = qitem.idx_offsets[idx + 1]
                # debug prints
                # print("s,e:",s_idx,e_idx, len(processed_results))
                response_array = array.array("B", np.array(processed_results[s_idx:e_idx], np.float32).tobytes())
                response_array_refs.append(response_array)
                bi = response_array.buffer_info()
                response.append(lg.QuerySampleResponse(query_id, bi[0], bi[1]))
            lg.QuerySamplesComplete(response)

    def enqueue(self, query_samples):
        idx = [q.index for q in query_samples]
        query_id = [q.id for q in query_samples]
        query_len = len(query_samples)

        if query_len < self.max_batchsize:
            batch_dense_X, batch_lS_o, batch_lS_i, batch_T, idx_offsets = self.ds.get_samples(idx)
            self.run_one_item(Item(query_id, idx, batch_dense_X, batch_lS_o, batch_lS_i, batch_T, idx_offsets))
        else:
            bs = self.max_batchsize
            for i in range(0, query_len, bs):
                ie = min(i + bs, query_len)
                batch_dense_X, batch_lS_o, batch_lS_i, batch_T, idx_offsets = self.ds.get_samples(idx[i:ie])
                self.run_one_item(Item(query_id[i:ie], idx[i:ie], batch_dense_X, batch_lS_o, batch_lS_i, batch_T, idx_offsets))

    def finish(self):
        pass


class QueueRunner(RunnerBase):
    def __init__(self, model, ds, threads, post_proc=None, max_batchsize=128):
        super().__init__(model, ds, threads, post_proc, max_batchsize)
        queue_size_multiplier = 4 #(args.samples_per_query_offline + max_batchsize - 1) // max_batchsize)
        self.tasks = JoinableQueue(maxsize=threads * queue_size_multiplier)
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
            self.run_one_item(qitem)
            tasks_queue.task_done()

    def enqueue(self, query_samples):
        idx = [q.index for q in query_samples]
        query_id = [q.id for q in query_samples]
        query_len = len(query_samples)

        if query_len < self.max_batchsize:
            batch_dense_X, batch_lS_o, batch_lS_i, batch_T, idx_offsets = self.ds.get_samples(idx)
            self.tasks.put(Item(query_id, idx, batch_dense_X, batch_lS_o, batch_lS_i, batch_T, idx_offsets))
        else:
            bs = self.max_batchsize
            for i in range(0, query_len, bs):
                ie = min(i + bs, query_len)
                batch_dense_X, batch_lS_o, batch_lS_i, batch_T, idx_offsets = self.ds.get_samples(idx[i:ie])
                self.tasks.put(Item(query_id[i:ie], idx[i:ie], batch_dense_X, batch_lS_o, batch_lS_i, batch_T, idx_offsets))

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
        if "roc_auc" in result_dict:
            result["roc_auc"] = 100. * result_dict["roc_auc"]
            acc_str += ", auc={:.3f}%".format(result["roc_auc"])

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
    backend = get_backend(args.backend, args.dataset, args.max_ind_range, args.data_sub_sample_rate, args.use_gpu)

    # dataset to use
    wanted_dataset, pre_proc, post_proc, kwargs = SUPPORTED_DATASETS[args.dataset]

    # --count-samples can be used to limit the number of samples used for testing
    ds = wanted_dataset(data_path=args.dataset_path,
                        name=args.dataset,
                        pre_process=pre_proc,  # currently an identity function
                        use_cache=args.cache,  # currently not used
                        count=args.count_samples,
                        samples_to_aggregate_fix=args.samples_to_aggregate_fix,
                        samples_to_aggregate_min=args.samples_to_aggregate_min,
                        samples_to_aggregate_max=args.samples_to_aggregate_max,
                        samples_to_aggregate_quantile_file=args.samples_to_aggregate_quantile_file,
                        samples_to_aggregate_trace_file=args.samples_to_aggregate_trace_file,
                        test_num_workers=args.test_num_workers,
                        max_ind_range=args.max_ind_range,
                        sub_sample_rate=args.data_sub_sample_rate,
                        mlperf_bin_loader=args.mlperf_bin_loader,
                        **kwargs)
    # load model to backend
    model = backend.load(args.model_path, inputs=args.inputs, outputs=args.outputs)
    final_results = {
        "runtime": model.name(),
        "version": model.version(),
        "time": int(time.time()),
        "cmdline": str(args),
    }

    mlperf_conf = os.path.abspath(args.mlperf_conf)
    if not os.path.exists(mlperf_conf):
        log.error("{} not found".format(mlperf_conf))
        sys.exit(1)

    user_conf = os.path.abspath(args.user_conf)
    if not os.path.exists(user_conf):
        log.error("{} not found".format(user_conf))
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
        batch_dense_X, batch_lS_o, batch_lS_i, _, _ = ds.get_samples([0])
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
        runner.enqueue(query_samples)

    def flush_queries():
        pass

    def process_latencies(latencies_ns):
        # called by loadgen to show us the recorded latencies
        global last_timeing
        last_timeing = [t / NANO_SEC for t in latencies_ns]

    settings = lg.TestSettings()
    settings.FromConfig(mlperf_conf, args.model_path, args.scenario)
    settings.FromConfig(user_conf, args.model_path, args.scenario)
    settings.scenario = scenario
    settings.mode = lg.TestMode.PerformanceOnly

    if args.accuracy:
        settings.mode = lg.TestMode.AccuracyOnly

    if args.find_peak_performance:
        settings.mode = lg.TestMode.FindPeakPerformance

    if args.duration:
        settings.min_duration_ms = args.duration
        settings.max_duration_ms = args.duration

    if args.target_qps:
        settings.server_target_qps = float(args.target_qps)
        settings.offline_expected_qps = float(args.target_qps)

    if args.count_queries:
        settings.min_query_count = args.count_queries
        settings.max_query_count = args.count_queries

    if args.samples_per_query_multistream:
        settings.multi_stream_samples_per_query = args.samples_per_query_multistream

    if args.max_latency:
        settings.server_target_latency_ns = int(args.max_latency * NANO_SEC)
        settings.multi_stream_expected_latency_ns = int(args.max_latency * NANO_SEC)

    sut = lg.ConstructSUT(issue_queries, flush_queries, process_latencies)
    qsl = lg.ConstructQSL(count, min(count, args.samples_per_query_offline), ds.load_query_samples, ds.unload_query_samples)

    log.info("starting {}".format(scenario))
    result_dict = {"good": 0, "total": 0, "roc_auc": 0, "scenario": str(scenario)}
    runner.start_run(result_dict, args.accuracy)
    lg.StartTest(sut, qsl, settings)

    result_dict["good"] = runner.post_process.good
    result_dict["total"] = runner.post_process.total

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
