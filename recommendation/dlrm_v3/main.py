# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-strict
"""
mlperf dlrm_v3 inference benchmarking tool.
"""

from utils import (
    get_dataset,
    profiler_or_nullcontext,
    SUPPORTED_DATASETS,
)
from model_family import HSTUModelFamily
from inference_modules import set_is_inference
from data_producer import (
    MultiThreadDataProducer,
    QueryItem,
    SingleThreadDataProducer,
)
from datasets.synthetic_streaming import (
    DLRMv3SyntheticStreamingDataset,
)
from datasets.dataset import Dataset, Samples
from configs import get_embedding_table_config, get_hstu_configs
from generative_recommenders.common import set_dev_mode, set_verbose_level
import torch
import numpy as np
import mlperf_loadgen as lg  # @manual
from typing import Any, Dict, List, Optional, Tuple, Union
import time
import sys
import os
import argparse
import array
import logging
import random
import threading

logging.basicConfig(level=logging.INFO)

# pyre-ignore [21]

logger: logging.Logger = logging.getLogger("main")

torch.multiprocessing.set_start_method("spawn", force=True)

USER_CONF = f"{os.path.dirname(__file__)}/user.conf"


SCENARIO_MAP = {  # pyre-ignore [5]
    "Server": lg.TestScenario.Server,
    "Offline": lg.TestScenario.Offline,
}


def get_args():  # pyre-ignore [3]
    """
    Parse command-line arguments for the MLPerf DLRMv3 inference benchmark.

    Returns:
        argparse.Namespace: Parsed arguments including dataset selection, model path,
            scenario configuration, batch size, and various benchmark parameters.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="sampled-streaming-100b", choices=SUPPORTED_DATASETS, help="name of the dataset"
    )
    parser.add_argument(
        "--model-path", type=str, default="", help="path to the model checkpoint. Example: /home/username/data/dlrmv3_trained_checkpoint/"
    )
    parser.add_argument(
        "--scenario-name", type=str, default="Server", choices={"Server", "Offline"}, help="inference benchmark scenario"
    )
    parser.add_argument(
        "--batchsize", type=int, default=10, help="batch size used in the benchmark"
    )
    parser.add_argument(
        "--output-trace", type=bool, default=False, help="Whether to output trace"
    )
    parser.add_argument(
        "--data-producer-threads", type=int, default=8, help="Number of threads used in data producer"
    )
    parser.add_argument(
        "--compute-eval", type=bool, default=False, help="If true, will run AccuracyOnly mode and outputs both predictions and labels for accuracy calcuations"
    )
    parser.add_argument(
        "--find-peak-performance", type=bool, default=False, help="Whether to find peak performance in the benchmark"
    )
    parser.add_argument(
        "--dataset-path-prefix", type=str, default=f"/home/{os.getlogin()}/dlrmv3_dataset/", help="Prefix to the dataset path. Example: /home/username/"
    )
    parser.add_argument(
        "--warmup-ratio", type=float, default=0.1, help="The ratio of the dataset used to warmup SUT"
    )
    parser.add_argument(
        "--num-queries", type=int, default=None, help="Number of queries to run in the benchmark"
    )
    parser.add_argument(
        "--target-qps", type=int, default=1000, help="Benchmark target QPS. Needs to be tuned for different implementations to balance latency and throughput"
    )
    parser.add_argument(
        "--numpy-rand-seed", type=int, default=123, help="Numpy random seed"
    )
    parser.add_argument(
        "--sparse-quant", type=bool, default=False, help="Whether to quantize sparse arch"
    )
    parser.add_argument(
        "--dataset-percentage", type=float, default=0.0001, help="Percentage of the dataset to run in the benchmark"
    )
    args, unknown_args = parser.parse_known_args()
    logger.warning(f"unknown_args: {unknown_args}")
    return args


class Runner:
    """
    Orchestrates inference benchmark execution.

    Manages data production, model inference, and result collection for
    MLPerf LoadGen-based benchmarking.

    Args:
        model: The HSTU model family instance for making predictions.
        ds: Dataset to fetch samples from.
        num_queries: Total number of queries to process.
        data_producer_threads: Number of threads for data loading (default: 1).
        batchsize: Batch size for inference (default: 128).
        compute_eval: Whether to compute evaluation metrics (default: False).
    """

    def __init__(
        self,
        model: HSTUModelFamily,
        ds: Dataset,
        num_queries: int,
        data_producer_threads: int = 1,
        batchsize: int = 128,
        compute_eval: bool = False,
    ) -> None:
        self.model = model
        if data_producer_threads == 1:
            self.data_producer: Union[
                MultiThreadDataProducer, SingleThreadDataProducer
            ] = SingleThreadDataProducer(ds, self.run_one_item)
        else:
            self.data_producer = MultiThreadDataProducer(
                ds, data_producer_threads, self.run_one_item
            )
        self.batchsize = batchsize
        self.compute_eval = compute_eval
        self.reset_states(num_queries=num_queries)

    def reset_states(self, num_queries: int) -> None:
        """
        Reset all internal state for a new benchmark run.

        Args:
            num_queries: Number of queries expected in this run.
        """
        self.result_timing: List[Dict[str, float]] = []
        self.result_batches: List[int] = []
        self.current_query_ids: List[int] = []
        self.current_content_ids: List[int] = []
        self.current_t0: List[float] = []
        self.num_queries: int = num_queries
        self.processed_queries: int = 0

    def run_one_item(self, qitem: QueryItem) -> None:
        """
        Process a single query item through model inference.

        Runs prediction, records timing metrics, and sends results back to LoadGen.

        Args:
            qitem: Query item containing batch of samples to process.
        """
        try:
            t0_prediction: float = time.time()
            prediction_output = self.model.predict(qitem.samples)
            dt_prediction: float = time.time() - t0_prediction
            assert prediction_output is not None
            (
                mt_target_preds,
                mt_target_labels,
                mt_target_weights,
                dt_sparse,
                dt_dense,
            ) = prediction_output
            if self.compute_eval:
                assert mt_target_labels is not None
                assert mt_target_weights is not None
            self.result_timing.append(
                {
                    "total": time.time() - qitem.start,
                    "prediction": dt_prediction,
                    "queue": qitem.dt_queue,
                    "batching": qitem.dt_batching,
                    "sparse": dt_sparse,
                    "dense": dt_dense,
                }
            )
            self.result_batches.append(len(qitem.query_ids))
        except Exception as ex:  # pylint: disable=broad-except
            logger.error("thread: failed, %s", ex)
        finally:
            candidate_size = mt_target_preds.size(1) // len(qitem.query_ids)
            if not self.compute_eval:
                for i, query_id in enumerate(qitem.query_ids):
                    query_mt_target_preds = (
                        mt_target_preds[  # pyre-ignore [61]
                            0,
                            candidate_size * i: candidate_size * (i + 1),
                        ]
                        .view(-1)
                        .float()
                        .numpy()
                    )
                    ts_idx_val = float(
                        qitem.ts_idx) if qitem.ts_idx is not None else -1.0
                    query_idx_val = float(
                        qitem.query_idx[i]) if qitem.query_idx is not None else -1.0
                    np_array = np.concatenate(
                        [
                            np.array([ts_idx_val]).astype(np.float32),
                            np.array([query_idx_val]).astype(np.float32),
                            query_mt_target_preds,
                        ]
                    )
                    response_array = array.array("B", np_array.tobytes())
                    bi = response_array.buffer_info()
                    # since we send buffer to loadgen, needs `response_array`
                    # in memory during send
                    lg.QuerySamplesComplete(
                        [lg.QuerySampleResponse(query_id, bi[0], bi[1])]
                    )
            else:
                for i, query_id in enumerate(qitem.query_ids):
                    query_mt_target_preds = (
                        mt_target_preds[  # pyre-ignore [61]
                            0, candidate_size * i: candidate_size * (i + 1)
                        ]
                        .view(-1)
                        .float()
                        .numpy()
                    )
                    query_mt_target_labels = (
                        mt_target_labels[  # pyre-ignore [16,61]
                            0, candidate_size * i: candidate_size * (i + 1)
                        ]
                        .view(-1)
                        .float()
                        .numpy()
                    )
                    query_mt_target_weights = (
                        mt_target_weights[  # pyre-ignore [61]
                            0, candidate_size * i: candidate_size * (i + 1)
                        ]
                        .view(-1)
                        .float()
                        .numpy()
                    )
                    ts_idx_val = float(
                        qitem.ts_idx) if qitem.ts_idx is not None else -1.0
                    query_idx_val = float(
                        qitem.query_idx[i]) if qitem.query_idx is not None else -1.0
                    np_array = np.concatenate(
                        [
                            np.array([ts_idx_val]).astype(np.float32),
                            np.array([query_idx_val]).astype(np.float32),
                            query_mt_target_preds,
                            query_mt_target_labels,
                            query_mt_target_weights,
                            np.array([candidate_size]).astype(np.float32),
                        ]
                    )
                    response_array = array.array("B", np_array.tobytes())
                    bi = response_array.buffer_info()
                    # since we send buffer to loadgen, needs `response_array`
                    # in memory during send
                    lg.QuerySamplesComplete(
                        [lg.QuerySampleResponse(query_id, bi[0], bi[1])]
                    )

    def enqueue(self, query_samples, t0: float) -> None:  # pyre-ignore [2]
        """
        Enqueue query samples for batch processing.

        Collects samples until batch size is reached, then dispatches to data producer.

        Args:
            query_samples: List of LoadGen query sample objects.
            t0: Timestamp when this batch started.
        """
        self.current_query_ids.extend([q.id for q in query_samples])
        self.current_content_ids.extend([q.index for q in query_samples])
        self.current_t0.append(t0)
        self.processed_queries += len(query_samples)
        t0: float = min(self.current_t0)
        dt_queue: float = max(self.current_t0) - min(self.current_t0)
        if (
            self.processed_queries >= self.num_queries
            or len(self.current_query_ids) >= self.batchsize
        ):
            for i in range(len(self.current_query_ids) // self.batchsize):
                self.data_producer.enqueue(
                    query_ids=self.current_query_ids[
                        i * self.batchsize: (i + 1) * self.batchsize
                    ],
                    content_ids=self.current_content_ids[
                        i * self.batchsize: (i + 1) * self.batchsize
                    ],
                    t0=t0,
                    dt_queue=dt_queue,
                )
            remaining_s: int = len(self.current_query_ids) % self.batchsize
            if remaining_s > 0:
                self.data_producer.enqueue(
                    query_ids=self.current_query_ids[-remaining_s:],
                    content_ids=self.current_content_ids[-remaining_s:],
                    t0=t0,
                    dt_queue=dt_queue,
                )
            self.current_query_ids = []
            self.current_content_ids = []
            self.current_t0 = []

    def finish(self) -> None:
        """Signal data producer to finish and wait for completion."""
        self.data_producer.finish()


def add_results(
    final_results: Dict[str, Any],
    result_timing: List[Dict[str, float]],
    result_batches: List[int],
) -> None:
    """
    Aggregate and log benchmark results.

    Computes percentile statistics and QPS metrics from timing data.

    Args:
        final_results: Dictionary to populate with aggregated results.
        result_timing: List of timing dictionaries for each batch.
        result_batches: List of batch sizes processed.
    """
    percentiles: list[float] = [50.0, 80.0, 90.0, 95.0, 99.0, 99.9]
    buckets_dict: Dict[str, List[float]] = {}
    buckets_str_dict: Dict[str, str] = {}
    total_timing: list[float] = [result["total"] for result in result_timing]
    for key in ["total", "prediction", "queue", "batching", "sparse", "dense"]:
        timing: list[float] = [result[key] for result in result_timing]
        buckets: List[float] = np.percentile(timing, percentiles).tolist()
        buckets_str: str = ",".join(
            ["| {}:{:.4f}| ".format(p, b)
             for p, b in zip(percentiles, buckets)]
        )
        buckets_dict[key] = buckets
        buckets_str_dict[key] = buckets_str
    total_batches = sum(result_batches)

    final_results["good"] = len(total_timing)
    final_results["avg_time"] = np.mean(total_timing)
    final_results["percentiles"] = {
        str(k): v for k, v in zip(percentiles, buckets_dict["total"])
    }
    final_results["qps"] = total_batches / final_results["took"]
    final_results["count"] = total_batches

    for i, timing in enumerate(result_timing):
        logger.warning(f"timing of {i}: {timing}")

    logger.warning(
        "{} qps={:.2f}, avg_query_time={:.4f}, time={:.3f}, queries={}, tiles={}".format(
            final_results["scenario"],
            final_results["qps"],
            final_results["avg_time"],
            final_results["took"],
            len(result_timing),
            buckets_str_dict["total"],
        )
    )
    for key in ["prediction", "queue", "batching", "sparse", "dense"]:
        logger.warning(f"{key}: {buckets_str_dict[key]}")


def get_num_queries(
    input_size: Optional[int],
    one_pass_size: int,
    scenario_name: str,
    offline_target_qps: int,
    target_duration: float,
) -> int:
    """
    Determine the number of queries to run based on scenario and settings.

    Args:
        input_size: User-specified query count (None to use defaults).
        one_pass_size: Size of one complete pass through the dataset.
        scenario_name: MLPerf scenario name ('Server' or 'Offline').
        offline_target_qps: Target QPS for offline scenario.
        target_duration: Target duration in milliseconds.

    Returns:
        Number of queries to execute in the benchmark run.
    """
    if scenario_name == "Offline":
        # consistent with
        # https://github.com/mlcommons/inference/blob/8999c4d686f6e4a180da14597c97063fce7c9f33/loadgen/test_settings_internal.cc#L147
        return int(1.1 * target_duration / 1000 * offline_target_qps)
    else:
        if input_size is None:
            return one_pass_size
        return input_size


class StreamingQuerySampler:
    """
    Sampler for streaming dataset
    The execution order is determined by `StreamingQuerySampler.run_order`, not by the QSL or input query ID.
    This ensures that queries are executed according to their timestamp constraints.
    """

    def __init__(
        self,
        ds: DLRMv3SyntheticStreamingDataset,
        dataset_percentage: float,
        scenario_name: str,
        offline_target_qps: int,
        target_duration: float,
        input_queries: Optional[int] = None,
        compute_eval: bool = False,
    ) -> None:
        self.ds: DLRMv3SyntheticStreamingDataset = ds
        self.ds.is_inference = True
        self.inference_ts: int = self.ds.total_ts - self.ds.train_ts
        self.start_ts: int = self.ds.train_ts
        self.dataset_percentage: float = dataset_percentage
        self.num_unique_requests: List[int] = self.get_num_unique_requests(
            warmup_ratio=1.0
        )
        self.num_unique_requests_cumsum: List[int] = np.cumsum(
            self.num_unique_requests
        ).tolist()
        self.total_requests: int = sum(self.num_unique_requests)
        self.run_order: List[List[int]] = self.build_random_exec_order()
        self.ts_idx: int = 0
        self.ts_processed_cnt: int = 0
        self.last_loaded: float = -1.0
        num_queries: int = get_num_queries(
            input_size=input_queries,
            one_pass_size=self.total_requests,
            scenario_name=scenario_name,
            offline_target_qps=offline_target_qps,
            target_duration=target_duration,
        )
        logger.warning(
            f"StreamingQuerySampler constructred to handle {num_queries} queries"
        )
        self.num_repeats: int = (
            max(1, num_queries // self.total_requests) if not compute_eval else 1
        )
        self.remaining_queries: int = (
            num_queries % self.total_requests if not compute_eval else 0
        )
        self._lock = threading.Lock()

    def get_num_unique_requests(self, warmup_ratio: float) -> List[int]:
        """
        Calculate number of unique requests per timestamp.

        Args:
            warmup_ratio: Fraction of users to include in warmup.

        Returns:
            List of request counts per timestamp.
        """
        num_unique_requests = [
            int(
                self.ds.ts_to_users_cumsum[t][-1]
                * self.dataset_percentage
                * warmup_ratio
            )
            for t in range(self.start_ts, self.start_ts + self.inference_ts)
        ]
        return num_unique_requests

    def build_random_exec_order(self) -> List[List[int]]:
        """
        Build randomized execution order for each timestamp.

        Returns:
            List of shuffled index lists, one per timestamp.
        """
        order = []
        for req_size in self.num_unique_requests:
            within_ts_order = list(range(req_size))
            random.shuffle(within_ts_order)
            order.append(within_ts_order)
        return order

    def init_sut(self) -> None:
        """Initialize System Under Test state for a new benchmark run."""
        self.ts_idx = 0
        self.ts_processed_cnt = 0
        self.ds.set_ts(self.start_ts)

    def load_query_samples(self, query_ids: List[Optional[int]]) -> None:
        """
        Load query samples into memory for the benchmark.

        Args:
            query_ids: List of query identifiers to load.
        """
        length = len(query_ids)
        ts_idx: int = 0
        while self.num_unique_requests_cumsum[ts_idx] < length:
            ts_idx += 1
        for i in range(0, ts_idx):
            self.ds.set_ts(i + self.start_ts)
            self.ds.load_query_samples(self.run_order[i])
        self.ds.set_ts(ts_idx + self.start_ts)
        delta_length = (
            length
            if ts_idx == 0
            else length - self.num_unique_requests_cumsum[ts_idx - 1]
        )
        self.ds.load_query_samples(self.run_order[ts_idx][:delta_length])
        self.init_sut()
        self.last_loaded = time.time()

    def unload_query_samples(self, sample_list: List[int]) -> None:
        """
        Unload query samples from memory.

        Args:
            sample_list: List of sample identifiers to unload.
        """
        self.ds.unload_query_samples(sample_list)

    def get_samples(
        self, id_list: List[int]
    ) -> List[Tuple[Samples, int, List[int]]]:
        """
        Get samples for a batch of queries, handling timestamp boundaries.

        Args:
            id_list: List of query identifiers.

        Returns:
            List of tuples, each containing:
                - Samples object for the batch.
                - ts_idx: The timestamp index for this batch.
                - query_idx: List of query indices for this batch.
        """
        batch_size: int = len(id_list)
        with self._lock:
            curr_ts_idx: int = self.ts_idx
            curr_ts_unique_requests: int = self.num_unique_requests[curr_ts_idx]
            curr_ts_queries: int = curr_ts_unique_requests * self.num_repeats
            if curr_ts_idx == self.inference_ts - 1:
                curr_ts_queries += self.remaining_queries
            begin_query_idx: int = self.ts_processed_cnt
            end_query_idx: int = min(
                begin_query_idx + batch_size, curr_ts_queries)
            begin_request_idx: int = begin_query_idx % curr_ts_unique_requests
            end_request_idx: int = end_query_idx % curr_ts_unique_requests
            if begin_query_idx + batch_size >= curr_ts_queries:
                self.ts_idx += 1
                self.ts_processed_cnt = begin_query_idx + batch_size - curr_ts_queries
            else:
                self.ts_processed_cnt = begin_query_idx + batch_size
        # requests of current ts
        outputs: List[Tuple[Samples, int, List[int]]] = []
        if end_request_idx > begin_request_idx:
            query_idx_list: List[int] = self.run_order[curr_ts_idx][begin_request_idx:end_request_idx]
            output: Samples = self.ds.get_samples_with_ts(
                query_idx_list,
                curr_ts_idx + self.start_ts,
            )
            outputs.append((output, curr_ts_idx, query_idx_list))
        else:
            if begin_request_idx < curr_ts_unique_requests:
                query_idx_list: List[int] = self.run_order[curr_ts_idx][begin_request_idx:]
                output: Samples = self.ds.get_samples_with_ts(
                    query_idx_list,
                    curr_ts_idx + self.start_ts,
                )
                outputs.append((output, curr_ts_idx, query_idx_list))
            if end_request_idx > 0:
                query_idx_list = self.run_order[curr_ts_idx][0:end_request_idx]
                output = self.ds.get_samples_with_ts(
                    query_idx_list,
                    curr_ts_idx + self.start_ts,
                )
                outputs.append((output, curr_ts_idx, query_idx_list))
        # requests of next ts
        if begin_query_idx + batch_size > curr_ts_queries:
            query_idx_list = self.run_order[curr_ts_idx + 1][
                : begin_query_idx + batch_size - curr_ts_queries
            ]
            output: Samples = self.ds.get_samples_with_ts(
                query_idx_list,
                curr_ts_idx + 1 + self.start_ts,
            )
            outputs.append((output, curr_ts_idx + 1, query_idx_list))
        return outputs

    def get_item_count(self) -> int:
        """
        Get total number of items in the dataset.

        Returns:
            Total request count across all timestamps.
        """
        return self.total_requests


def run(
    dataset: str = "sampled-streaming-100b",
    model_path: str = "",
    scenario_name: str = "Server",
    batchsize: int = 16,
    output_trace: bool = False,
    data_producer_threads: int = 4,
    compute_eval: bool = False,
    find_peak_performance: bool = False,
    dataset_path_prefix: str = "",
    warmup_ratio: float = 0.1,
    target_qps: Optional[int] = None,
    num_queries: Optional[int] = None,
    numpy_rand_seed: int = 123,
    sparse_quant: bool = False,
    dataset_percentage: float = 1.0,
) -> None:
    """
    Execute the MLPerf DLRMv3 inference benchmark.

    Sets up the model, dataset, and LoadGen infrastructure, then runs
    warmup and official benchmark phases.

    Args:
        dataset: Dataset identifier to use.
        model_path: Path to model checkpoint directory.
        scenario_name: MLPerf scenario ('Server' or 'Offline').
        batchsize: Batch size for inference.
        output_trace: Whether to output profiling traces.
        data_producer_threads: Number of data loading threads.
        compute_eval: Whether to compute accuracy metrics.
        find_peak_performance: Whether to run peak performance finding mode.
        dataset_path_prefix: Prefix path for dataset files.
        warmup_ratio: Fraction of data to use for warmup.
        target_qps: Target queries per second.
        num_queries: Number of queries to run (None for automatic).
        numpy_rand_seed: Random seed for reproducibility.
        sparse_quant: Whether to quantize sparse embeddings.
        dataset_percentage: Fraction of dataset to use.
    """
    set_dev_mode(False)
    if scenario_name not in SCENARIO_MAP:
        raise NotImplementedError(
            "valid scanarios:" + str(list(SCENARIO_MAP.keys())))
    scenario = SCENARIO_MAP[scenario_name]
    np.random.seed(numpy_rand_seed)
    random.seed(numpy_rand_seed)

    hstu_config = get_hstu_configs(dataset)
    hstu_config.max_num_candidates = hstu_config.max_num_candidates_inference
    table_config = get_embedding_table_config(dataset)
    set_is_inference(is_inference=not compute_eval)

    user_conf = os.path.abspath(USER_CONF)
    if not os.path.exists(user_conf):
        logger.error("{} not found".format(user_conf))
        sys.exit(1)

    settings = lg.TestSettings()
    settings.FromConfig(user_conf, model_path, scenario_name)
    settings.scenario = scenario
    settings.mode = lg.TestMode.PerformanceOnly
    if compute_eval:
        settings.mode = lg.TestMode.AccuracyOnly
    if find_peak_performance:
        settings.mode = lg.TestMode.FindPeakPerformance
    if target_qps:
        settings.server_target_qps = float(target_qps)
        settings.offline_expected_qps = float(target_qps)

    model_family = HSTUModelFamily(
        hstu_config=hstu_config,
        table_config=table_config,
        sparse_quant=sparse_quant,
        output_trace=output_trace,
        compute_eval=compute_eval,
    )
    is_streaming: bool = "streaming" in dataset
    dataset, kwargs = get_dataset(dataset, dataset_path_prefix)

    ds: Dataset = dataset(
        hstu_config=hstu_config,
        embedding_config=table_config,
        is_inference=not compute_eval,
        **kwargs,
    )
    if is_streaming:
        ds = StreamingQuerySampler(  # pyre-ignore
            ds=ds,  # pyre-ignore [6]
            dataset_percentage=dataset_percentage,
            input_queries=num_queries,
            compute_eval=compute_eval,
            scenario_name=scenario_name,
            offline_target_qps=settings.offline_expected_qps,
            target_duration=settings.min_duration_ms,
        )
    model_family.load(model_path)

    # warmup
    for autotune_bs in range(batchsize, 0, -1):
        logger.warning(f"Autotune for batch size {autotune_bs}")
        warmup_ids = list(range(autotune_bs))
        ds.load_query_samples(warmup_ids)
        for _ in range(4 * int(os.environ.get("WORLD_SIZE", 1))):
            if is_streaming:
                ds.init_sut()  # pyre-ignore [16]
            result = ds.get_samples(warmup_ids)
            if isinstance(result, list) and len(
                    result) > 0 and isinstance(result[0], tuple):
                for sample, _, _ in result:
                    model_family.predict(sample)
            elif isinstance(result, Samples):
                model_family.predict(result)
            else:
                for s in result:
                    model_family.predict(s)
        ds.unload_query_samples(None)
    for h in logger.handlers:
        h.flush()
    logger.info("Model forward warmup done")

    count = int(
        ds.get_item_count() * dataset_percentage
        if not is_streaming
        else ds.get_item_count()
    )
    train_size: int = 0
    if compute_eval:
        count = count - train_size

    runner: Runner = Runner(
        model_family,
        ds,
        data_producer_threads=data_producer_threads,
        batchsize=batchsize,
        compute_eval=compute_eval,
        num_queries=count,
    )

    def issue_queries(query_samples) -> None:  # pyre-ignore [2]
        if compute_eval:
            for sample in query_samples:
                sample.index = sample.index + train_size
        runner.enqueue(query_samples, time.time())

    def load_query_samples(query_ids: List[int]) -> None:
        if compute_eval:
            query_ids = [q + train_size for q in query_ids]
        ds.load_query_samples(query_ids)

    def flush_queries() -> None:
        pass

    if scenario == lg.TestScenario.Server:
        # inference benchmark warmup
        if is_streaming:
            ds.init_sut()
            warmup_count: int = sum(
                ds.get_num_unique_requests(  # pyre-ignore [16]
                    warmup_ratio=warmup_ratio
                )
            )
        else:
            warmup_count: int = int(count * warmup_ratio)
        runner.reset_states(num_queries=warmup_count)
        final_results = {
            "runtime": model_family.name(),
            "version": model_family.version(),
            "time": int(time.time()),
            "scenario": str(scenario),
        }
        settings.min_query_count = warmup_count
        settings.max_query_count = warmup_count
        sut = lg.ConstructSUT(issue_queries, flush_queries)
        qsl = lg.ConstructQSL(
            warmup_count,
            warmup_count,
            load_query_samples,
            ds.unload_query_samples,
        )
        with profiler_or_nullcontext(enabled=output_trace, with_stack=False):
            logger.info(
                f"starting warmup {scenario} with {warmup_count} queries")
            lg.StartTest(sut, qsl, settings)
            lg.DestroyQSL(qsl)
            lg.DestroySUT(sut)

    # official run
    if is_streaming:
        ds.init_sut()
    final_results = {
        "runtime": model_family.name(),
        "version": model_family.version(),
        "time": int(time.time()),
        "scenario": str(scenario),
    }
    query_size: int = get_num_queries(
        input_size=num_queries,
        one_pass_size=count,
        scenario_name=scenario_name,
        offline_target_qps=settings.offline_expected_qps,
        target_duration=settings.min_duration_ms,
    )
    settings.min_query_count = query_size
    settings.max_query_count = query_size
    runner.reset_states(num_queries=query_size if not compute_eval else count)
    sut = lg.ConstructSUT(issue_queries, flush_queries)
    qsl = lg.ConstructQSL(
        count,
        count,
        load_query_samples,
        ds.unload_query_samples,
    )
    with profiler_or_nullcontext(enabled=output_trace, with_stack=False):
        logger.info(
            f"starting {scenario} with {query_size} queries and {query_size // count} repeats"
        )
        lg.StartTest(sut, qsl, settings)
        runner.finish()
        final_results["took"] = time.time() - ds.last_loaded
        lg.DestroyQSL(qsl)
        lg.DestroySUT(sut)

    add_results(
        final_results,
        runner.result_timing,
        runner.result_batches,
    )
    # If multiple subprocesses are running the model send a signal to stop them
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        model_family.predict(None)


def main() -> None:
    set_verbose_level(1)
    args = get_args()
    logger.info(args)
    run(
        dataset=args.dataset,
        model_path=args.model_path,
        scenario_name=args.scenario_name,
        batchsize=args.batchsize,
        output_trace=args.output_trace,
        data_producer_threads=args.data_producer_threads,
        compute_eval=args.compute_eval,
        find_peak_performance=args.find_peak_performance,
        dataset_path_prefix=args.dataset_path_prefix,
        warmup_ratio=args.warmup_ratio,
        target_qps=args.target_qps,
        num_queries=args.num_queries,
        numpy_rand_seed=args.numpy_rand_seed,
        sparse_quant=args.sparse_quant,
        dataset_percentage=args.dataset_percentage,
    )


if __name__ == "__main__":
    main()
