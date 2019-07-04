# Copyright 2019 The MLPerf Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""
Python demo showing how to use the MLPerf Inference load generator bindings.
"""

from __future__ import print_function
from absl import app
import mlperf_loadgen
import threading
import time
import numpy


def load_samples_to_ram(query_samples):
    return


def unload_samples_from_ram(query_samples):
    return


# Processes queries in 3 slices that complete at different times.
def process_query_async(query_samples, i_slice):
    time.sleep(.001 * (i_slice + 1))
    responses = []
    samples_to_complete = query_samples[i_slice:len(query_samples):3]
    for s in samples_to_complete:
        responses.append(mlperf_loadgen.QuerySampleResponse(s.id, 0, 0))
    mlperf_loadgen.QuerySamplesComplete(responses)


def issue_query(query_samples):
    threading.Thread(
            target=process_query_async,
            args=(query_samples, 0)).start()
    threading.Thread(
            target=process_query_async,
            args=(query_samples, 1)).start()
    threading.Thread(
            target=process_query_async,
            args=(query_samples, 2)).start()


def process_latencies(latencies_ns):
    print("Average latency: ")
    print(numpy.mean(latencies_ns))
    print("Median latency: ")
    print(numpy.percentile(latencies_ns, 50))
    print("90 percentile latency: ")
    print(numpy.percentile(latencies_ns, 90))


def main(argv):
    settings = mlperf_loadgen.TestSettings()
    settings.scenario = mlperf_loadgen.TestScenario.MultiStreamFree
    settings.mode = mlperf_loadgen.TestMode.PerformanceOnly
    settings.multi_stream_samples_per_query = 4
    settings.target_latency_ns = 100000000
    settings.multi_stream_max_async_queries = 2
    settings.min_query_count = 100
    settings.min_duration_ms = 10000

    sut = mlperf_loadgen.ConstructSUT(issue_query, process_latencies)
    qsl = mlperf_loadgen.ConstructQSL(
        1024, 128, load_samples_to_ram, unload_samples_from_ram)
    mlperf_loadgen.StartTest(sut, qsl, settings)
    mlperf_loadgen.DestroyQSL(qsl)
    mlperf_loadgen.DestroySUT(sut)


if __name__ == '__main__':
    app.run(main)
