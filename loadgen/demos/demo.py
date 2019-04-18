"""
Python demo showing how to use the MLPerf Inference load generator bindings.
"""

from __future__ import print_function
from absl import app
import mlperf_loadgen
import threading
import time


def load_samples_to_ram(query_samples):
    return


def unload_samples_from_ram(query_samples):
    return

# Processes queries in 3 slices that complete at different times.
def process_query_async(query_samples, i_slice):
    time.sleep(.01 * (i_slice + 1))
    responses = []
    samples_to_complete = query_samples[i_slice:len(query_samples):3]
    for s in samples_to_complete:
        responses.append(mlperf_loadgen.QuerySampleResponse(s.id, 0, 0))
    mlperf_loadgen.QuerySamplesComplete(responses)


def issue_query(query_samples):
    print(query_samples)
    threading.Thread(
            target=process_query_async,
            args=(query_samples, 0)).start()
    threading.Thread(
            target=process_query_async,
            args=(query_samples, 1)).start()
    threading.Thread(
            target=process_query_async,
            args=(query_samples, 2)).start()

def main(argv):
    settings = mlperf_loadgen.TestSettings()
    settings.scenario = mlperf_loadgen.TestScenario.MultiStream
    settings.mode = mlperf_loadgen.TestMode.SubmissionRun
    settings.samples_per_query = 4
    settings.target_qps = 1000
    settings.target_latency_ns = 1000000000

    sut = mlperf_loadgen.ConstructSUT(issue_query)
    qsl = mlperf_loadgen.ConstructQSL(
        1024, 128, load_samples_to_ram, unload_samples_from_ram)
    mlperf_loadgen.StartTest(sut, qsl, settings)
    mlperf_loadgen.DestroyQSL(qsl)
    mlperf_loadgen.DestroySUT(sut)


if __name__ == '__main__':
    app.run(main)
