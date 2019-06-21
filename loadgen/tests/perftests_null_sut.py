"""
Python version of perftests_null_sut.cc.
"""

from __future__ import print_function
from absl import app
import mlperf_loadgen
import numpy


def load_samples_to_ram(query_samples):
    return


def unload_samples_from_ram(query_samples):
    return


def issue_query(query_samples):
    responses = []
    for s in query_samples:
        responses.append(mlperf_loadgen.QuerySampleResponse(s.id, 0, 0))
    mlperf_loadgen.QuerySamplesComplete(responses)


def process_latencies(latencies_ns):
    print("Average latency: ")
    print(numpy.mean(latencies_ns))
    print("Median latency: ")
    print(numpy.percentile(latencies_ns, 50))
    print("90 percentile latency: ")
    print(numpy.percentile(latencies_ns, 90))


def main(argv):
    settings = mlperf_loadgen.TestSettings()
    settings.scenario = mlperf_loadgen.TestScenario.SingleStream
    settings.mode = mlperf_loadgen.TestMode.PerformanceOnly

    sut = mlperf_loadgen.ConstructSUT(issue_query, process_latencies)
    qsl = mlperf_loadgen.ConstructQSL(
        1024 * 1024, 1024, load_samples_to_ram, unload_samples_from_ram)
    mlperf_loadgen.StartTest(sut, qsl, settings)
    mlperf_loadgen.DestroyQSL(qsl)
    mlperf_loadgen.DestroySUT(sut)


if __name__ == '__main__':
    app.run(main)
