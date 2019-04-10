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


def process_query_async(query_id, response_count):
    time.sleep(.01)
    r1 = mlperf_loadgen.QuerySampleResponse(0, 0)
    r2 = mlperf_loadgen.QuerySampleResponse(0, 0)
    r3 = mlperf_loadgen.QuerySampleResponse(0, 0)
    r4 = mlperf_loadgen.QuerySampleResponse(0, 0)
    responses = [r1, r2, r3, r4]
    mlperf_loadgen.QueryComplete(query_id, responses)


def issue_query(query_id, query_samples):
    print(query_samples)
    threading.Thread(
            target=process_query_async,
            args=(query_id, len(query_samples))).start()


def main(argv):
    sut_name = "DemoSUT"
    sut = mlperf_loadgen.ConstructSUT(sut_name, issue_query)
    qsl_name = "DemoQSL"
    qsl = mlperf_loadgen.ConstructQSL(
        qsl_name, 1024, 128, load_samples_to_ram, unload_samples_from_ram)
    command_line = "--mlperf_scenario edge"
    mlperf_loadgen.StartTest(sut, qsl, command_line)
    mlperf_loadgen.DestroyQSL(qsl)
    mlperf_loadgen.DestroySUT(sut)


if __name__ == '__main__':
    app.run(main)
