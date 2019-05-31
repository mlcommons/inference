# Copyright 2018 The MLPerf Authors. All Rights Reserved.
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

from queue import Queue
import threading
import time
import mlperf_loadgen
import numpy

class ImplementationException (Exception):
    def __init__(self, msg):
        self.msg = msg

    def __repr__(self):
        return "ImplementationException: {}".format(self.msg)

##
# @brief Simple way to process and display latencies
# @param latencies_ns is an array of durations (in ns) it took per sample to finish
# @note that the duration is measured from query submission time to query finish time,
# hence the samples themselves could actually have been finished earlier
def process_latencies(latencies_ns):
    print("Average latency (ms) per query:")
    print(numpy.mean(latencies_ns)/1000000.0)
    print("Median latency (ms): ")
    print(numpy.percentile(latencies_ns, 50)/1000000.0)
    print("90 percentile latency (ms): ")
    print(numpy.percentile(latencies_ns, 90)/1000000.0)

class Task:
    def __init__(self, query_id, sample_id):
        self.query_id = query_id
        self.sample_id = sample_id

class Runner:
    
    def __init__(self, qSize=5):
        self.tasks = Queue(maxsize=qSize)

    def load_samples_to_ram(self, query_samples):
        return

    def unload_samples_from_ram(self, query_samples):
        return

    ##
    # @brief Invoke process a task
    def process(self, qitem):
        raise ImplementationException("Please implement Process function")

    ##
    # @brief infinite loop that pulls translation tasks from a queue
    # @note This needs to be run by a worker thread
    def handle_tasks(self):
        while True:
            # Block until an item becomes available
            qitem = self.tasks.get(block=True)

            # When a "None" item was added, it is a 
            # signal from the parent to indicate we should stop
            # working (see finish)
            if qitem is None:
                break

            result = self.process(qitem)
            response = []

            # TBD: do something when we are running accuracy mode
            # We need to properly store the result. Perhaps through QuerySampleResponse, otherwise internally
            # in this instance of Runner.
            # QuerySampleResponse contains an ID, a size field and a data pointer field
            for query_id in qitem.query_id:
                response.append(mlperf_loadgen.QuerySampleResponse(query_id, 0, 0))

            # Tell loadgen that we're ready with this query
            mlperf_loadgen.QuerySamplesComplete(response)

            self.tasks.task_done()
    
    ##
    # @brief Stop worker thread
    def finish(self):
        print("empty queue")
        self.tasks.put(None)
        self.worker.join()

    ##
    # @brief function to handle incomming querries, by placing them on the task queue
    # @note a query has the following fields:
    # * index: this is the sample_ID, and indexes in e.g., an image or sentence.
    # * id: this is the query ID
    def enqueue(self, query_samples):
        raise ImplementationException("Please implement Enqueue function")

    ##
    # @brief start worker thread
    def start_worker(self):
        self.worker = threading.Thread(target=self.handle_tasks)
        self.worker.daemon = True
        self.worker.start()

class DummyRunner (Runner):
    def __init__(self):
        Runner.__init__(self)
        self.count = 0

    def enqueue(self, query_samples):
        for sample in query_samples:
            print("Adding Dummy task to the queue.")
            task = Task([sample.id], [sample.index])
            self.tasks.put(task)

    def process(self, qitem):
        print("Default dummy process, processing the {}'th query for sample ID {}.".format(self.count, qitem.sample_id[0]))
        self.count += 1
        
        return self.count

if __name__ == "__main__":
    runner = DummyRunner()

    runner.start_worker()

    settings = mlperf_loadgen.TestSettings()
    settings.scenario = mlperf_loadgen.TestScenario.SingleStream
    settings.mode = mlperf_loadgen.TestMode.PerformanceOnly

    # Specify exactly how many queries need to be made
    settings.enable_spec_overrides = True
    settings.override_min_query_count = 3003
    settings.override_max_query_count = 3003
    
    total_queries = 256 # Maximum sample ID + 1
    perf_queries = 8   # TBD: Doesn't seem to have an effect

    sut = mlperf_loadgen.ConstructSUT(runner.enqueue, process_latencies)
    qsl = mlperf_loadgen.ConstructQSL(
        total_queries, perf_queries, runner.load_samples_to_ram, runner.unload_samples_from_ram)
    mlperf_loadgen.StartTest(sut, qsl, settings)
    mlperf_loadgen.DestroyQSL(qsl)
    mlperf_loadgen.DestroySUT(sut)

