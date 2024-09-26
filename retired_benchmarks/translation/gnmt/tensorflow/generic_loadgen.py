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

import queue
from queue import Queue
import threading
import time
import mlperf_loadgen
import array

class ImplementationException (Exception):
    def __init__(self, msg):
        self.msg = msg

    def __repr__(self):
        return "ImplementationException: {}".format(self.msg)

def flush_queries(): pass


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

            results = self.process(qitem)

            # Call post_process on all samples
            self.post_process(qitem.query_id, results)

            self.tasks.task_done()
    
    ##
    # @brief Post process results
    # @note This should serialize the results for query_ids and hand it over to loadgen
    # @note Here it is a dummy implementation that doesn't return anything useful
    def post_process(self, query_ids, results):
        response = []
        for res, q_id in zip(results, query_ids):
            response.append(mlperf_loadgen.QuerySampleResponse(q_id, 0, 0))

        # Tell loadgen that we're ready with this query
        mlperf_loadgen.QuerySamplesComplete(response)

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
    settings.min_query_count = 3003
    settings.max_query_count = 3003
    
    total_queries = 256 # Maximum sample ID + 1
    perf_queries = 8   # TBD: Doesn't seem to have an effect

    sut = mlperf_loadgen.ConstructSUT(runner.enqueue, flush_queries)
    qsl = mlperf_loadgen.ConstructQSL(
        total_queries, perf_queries, runner.load_samples_to_ram, runner.unload_samples_from_ram)
    mlperf_loadgen.StartTest(sut, qsl, settings)
    mlperf_loadgen.DestroyQSL(qsl)
    mlperf_loadgen.DestroySUT(sut)

