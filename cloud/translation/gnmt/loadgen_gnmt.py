from queue import Queue
import threading
import time
import mlperf_loadgen
import numpy

def load_samples_to_ram(query_samples):
    return

def unload_samples_from_ram(query_samples):
    return

def process_latencies(latencies_ns):
    print("Average latency: ")
    print(numpy.mean(latencies_ns))
    print("Median latency: ")
    print(numpy.percentile(latencies_ns, 50))
    print("90 percentile latency: ")
    print(numpy.percentile(latencies_ns, 90))


class TranslationTask:
    def __init__(self, query_id, input_file, output_file):
        self.query_id = query_id
        self.input_file = input_file
        self.output_file = output_file
        self.start = time.time()

class Runner:
    
    def __init__(self):
        self.count = 0
        self.tasks = Queue(maxsize=5)

    ##
    # @brief Invoke GNMT to translate the input file
    def translate(self, qitem):
        print("translate {} (QID {}): {} --> {}".format(self.count, qitem.query_id, qitem.input_file, qitem.output_file))
        self.count += 1
        
        return self.count

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

            result = self.translate(qitem)
            response = []

            # TBD: do something when we are running accuracy mode
            # We need to properly store the result. Perhaps through QuerySampleResponse, otherwise internally
            # in this instance of Runner.
            # QuerySampleResponse contains an ID, a size field and a data pointer field
            response.append(mlperf_loadgen.QuerySampleResponse(qitem.query_id, 0, 0))
            mlperf_loadgen.QuerySamplesComplete(response)
    
    ##
    # @brief Stop worker thread
    def finish(self):
        print("empty queue")
        self.tasks.put(None)
        self.worker.join()

    ##
    # @brief Create a task and add it to the queue
    def enqueue(self, input_file, output_file, ID=-1):
        print("Add to the queue")
        task = TranslationTask(ID, input_file, output_file)
        self.tasks.put(task)

    ##
    # @brief start worker thread
    def start_worker(self):
        self.worker = threading.Thread(target=self.handle_tasks)
        self.worker.daemon = True
        self.worker.start()

if __name__ == "__main__":
    runner = Runner()

    ##
    # @brief function to handle incomming querries
    # @note a query has the following fields:
    # * index: this is the sample_ID, and indexes in e.g., an image or sentence.
    # * ID: this is the query ID
    def issue_query(query_samples):
        for sample in query_samples:
            runner.enqueue("in_{}".format(sample.index), "out_{}".format(sample.index), sample.id)

    runner.start_worker()

    print ("Starting pool")

    settings = mlperf_loadgen.TestSettings()
    settings.scenario = mlperf_loadgen.TestScenario.SingleStream
    settings.mode = mlperf_loadgen.TestMode.PerformanceOnly
    settings.samples_per_query = 1
    settings.target_qps = 10        # Doesn't seem to have an effect
    settings.target_latency_ns = 1000000000

    
    total_queries = 256 # Maximum sample ID + 1
    perf_queries = 8   # TBD: Doesn't seem to have an effect

    sut = mlperf_loadgen.ConstructSUT(issue_query, process_latencies)
    qsl = mlperf_loadgen.ConstructQSL(
        total_queries, perf_queries, load_samples_to_ram, unload_samples_from_ram)
    mlperf_loadgen.StartTest(sut, qsl, settings)
    mlperf_loadgen.DestroyQSL(qsl)
    mlperf_loadgen.DestroySUT(sut)

