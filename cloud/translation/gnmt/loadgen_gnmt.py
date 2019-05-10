from generic_loadgen import *

class TranslationTask:
    def __init__(self, query_id, input_file, output_file):
        self.query_id = query_id
        self.input_file = input_file
        self.output_file = output_file
        self.start = time.time()

class GNMTRunner (Runner):
    
    def __init__(self):
        Runner.__init__(self)
        self.count = 0

    ##
    # @brief Invoke GNMT to translate the input file
    def process(self, qitem):
        print("translate {} (QID {}): {} --> {}".format(self.count, qitem.query_id, qitem.input_file, qitem.output_file))
        self.count += 1
        
        return self.count

    ##
    # @brief Create a task and add it to the queue
    def enqueue(self, query_samples):
        for sample in query_samples:
            input_file = "in_{}".format(sample.index)
            output_file = "out_{}".format(sample.index)

            task = TranslationTask(sample.id, input_file, output_file)
            self.tasks.put(task)

if __name__ == "__main__":
    runner = GNMTRunner()

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

    sut = mlperf_loadgen.ConstructSUT(runner.enqueue, process_latencies)
    qsl = mlperf_loadgen.ConstructQSL(
        total_queries, perf_queries, load_samples_to_ram, unload_samples_from_ram)
    mlperf_loadgen.StartTest(sut, qsl, settings)
    mlperf_loadgen.DestroyQSL(qsl)
    mlperf_loadgen.DestroySUT(sut)

