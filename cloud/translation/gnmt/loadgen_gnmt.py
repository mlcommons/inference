from queue import Queue
import threading
import time


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
    def translate(self, input_file, output_file):
        print("translate {}".format(self.count))
        self.count += 1
        pass

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

            self.translate(qitem.input_file, qitem.output_file)

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

    print ("Starting pool")

    # Add two items to the task list
    runner.enqueue("a", "B")
    runner.enqueue("b", "c")

    # Start the worker thread
    runner.start_worker()

    # Add another item to indicate blocking calls to task list are working
    runner.enqueue("c", "d")

    print ("starting sleep")
    time.sleep(1)

    # Gracefully stop the worker thread
    runner.finish()
    runner.enqueue("d", "e")

    print("running toward end of program")

