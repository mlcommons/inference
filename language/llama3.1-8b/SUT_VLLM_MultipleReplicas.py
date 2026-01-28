import os
from multiprocessing import Process, Queue, Manager, Event
import time
import random
from typing import List, Dict, Any, Tuple
import argparse
import ctypes # For C-compatible types for Loadgen
import math # For ceil in batching
from dataset import Dataset
from vllm import TokensPrompt
import logging
import array # For converting token IDs to bytes
import numpy as np # For proper token ID conversion
import sys
import pkg_resources
from datetime import datetime   

# Attempt to import vLLM. If not found, provide a clear message.
try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("vLLM is not installed.")
    print("Please install it using: pip install vllm")
    print("Note: vLLM requires a compatible GPU (NVIDIA with CUDA).")
    exit(1)

# Attempt to import mlperf.loadgen. If not found, provide a clear message.
try:
    import mlperf_loadgen as lg
except ImportError:
    print("mlperf_loadgen is not installed.")
    print("Please install it from the MLPerf Inference repository.")
    print("Example: pip install -e inference/loadgen")
    exit(1)

# Attempt to import NVTX for profiling
nvtx = None
def setup_nvtx(enable_nvtx):
    """Setup NVTX based on command line option"""
    global nvtx
    if enable_nvtx:
        try:
            import nvtx
            print("NVTX enabled for profiling")
            return nvtx
        except ImportError:
            print("NVTX is not installed. NVTX markers will be disabled.")
            print("Please install it using: pip install nvtx")
            return None
    else:
        print("NVTX disabled via command line option")
        return None

def load_samples_to_ram(query_samples):
    """
    Placeholder for loading samples to RAM for the QSL.
    In the offline scenario, the actual prompts are already in `text_prompts`.
    """
    del query_samples
    return

def unload_samples_from_ram(query_samples):
    """
    Placeholder for unloading samples from RAM for the QSL.
    """
    del query_samples
    return

#Does not help printing the vllm metrics . Its kind of disabled with offline serving
#https://github.com/vllm-project/vllm/issues/15775
def print_vllm_output_metrics(outputs):
    print(outputs)
    #for output in outputs:
        #metrics = output.metrics
        #print(metrics)
        #print(f"Request ID: {output.request_id}")
        #print(f"Prompt: {output.prompt}")
        #print(f"Generated Text: {output.outputs[0].text}")
        #print(f"Time to First Token (TTFT): {metrics.time_to_first_token:.4f} seconds")
        #print(f"Time per Output Token (TPOT): {metrics.time_per_output_token:.4f} seconds/token")
        #print(f"End-to-End Latency: {metrics.e2e_latency:.4f} seconds")
        #print(f"Prefill Time: {metrics.prefill_time:.4f} seconds")
        #print(f"Decode Time: {metrics.decode_time:.4f} seconds")
        #print(f"Total Prompt Tokens: {metrics.num_prompt_tokens}")
        #print(f"Total Generated Tokens: {metrics.num_generation_tokens}")
        #print(f"Finish Reason: {output.outputs[0].finish_reason}")

# --- Worker Process Function ---
def vllm_worker_process(
    process_id: int,
    model_name: str,
    worker_input_queue: Queue, # Queue for incoming MLPerf QuerySample objects
    output_queue: Queue, # Queue for outgoing MLPerf QuerySampleResponse objects
    worker_status: Manager().dict, # Shared dictionary for load tracking (num prompts in queue)
    cuda_device_ids: List[int],
    gpu_memory_utilization: float,
    ready_event: Event,
    max_model_len: int = None,
    max_num_seqs: int = 512,
    test_mode: str = "performance",
    cuda_arch_version: str = "8.9"
) -> None:
    """
    Function to be run by each separate process for vLLM text generation.
    It continuously fetches a BATCH of MLPerf QuerySample objects from its input queue,
    processes them, and sends a BATCH of responses back to the main process.
    """
    # --- IMPORTANT: Set CUDA_VISIBLE_DEVICES for THIS process ---
    os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(map(str, cuda_device_ids))
    os.environ['VLLM_CONFIGURE_LOGGING'] = "0"
    # Set CUDA arch and OMP threads
    os.environ['TORCH_CUDA_ARCH_LIST'] = cuda_arch_version
    os.environ['OMP_NUM_THREADS'] = "16"
    #os.environ['VLLM_LOGGING_LEVEL'] = "DEBUG"
    logging.info(f"Process {process_id}: Configured to use CUDA device: {os.environ['CUDA_VISIBLE_DEVICES']}")
    logging.info(f"Process {process_id}: Set TORCH_CUDA_ARCH_LIST={cuda_arch_version}, OMP_NUM_THREADS=16")

    logging.info(f"Process {process_id}: Starting to load model '{model_name}'...")
    try:
        # Initialize the LLM within each child process.
        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=len(cuda_device_ids),
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs
        )
        print(f"\n--- Replica {process_id} vLLM Configuration ---")
        print("--------------START CONFIG---------------------------\n")
        try:
            # Access the internal LLMEngine instance.
            # In vLLM 0.9.1, it's typically accessed via llm_engine private attribute.
            # The public attributes of LLM object (e.g., llm.engine) might also work for some versions.
            engine_instance = llm.llm_engine 
            print(engine_instance.vllm_config) 
            print(engine_instance.vllm_config.model_config) 
            print(engine_instance.vllm_config.cache_config) 
            
        except Exception as e:
            print(f"  An unexpected error occurred while dumping config: {e}")

        print("--------------END   CONFIG---------------------------\n")
        
        logging.info(f"Process {process_id}: Model loaded successfully on device {cuda_device_ids}.")
        ready_event.set()#Set the event that the model loaded successfully

        worker_status[process_id] = 0 # Initialize this worker's load

        sampling_params = SamplingParams(
            temperature=0.0,
           max_tokens=128,
           min_tokens=1,
           top_p=1,
           top_k=1,
            #stop=["\n\n", "User:", "###", "Human:"],
        )

        batch_counter = 0
        while True:
            # Get a BATCH of QuerySample data from its dedicated input queue
            # This will be a list of dictionaries, where each dict has "query_id" and "prompt_text"
            batch_of_query_data = worker_input_queue.get()

            if batch_of_query_data == "STOP":
                logging.info(f"Process {process_id}: Received STOP signal. Shutting down.")
                break

            batch_counter += 1
            batch_label = f"batch_{batch_counter}_process_{process_id}"
            
            # NVTX marker for batch start
            if nvtx:
                nvtx.mark(f"batch_begin_{batch_label}")

            # Extract prompts and original query_ids for the batch
            prompts_to_process = [TokensPrompt(prompt_token_ids=item["prompt_text"]) for item in batch_of_query_data]
            original_query_ids = [item["query_id"] for item in batch_of_query_data]
            
            # Increment load for the entire batch
            worker_status[process_id] = worker_status[process_id] + len(prompts_to_process)
            logging.info(f"Process {process_id}: Started processing batch of {len(prompts_to_process)} queries . Current load: {worker_status[process_id]}\n")

            start_time = time.time()
            batch_responses = [] # To collect responses for this batch
            try:
                # Perform batched inference using vLLM
                outputs = llm.generate(prompts_to_process, sampling_params)
                end_time = time.time()
                batch_duration = end_time - start_time

                logging.info(f"Process {process_id}: Completed batch of {len(prompts_to_process)} queries in {batch_duration:.2f}s.")

                # Process each output in the batch and prepare for reporting
                for i, output in enumerate(outputs):
                    generated_text = output.outputs[0].text
                    token_ids = output.outputs[0].token_ids
                    token_count = len(token_ids)
                    current_query_id = original_query_ids[i]
                    logging.debug(f"Process {process_id}: Generated text: {generated_text}")

                    # Prepare response data for the collector thread
                    response_data = {
                        "process_id": process_id,
                        "query_id": current_query_id,
                        "generated_text": generated_text, # For debugging/logging in collector
                        "token_count": token_count, # Metric for Loadgen size
                        "duration": batch_duration, # Total batch duration
                        "cuda_device_used": cuda_device_ids,
                        "status": "success"
                    }
                    
                    # Only include token data in accuracy mode for performance optimization
                    if test_mode == "accuracy":
                        # Convert token IDs to bytes for LoadGen accuracy testing
                        # Convert to numpy int32 array then to bytes (standard MLPerf practice)
                        token_array = np.array(token_ids, dtype=np.int32)
                        token_bytes = token_array.tobytes()
                        response_data["token_ids"] = token_ids  # Raw token IDs for reference
                        response_data["token_bytes"] = token_bytes  # Token IDs as bytes for LoadGen
                    
                    batch_responses.append(response_data)
                
                # Send the entire list of responses for this batch to the output queue
                output_queue.put(batch_responses)

            except Exception as e:
                logging.error(f"Process {process_id}: Error processing batch - {e}")
                # For errors, still report completions (as failures) for the entire batch
                error_msg = str(e)
                batch_error_responses = []
                for current_query_id in original_query_ids:
                    batch_error_responses.append({
                        "process_id": process_id,
                        "query_id": current_query_id,
                        "error": error_msg,
                        "cuda_device_attempted": cuda_device_ids,
                        "token_count": 0, # 0 tokens on error
                        "status": "error"
                    })
                output_queue.put(batch_error_responses)
            finally:
                # NVTX marker for batch end
                if nvtx:
                    nvtx.mark(f"batch_end_{batch_label}")
                
                # Decrement load after processing the entire batch
                worker_status[process_id] = worker_status[process_id] - len(prompts_to_process)
                ready_event.set()
                logging.info(f"Process {process_id}: Finished batch. Current load: {worker_status[process_id]}")
            

    except Exception as e:
        logging.critical(f"Process {process_id}: Critical error during setup or main loop - {e}")
        # If setup fails, send a special message to the output queue for the collector to handle
        output_queue.put([{"process_id": process_id, "setup_error": str(e), "cuda_device_attempted": cuda_device_ids, "status": "critical_error"}])

# --- System Under Test (SUT) Class for MLPerf Loadgen ---
class VLLMSchedulingSUT:
    def __init__(self, num_replicas: int, num_gpus: int, model_name: str, dataset_path: str, 
                 scheduling_policy: str, max_model_len: int = None, gpu_memory_utilization: float = 0.9, 
                 max_num_seqs: int = 512, test_mode: str = "performance", cuda_arch_version: str = "8.9"):
        self.num_replicas = num_replicas
        self.num_gpus = num_gpus
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.scheduling_policy = scheduling_policy
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_num_seqs = max_num_seqs
        self.test_mode = test_mode
        self.gpus_per_replica = self.num_gpus // self.num_replicas
        self.cuda_arch_version = cuda_arch_version

        # Multiprocessing components
        self.manager = Manager()
        self.worker_input_queues: List[Queue] = [] # One queue per worker process
        self.results_queue = Queue() # Global queue for completed batches of responses
        self.worker_status = self.manager.dict() # Shared dict {worker_id: current_load_in_queue}
        self.processes: List[Process] = []
        self.worker_ready_events: List[Event] = []

        self.last_assigned_idx = -1 # For Round Robin scheduling
        self.query_id_to_prompt = {} # To store original prompts by query_id for debugging/tracking
        self.data_object = Dataset(self.model_name,dataset_path=self.dataset_path,total_sample_count=13368)

        #Make this a function to print any dataset related information 
        print("="*80)
        print("Dataset Information")
        logging.info("Dataset Max Tokens    = %d", max(self.data_object.input_lens)) 
        logging.info("Dataset Min Tokens    = %d", min(self.data_object.input_lens)) 
        logging.info("Dataset Total Samples = %d", len(self.data_object.input_lens)) 
        print("="*80)

        self._start_workers()
        print("="*80)
        print("LOAD MODELS BEGIN")
        print("="*80)
        self._wait_for_replicas_to_load_models()
        print("="*80)
        print("LOAD MODELS END")
        print("="*80)


        self._start_result_collector() # Start a thread to continuously collect results

    def _start_workers(self):
        """Starts all vLLM worker processes."""
        logging.info(f"SUT: Starting {self.num_replicas} vLLM worker processes...")
        for i in range(self.num_replicas):
            worker_id = i + 1
            q = Queue()
            ready_e = Event()
            self.worker_input_queues.append(q)
            self.worker_ready_events.append(ready_e)
            self.worker_status[worker_id] = 0 # Initialize load for each worker
            start_global_gpu_id = i * self.gpus_per_replica
            assigned_cuda_device_ids = list(range(start_global_gpu_id, start_global_gpu_id + self.gpus_per_replica))
            
    

            process = Process(
                target=vllm_worker_process,
                args=(
                    worker_id,
                    self.model_name,
                    q,
                    self.results_queue,
                    self.worker_status,
                    assigned_cuda_device_ids,
                    self.gpu_memory_utilization,
                    ready_e,
                    self.max_model_len,
                    self.max_num_seqs,
                    self.test_mode,
                    self.cuda_arch_version
                )
            )
            self.processes.append(process)
            process.start()
            logging.info(f"SUT: Worker {worker_id} started (targets GPU {assigned_cuda_device_ids}).")

    def _wait_for_replicas_to_load_models(self, timeout: int = 600):
        """
        Waits for all replica processes to signal that their vLLM models have loaded.
        """
        logging.info(f"SUT: Waiting for all {self.num_replicas} replicas to load models (timeout: {timeout}s)...")
        all_ready = True
        for i, event in enumerate(self.worker_ready_events):
            replica_id = i + 1
            if not event.wait(timeout): # Wait for each replica's event with a timeout
                logging.error(f"SUT Error: Replica {replica_id} failed to signal readiness within {timeout} seconds.")
                all_ready = False
                # If a replica failed to set its event, it likely encountered an error
                # during model loading. For MLPerf, if models aren't ready, results will be invalid anyway.
                raise RuntimeError(f"Replica {replica_id} failed to load model within {timeout}s. Exiting.")
            else:
                logging.info(f"SUT: Replica {replica_id} is ready.")
            
        if not all_ready: # This check is mainly for clarity, as raise would have happened.
            logging.error("SUT: Not all replicas became ready. This indicates a problem with model loading.")
            raise RuntimeError("One or more vLLM replicas failed to load models.")
        logging.info("SUT: All vLLM replicas have signaled readiness.")

    def _start_result_collector(self):
        """Starts a separate thread to collect results from workers and report to Loadgen."""
        def collect_and_report():
            while True:
                try:
                    # Expecting a LIST of results from a worker process
                    batch_of_results = self.results_queue.get(timeout=0.5) 
                    if batch_of_results == "STOP_COLLECTOR":
                        logging.info("SUT Result Collector: Received STOP signal. Exiting.")
                        break

                    # Ensure it's a list (even if only one item)
                    if not isinstance(batch_of_results, list):
                        logging.warning(f"SUT Result Collector: Received unexpected non-list item: {batch_of_results}. Skipping.")
                        continue

                    responses_to_loadgen = []
                    for result_data in batch_of_results:
                        query_id = result_data["query_id"]
                        token_count = result_data["token_count"] # Use token_count for size

                        # If there was a critical setup error from a worker, handle it
                        if result_data.get("status") == "critical_error":
                            error_msg = result_data.get("setup_error")
                            logging.warning(f"SUT Result Collector: Critical setup error from Process {result_data.get('process_id')}, GPU {result_data.get('cuda_device_attempted')}: {error_msg}. This worker might be down.")
                            # For Loadgen, we still need to complete any outstanding queries it might have been assigned
                            # before the critical error. However, in this batched model, the SUT assigns the whole batch
                            # and if the worker crashes before sending the batch, those won't be completed.
                            # A more robust solution would involve SUT tracking all issued queries and
                            # marking them failed if a worker dies without reporting.
                            continue # Skip reporting this specific item as a Loadgen completion

                        # Create a Loadgen QuerySampleResponse for each item in the batch
                        if result_data.get("status") == "success":
                            if self.test_mode == "accuracy":
                                # For accuracy mode, include token bytes for accuracy testing
                                token_bytes = result_data.get("token_bytes", b"")
                                # Convert bytes back to numpy array for proper memory management
                                token_array = np.frombuffer(token_bytes, dtype=np.int32)
                                response_data = token_array.ctypes.data
                                response_size = len(token_bytes)
                                response = lg.QuerySampleResponse(query_id, response_data, response_size, token_count)
                            else:
                                # For performance mode, create response with no data for better performance
                                response = lg.QuerySampleResponse(query_id, 0, 0, token_count)
                        else:
                            # For errors, create response with no data
                            response = lg.QuerySampleResponse(query_id, 0, 0, token_count)
                        
                        responses_to_loadgen.append(response)

                        if result_data.get("status") == "error":
                            error_msg = result_data.get("error")
                            logging.warning(f"SUT Result Collector: Reported processing ERROR for Query {query_id} (Process {result_data.get('process_id')}, GPU {result_data.get('cuda_device_used')}): {error_msg}")
                        else:
                            logging.debug(f"SUT Result Collector: Reported completion for Query {query_id} (Process {result_data.get('process_id')}, GPU {result_data.get('cuda_device_used')}, Tokens: {token_count}).")

                    if responses_to_loadgen:
                        lg.QuerySamplesComplete(responses_to_loadgen) # Inform Loadgen about all completions in this batch
                        logging.info(f"SUT Result Collector: Called QuerySamplesComplete for {len(responses_to_loadgen)} queries in this batch.")

                except Exception as e:
                    # In a real system, you might log this error or handle it more robustly
                    # print(f"SUT Result Collector: Error in collector loop - {e}")
                    pass # Continue looping if no results for a short period

        import threading
        self.collector_thread = threading.Thread(target=collect_and_report, daemon=True) # Daemon to allow main program exit
        self.collector_thread.start()
        logging.info("SUT Result Collector thread started.")


    # --- MLPerf Loadgen Callbacks ---

    def issue_query(self, query_samples: List[lg.QuerySample]):
        """
        Callback from Loadgen: Issue new queries to the SUT.
        In offline scenario, all queries arrive here in one call.
        We divide them into batches and distribute to workers according to the scheduling policy.
        """
        total_samples = len(query_samples)

        if self.num_replicas == 0:
            logging.error("Error: num_replicas is 0, cannot distribute samples.")
            lg.QuerySamplesComplete([lg.QuerySampleResponse(qs.id, 0, 0) for qs in query_samples])
            return

        batch_size = BATCH_SIZE
        num_batches = (total_samples + batch_size - 1) // batch_size
        logging.info(f"\nSUT issue_query: Received {len(query_samples)} queries from Loadgen. Batch size: {batch_size}. Number of batches: {num_batches}.")
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, total_samples)
            batch = query_samples[start:end]
            batch_for_worker = []
            for q_sample in batch:
                prompt_text = self.data_object.input_ids[q_sample.index]
                batch_for_worker.append({
                    "query_id": q_sample.id,
                    "prompt_text": prompt_text
                })
            # Scheduling: choose which replica to send this batch to
            assigned_worker_id = None
            active_worker_ids = [pid for pid in self.worker_status if self.processes[pid-1].is_alive()]
            if not active_worker_ids:
                logging.error(f"SUT issue_query: No active workers available for batch {batch_idx}. Skipping.")
                continue
            if self.scheduling_policy == "first_come_first_served" or self.scheduling_policy == "least_load":
                min_load = float('inf')
                least_loaded_worker_id = None
                for wid in active_worker_ids:
                    current_load = self.worker_status[wid]
                    if current_load < min_load:
                        min_load = current_load
                        least_loaded_worker_id = wid
                assigned_worker_id = least_loaded_worker_id
            elif self.scheduling_policy == "round_robin":
                self.last_assigned_idx = (self.last_assigned_idx + 1) % len(active_worker_ids)
                assigned_worker_id = active_worker_ids[self.last_assigned_idx]
            else:
                assigned_worker_id = active_worker_ids[0] # fallback
            if assigned_worker_id:
                self.worker_input_queues[assigned_worker_id - 1].put(batch_for_worker)
                logging.info(f"SUT issue_query: Sent batch of {len(batch_for_worker)} queries to Process {assigned_worker_id} ({self.scheduling_policy}).")
            else:
                logging.error(f"SUT issue_query: Failed to assign batch {batch_idx}. No suitable worker found.")
        # Periodically log the load on each replica
        if hasattr(self, 'worker_status'):
            status_str = ', '.join([f"Replica {k}: {v}" for k, v in self.worker_status.items()])
            logging.info(f"Replica loads: {status_str}")

    def flush_queries(self):
        """
        Callback from Loadgen: Flush any pending queries.
        (Less critical for offline mode, as all queries are issued at once and processed in batches)
        """
        logging.info("SUT flush_queries: Flushing (no specific action for offline in this demo).")

    def __del__(self):
        """Clean up resources when SUT object is destroyed."""
        self.stop_workers()

    def stop_workers(self):
        """Sends STOP signals to all worker processes and joins them."""
        logging.info("SUT: Sending STOP signals to all worker processes...")
        for q in self.worker_input_queues:
            q.put("STOP")

        # Stop the result collector thread
        if hasattr(self, 'collector_thread') and self.collector_thread.is_alive():
            self.results_queue.put("STOP_COLLECTOR")
            self.collector_thread.join(timeout=5) # Wait for thread to finish
            if self.collector_thread.is_alive():
                logging.warning("SUT: Result collector thread did not terminate gracefully.")

        logging.info("SUT: Waiting for worker processes to terminate...")
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                logging.warning(f"Process {process.pid} (ID {process.name}) did not terminate gracefully. Terminating forcefully.")
                process.terminate()
        self.manager.shutdown()
        logging.info("SUT: All worker processes and manager shut down.")

# --- Main Program ---
if __name__ == "__main__":

    # Print command line and executable information
    import sys
    print("="*80)
    print("COMMAND LINE AND EXECUTABLE INFORMATION")
    print("="*80)
    print(f"Executable: {sys.executable}")
    print(f"Command line: {' '.join(sys.argv)}")
    print("="*80)
    print()

    # Print date and time
    now = datetime.now()
    print(f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    # Print installed packages
    print("Installed Python packages:")
    pkgs = sorted([(d.project_name, d.version) for d in pkg_resources.working_set], key=lambda x: x[0].lower())
    for name, version in pkgs:
        print(f"  {name:<30} {version}")
    print("="*80)
    print()


    parser = argparse.ArgumentParser(
        description="Run vLLM generation with MLPerf Loadgen in offline scenario.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--num_replicas",
        type=int,
        default=1,
        help="Number of parallel processes to create for vLLM generation (each running an LLM instance)."
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Total number of physical GPUs available on the system. Processes will be assigned GPUs in a round-robin fashion."
    )
    parser.add_argument(
        "--scheduling_policy",
        type=str,
        default="round_robin", # Note: Policy is less relevant now as samples are batched once upfront
        choices=["first_come_first_served", "least_load", "round_robin"],
        help="Scheduling policy for distributing prompts to workers in SUT (primarily impacts how batches are assigned)."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=13368, # Default to use all pre-defined prompts
        help="Number of samples (prompts) Loadgen will issue for the offline test."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to the processed dataset pickle file containing tokenized inputs"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="HuggingFaceH4/tiny-random-LlamaForCausalLM",
        help="The name of the LLM model to load. Use a tiny model for testing."
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=None,
        help="Maximum sequence length for the model. Adjust based on model capabilities and available GPU memory"
    )
    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=512,
        help="Maximum number of sequences that can be processed simultaneously by vLLM"
    )
    parser.add_argument(
        "--gpu_mem_util",
        type=float,
        default=0.9,
        help="GPU memory utilization factor (0.0 to 1.0) for vLLM model loading"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for each worker process."
    )
    parser.add_argument(
        "--vllm_api",
        action="store_true",
        help="Enable vLLM API server mode (future use)."
    )
    parser.add_argument(
        "--api_servers",
        type=str,
        default=None,
        help="Comma-separated list of API server URLs (future use)."
    )
    parser.add_argument(
        "--user-conf",
        type=str,
        default="user.conf",
        help="user config for user LoadGen settings such as target QPS",
    )
    parser.add_argument(
        "--lg_model_name",
        type=str,
        default="llama3_1-8b",
        choices=["llama3_1-8b", "llama3_1-8b-interactive","test-model"],
        help="Model name(specified in llm server)",
    )
    parser.add_argument(
        "--output-log-dir", type=str, default="./", help="Where logs are saved"
    )
    parser.add_argument(
        "--enable-nvtx",
        action="store_true",
        help="Enable NVTX markers for profiling batch processing"
    )
    parser.add_argument(
        "--help-args",
        action="store_true",
        help="Show detailed help for all command line arguments"
    )
    parser.add_argument(
        "--test-mode",
        type=str,
        default="performance",
        choices=["performance", "accuracy"],
        help="Test mode: 'performance' for performance testing, 'accuracy' for accuracy testing with raw bytes logging"
    )
    parser.add_argument(
        "--cuda-arch-version",
        type=str,
        default="8.9",
        choices=["8.9", "9.0"],
        help="CUDA arch version for TORCH_CUDA_ARCH_LIST (default: 8.9)"
    )

    args = parser.parse_args()

    # Show detailed help if requested
    if args.help_args:
        print("\n" + "="*80)
        print("DETAILED COMMAND LINE ARGUMENTS")
        print("="*80)
        parser.print_help()
        print("\n" + "="*80)
        print("EXAMPLES:")
        print("="*80)
        print("Basic run with 2 GPUs and 2 replicas:")
        print("  python SUT_VLLM.py --num_gpus 2 --num_replicas 2 --dataset_path /path/to/dataset.pkl")
        print("\nRun with NVTX profiling enabled:")
        print("  python SUT_VLLM.py --enable-nvtx --num_gpus 4 --num_replicas 2")
        print("\nRun with custom batch size and GPU memory utilization:")
        print("  python SUT_VLLM.py --batch_size 2048 --gpu_mem_util 0.8 --max_num_seqs 256")
        print("\nRun with specific model and dataset:")
        print("  python SUT_VLLM.py --model_name <model_path> --dataset_path /path/to/cnn_eval.json")
        print("\nRun in accuracy mode with raw bytes logging:")
        print("  python SUT_VLLM.py --test-mode accuracy --dataset_path /path/to/dataset.pkl")
        print("="*80)
        exit(0)

    # --- Logging Configuration ---
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.ERROR),
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # --- Configuration ---
    MODEL_NAME = args.model_name
    DATASET_PATH = args.dataset_path
    NUM_REPLICAS = args.num_replicas
    NUM_GPUS = args.num_gpus
    SCHEDULING_POLICY = args.scheduling_policy
    NUM_SAMPLES = args.num_samples
    MAX_MODEL_LEN = args.max_model_len
    MAX_NUM_SEQS = args.max_num_seqs
    GPU_MEM_UTIL = args.gpu_mem_util
    BATCH_SIZE = args.batch_size
    VLLM_API = args.vllm_api
    API_SERVERS = args.api_servers.split(',') if args.api_servers else []
    ENABLE_NVTX = args.enable_nvtx
    TEST_MODE = args.test_mode
    CUDA_ARCH_VERSION = args.cuda_arch_version
    
    # Setup NVTX if enabled
    nvtx = setup_nvtx(ENABLE_NVTX)
    
    #Trying with dataset

    if NUM_REPLICAS <= 0:
        logging.error("Error: Number of processes (--num_replicas) must be at least 1.")
        exit(1)
    if NUM_GPUS <= 0:
        logging.error("Error: Number of GPUs (--num_gpus) must be at least 1.")
        exit(1)
    if NUM_SAMPLES <= 0:
        logging.error("Error: Number of samples (--num_samples) must be at least 1.")
        exit(1)


    logging.info("-" * 50)

    # --- Initialize SUT ---
    sut = None
    try:
        sut = VLLMSchedulingSUT(
            num_replicas=NUM_REPLICAS,
            num_gpus=NUM_GPUS,
            model_name=MODEL_NAME,
            dataset_path=DATASET_PATH,
            scheduling_policy=SCHEDULING_POLICY,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEM_UTIL,
            max_num_seqs=MAX_NUM_SEQS,
            test_mode=TEST_MODE,
            cuda_arch_version=CUDA_ARCH_VERSION
        )

        # --- MLPerf Loadgen Setup ---
        settings = lg.TestSettings()
        settings.scenario = lg.TestScenario.Offline
        if TEST_MODE == "accuracy":
            settings.mode = lg.TestMode.AccuracyOnly
        else:
            settings.mode = lg.TestMode.PerformanceOnly
        settings.use_token_latencies = True
        logging.info(f"This may take some time as vLLM models are loaded in each process. Test mode: {TEST_MODE}")
        settings.FromConfig(args.user_conf,args.lg_model_name,"Offline")
        log_output_settings = lg.LogOutputSettings()
        log_output_settings.outdir = args.output_log_dir
        log_output_settings.copy_summary_to_stdout = True
        log_settings = lg.LogSettings()
        log_settings.log_output = log_output_settings
        log_settings.enable_trace = False

        # Construct QSL and SUT for Loadgen.
        # Loadgen will call get_query_samples to get individual data elements,
        # but the SUT's issue_query now expects to receive the lg.QuerySample objects
        # and manages its own internal batching to workers.
        # Note: Loadgen's QSL does not need the actual data passed to its constructor in this way
        # when we manage it via `GetQSLSample(index)` and `issue_query` uses the index.
        # We pass `NUM_SAMPLES` as the total number of items and performance sample count.
        qsl = lg.ConstructQSL(
            13368, # Total samples
            NUM_SAMPLES, # Performance samples
            load_samples_to_ram, # Callback to load data
            unload_samples_from_ram # Callback to unload data
        )

        # SUT for Loadgen: The `issue_query` callback is from our VLLMSchedulingSUT instance
        # The `flush_queries` callback is also from our VLLMSchedulingSUT instance
        SUTToTest = lg.ConstructSUT(sut.issue_query, sut.flush_queries)

        logging.info(f"MLPerf Loadgen: Starting test with {NUM_SAMPLES} samples in Offline mode...")
        logging.info(f"Model: {MODEL_NAME}, Processes: {NUM_REPLICAS}, GPUs: {NUM_GPUS}, Policy: {SCHEDULING_POLICY}, Test Mode: {TEST_MODE}")
        if TEST_MODE == "accuracy":
            logging.info("Token data will be included in responses for accuracy testing")
        else:
            logging.info("Performance mode: Token data excluded for optimal performance")

        lg.StartTestWithLogSettings(SUTToTest, qsl, settings,log_settings)

        logging.info("\nMLPerf Loadgen test finished.")
        logging.info("Main: Program finished.")
        logging.info("Run Completed!")
        #logging.info("Destroying SUT...")
        #lg.DestroySUT(SUTToTest)
        #logging.info("Destroying QSL...")
        #lg.DestroyQSL(qsl)

    except Exception as e:
        logging.critical(f"\nMain program encountered an error: {e}")
    finally:
        # --- Clean up ---
        if sut:
            logging.info("Main: Cleaning up SUT resources (stopping worker processes)...")
            #time.sleep(10)
            sut.stop_workers()
        logging.info("Main: Program finished.")

