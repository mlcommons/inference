import subprocess
import mlperf_loadgen as lg
import argparse
import os
import math
import sys
from backend_PyTorch import get_SUT
from GPTJ_QDL import GPTJ_QDL
from GPTJ_QSL import get_GPTJ_QSL

# Function to parse the arguments passed during python file execution
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices=["pytorch"], default="pytorch", help="Backend")
    parser.add_argument("--scenario", choices=["SingleStream", "Offline",
                        "Server"], default="Offline", help="Scenario")
    parser.add_argument("--model-path", default="EleutherAI/gpt-j-6B", help="")
    parser.add_argument(
        "--dataset-path", default="./data/cnn_eval.json", help="")
    parser.add_argument("--accuracy", action="store_true",
                        help="enable accuracy pass")
    parser.add_argument("--dtype", default="float32", help="data type of the model, choose from float16, bfloat16 and float32")
    parser.add_argument("--quantized", action="store_true",
                        help="use quantized model (only valid for onnxruntime backend)")
    parser.add_argument("--profile", action="store_true",
                        help="enable profiling (only valid for onnxruntime backend)")
    parser.add_argument("--gpu", action="store_true",
                        help="use GPU instead of CPU for the inference")
    parser.add_argument("--audit_conf", default="audit.conf",
                        help="audit config for LoadGen settings during compliance runs")
    parser.add_argument(
        "--mlperf_conf", default="mlperf.conf", help="mlperf rules config")
    parser.add_argument("--user_conf", default="user.conf",
                        help="user config for user LoadGen settings such as target QPS")
    parser.add_argument("--max_examples", type=int, default=13368,
                        help="Maximum number of examples to consider (not limited by default)")
    parser.add_argument("--network", choices=["sut","lon",None], default=None, help="Loadgen network mode")
    parser.add_argument('--node', type=str, default="")
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--sut_server', nargs="*", default= ['http://localhost:8000'],
                    help='Address of the server(s) under test.')
    args = parser.parse_args()
    return args

# Function to get the amount of temporary cache generated when running the GPT-J model
# Varies with the beam size set(Estimate: 6GB x Beam size)
def get_temp_cache():
    beam_size = int(os.environ.get("GPTJ_BEAM_SIZE", "4"))
    return 6 * beam_size

# Map the loadgen scenario as per the option given by the user
scenario_map = {
    "SingleStream": lg.TestScenario.SingleStream,
    "Offline": lg.TestScenario.Offline,
    "Server": lg.TestScenario.Server,
    "MultiStream": lg.TestScenario.MultiStream
}

# Main function triggered when the script is run
def main():
    args = get_args()
    qsl = None
    if args.network != "sut":
        # Gets the Query Data Loader and Query Sample Loader
        # Responsible for loading(Query Sample Loader) and sending samples over the network(Query Data Loader) to the server
        qsl = get_GPTJ_QSL(
            dataset_path=args.dataset_path,
            max_examples=args.max_examples
        )
        if args.network == "lon":
            qdl = GPTJ_QDL(
                sut_server_addr=args.sut_server,
                scenario=args.scenario,
                qsl = qsl
            )

        # Initiates and loads loadgen test settings and log path
        settings = lg.TestSettings()
        settings.scenario = scenario_map[args.scenario]
        # Need to update the conf
        settings.FromConfig(args.mlperf_conf, "gptj", args.scenario)
        settings.FromConfig(args.user_conf, "gptj", args.scenario)

        # Chosing test mode Accutacy/Performance
        if args.accuracy:
            settings.mode = lg.TestMode.AccuracyOnly
        else:
            settings.mode = lg.TestMode.PerformanceOnly

        # Set log path
        log_path = os.environ.get("LOG_PATH")
        if not log_path:
            log_path = "build/logs"
        if not os.path.exists(log_path):
            os.makedirs(log_path)

        log_output_settings = lg.LogOutputSettings()
        log_output_settings.outdir = log_path
        log_output_settings.copy_summary_to_stdout = True
        log_settings = lg.LogSettings()
        log_settings.log_output = log_output_settings
        log_settings.enable_trace = True

    if args.network != "lon":
        # Gets SUT.
        # SUT only initialised when network is either None or is SUT as it is not needed in case of client(LON).
        # Only the server loads the model.
        sut = get_SUT(
            model_path=args.model_path,
            scenario=args.scenario,
            dtype=args.dtype,
            use_gpu=args.gpu,
            network=args.network,
            dataset_path=args.dataset_path,
            max_examples=args.max_examples,
            qsl=qsl # If args.network is None, then only QSL get passed to the SUT, else it will be None
        )
    
    if args.network == "lon" and args.scenario == "SingleStream":
        print("ERROR: Single stream scenario in Loadgen Over the Network is not supported!")

    # If network option is LON, QDL is loaded and request is served to SUT based on the scenario given by user(Offline or SingleStream)
    elif args.network == "lon":
        lg.StartTestWithLogSettings(qdl.qdl, qsl.qsl, settings, log_settings, args.audit_conf)

    # If network option is SUT, a flask server is initiated, request is processed and output is being sent back to LON client
    elif args.network == "sut":
        temp_cache = get_temp_cache()
        from network_SUT import app, node, set_backend, set_semaphore
        from systemStatus import get_cpu_memory_info, get_gpu_memory_info

        # Calculating free memory inorder to determine the number of instances that can be run at a particular time
        # lockVar contains the value of number of instances
        # Formula:
        #   Number of Instances = (free_memory_of_system - memory_size_taken_by_model)/temp_cache_size
        # Based on the got value(lockVar) a semaphore is initialised
        # Acquire and release of semaphore can be found in network_SUT.py
        model_mem_size = sut.total_mem_size
        if args.gpu:
            free_mem = int(os.environ.get("CM_CUDA_DEVICE_PROP_GLOBAL_MEMORY", get_gpu_memory_info())) / (1024**3)
        else:
            free_mem = get_cpu_memory_info()
        lockVar = math.floor((free_mem - model_mem_size)/temp_cache)
        node = args.node
        # Semaphore is set inorder to create multiple instances upon request incomming
        set_semaphore(lockVar)
        print(f"Set the semaphore lock variable to {lockVar}")
        # Pass SUT as the backend
        set_backend(sut)
        app.run(debug=False, port=args.port, host="0.0.0.0")

    else:
        # Test not run in Loadgen Over the Network
        print("Running LoadGen test...")
        lg.StartTestWithLogSettings(sut.sut, qsl.qsl, settings, log_settings, args.audit_conf)

    print("Test Done!")

    if args.network != "lon":
        print("Destroying SUT...")
        lg.DestroySUT(sut.sut)

    if args.network != "sut":
        print("Destroying QSL...")
        lg.DestroyQSL(qsl.qsl)


if __name__ == "__main__":
    main()