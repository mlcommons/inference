import subprocess
import mlperf_loadgen as lg
import argparse
import os
import logging
import sys
import requests
import json

sys.path.insert(0, os.getcwd())

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Llama-8B-MAIN")

# function to check the model name in server matches the user specified one


def verify_model_name(user_specified_name, url):
    response = requests.get(url)
    if response.status_code == 200:
        response_dict = response.json()
        server_model_name = response_dict["data"][0]["id"]
        if user_specified_name == server_model_name:
            return {"matched": True, "error": False}
        else:
            return {
                "matched": False,
                "error": f"User specified {user_specified_name} and server model name {server_model_name} mismatch!",
            }
    else:
        return {
            "matched": False,
            "error": f"Failed to get a valid response. Status code: {response.status_code}",
        }


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["Offline", "Server", "SingleStream"],
        default="Offline",
        help="Scenario",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Model name",
    )
    parser.add_argument("--dataset-path", type=str, default=None, help="")
    parser.add_argument(
        "--accuracy",
        action="store_true",
        help="Run accuracy mode")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        help="data type of the model, choose from float16, bfloat16 and float32",
    )
    parser.add_argument(
        "--audit-conf",
        type=str,
        default="audit.conf",
        help="audit config for LoadGen settings during compliance runs",
    )
    parser.add_argument(
        "--user-conf",
        type=str,
        default="user.conf",
        help="user config for user LoadGen settings such as target QPS",
    )
    # TODO: This interpretation of 'total-sample-count' is a little
    # misleading. Fix it
    parser.add_argument(
        "--total-sample-count",
        type=int,
        default=13368,
        help="Number of samples to use in benchmark.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Model batch-size to use in benchmark.",
    )
    parser.add_argument(
        "--output-log-dir", type=str, default="output-logs", help="Where logs are saved"
    )
    parser.add_argument(
        "--enable-log-trace",
        action="store_true",
        help="Enable log tracing. This file can become quite large",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of workers to process queries",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=8,
        help="Number of tensor parallel GPUs",
    )
    parser.add_argument("--vllm", action="store_true", help="vllm mode")
    parser.add_argument(
        "--api-model-name",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Model name(specified in llm server)",
    )
    parser.add_argument(
        "--api-server",
        type=str,
        default=None,
        help="Specify an api endpoint call to use api mode",
    )
    parser.add_argument(
        "--lg-model-name",
        type=str,
        default="llama3_1-8b",
        choices=["llama3_1-8b", "llama3_1-8b-edge"],
        help="Model name(specified in llm server)",
    )

    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="Fraction of GPU memory for vLLM to use (default: 0.90)",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="Max tokens in a single batch (default: vLLM engine default)",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=256,
        help="Max concurrent sequences (default: 256)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="KV cache block size (default: 16)",
    )
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use eager mode instead of CUDA graphs (default: disabled)",
    )
    parser.add_argument(
        "--enable-chunked-prefill",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable chunked prefill (default: vLLM engine default)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Max model context length (default: vLLM engine default)",
    )

    args = parser.parse_args()
    return args


scenario_map = {
    "offline": lg.TestScenario.Offline,
    "server": lg.TestScenario.Server,
    "singlestream": lg.TestScenario.SingleStream,
}


def main():
    args = get_args()

    settings = lg.TestSettings()
    settings.scenario = scenario_map[args.scenario.lower()]
    # mlperf.conf is automatically loaded by the loadgen
    # settings.FromConfig(args.mlperf_conf, "llama3_1-8b", args.scenario)
    settings.FromConfig(args.user_conf, args.lg_model_name, args.scenario)

    if args.accuracy:
        settings.mode = lg.TestMode.AccuracyOnly
    else:
        settings.mode = lg.TestMode.PerformanceOnly

    os.makedirs(args.output_log_dir, exist_ok=True)
    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = args.output_log_dir
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings
    log_settings.enable_trace = args.enable_log_trace

    if args.vllm:
        from SUT_VLLM import SUT, SUTServer
    else:
        raise NotImplementedError

    sut_map = {"offline": SUT, "server": SUTServer, "singlestream": SUTServer}

    sut_cls = sut_map[args.scenario.lower()]

    if args.vllm:
        sut = sut_cls(
            model_path=args.model_path,
            dtype=args.dtype,
            batch_size=args.batch_size,
            dataset_path=args.dataset_path,
            total_sample_count=args.total_sample_count,
            workers=args.num_workers,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=args.max_num_seqs,
            block_size=args.block_size,
            enforce_eager=args.enforce_eager,
            enable_chunked_prefill=args.enable_chunked_prefill,
            max_model_len=args.max_model_len
        )
    else:
        sut = sut_cls(
            model_path=args.model_path,
            dtype=args.dtype,
            batch_size=args.batch_size,
            dataset_path=args.dataset_path,
            total_sample_count=args.total_sample_count,
            workers=args.num_workers,
        )

    # Start sut before loadgen starts
    sut.start()
    lgSUT = lg.ConstructSUT(sut.issue_queries, sut.flush_queries)
    log.info("Starting Benchmark run")
    lg.StartTestWithLogSettings(
        lgSUT,
        sut.qsl,
        settings,
        log_settings,
        args.audit_conf)

    # Stop sut after completion
    sut.stop()

    log.info("Run Completed!")

    log.info("Destroying SUT...")
    lg.DestroySUT(lgSUT)

    log.info("Destroying QSL...")
    lg.DestroyQSL(sut.qsl)


if __name__ == "__main__":
    main()
