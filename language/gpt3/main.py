import subprocess
import mlperf_loadgen as lg
import argparse
import os
import sys

sys.path.append(os.environ["MEGATRON_PATH"])
from megatron.global_vars import set_args

import sys
from backend import get_SUT

sys.path.insert(0, os.getcwd())


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=["tf", "pytorch", "onnxruntime", "tf_estimator"],
        default="pytorch",
        help="Backend",
    )
    parser.add_argument(
        "--scenario",
        choices=["SingleStream", "Offline", "Server", "MultiStream"],
        default="Offline",
        help="Scenario",
    )
    parser.add_argument("--model-path", default="EleutherAI/gpt-j-6B", help="")
    parser.add_argument("--dataset-path", default="./data/cnn_eval.json", help="")
    parser.add_argument("--accuracy", action="store_true", help="enable accuracy pass")
    parser.add_argument(
        "--dtype",
        default="float32",
        help="data type of the model, choose from float16, bfloat16 and float32",
    )
    parser.add_argument(
        "--quantized",
        action="store_true",
        help="use quantized model (only valid for onnxruntime backend)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="enable profiling (only valid for onnxruntime backend)",
    )
    parser.add_argument(
        "--gpu", action="store_true", help="use GPU instead of CPU for the inference"
    )
    parser.add_argument(
        "--mlperf_conf", default="mlperf.conf", help="mlperf rules config"
    )
    parser.add_argument(
        "--user_conf",
        default="user.conf",
        help="user config for user LoadGen settings such as target QPS",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=13368,
        help="Maximum number of examples to consider (not limited by default)",
    )
    parser.add_argument("--rank", default=0)
    global_names = set(vars(parser.parse_known_args()[0]).keys())
    # Add megatron arguments
    megatron_parser = parser.add_argument_group("megatron")
    megatron_parser.add_argument(
        "--tensor-model-parallel-size", default=1
    )  # TODO change to 8
    megatron_parser.add_argument(
        "--pipeline-model-parallel-size", default=1
    )  # TODO change to 8
    megatron_parser.add_argument("--sequence-parallel", action="store_true")
    # megatron_parser.add_argument("--recompute-activations", action="store_true")
    megatron_parser.add_argument("--num-layers", default=1)  # TODO change to 96
    megatron_parser.add_argument("--hidden-size", default=128)  # TODO change to 12288
    megatron_parser.add_argument(
        "--num-attention-heads", default=2
    )  # TODO change to 96
    megatron_parser.add_argument("--seq-length", default=2048)
    megatron_parser.add_argument("--max-position-embeddings", default=2048)
    megatron_parser.add_argument("--micro-batch-size", default=1)
    megatron_parser.add_argument("--global-batch-size", default=1)
    megatron_parser.add_argument("--train-samples", default=20000000)
    megatron_parser.add_argument("--lr-warmup-samples", default=407040)
    megatron_parser.add_argument("--lr-decay-samples", default=166809600)
    megatron_parser.add_argument("--lr", default=1.0)
    megatron_parser.add_argument("--min-lr", default=1.0)
    megatron_parser.add_argument("--lr-decay-style", default="cosine")
    megatron_parser.add_argument("--log-interval", default=1)
    megatron_parser.add_argument("--eval-iters", default=-1)
    megatron_parser.add_argument("--eval-interval", default=1)
    megatron_parser.add_argument("--attention-dropout", default=0.0)
    megatron_parser.add_argument("--hidden-dropout", default=0.0)
    megatron_parser.add_argument("--train-data-path", default="")
    megatron_parser.add_argument("--valid-data-path", default="")
    megatron_parser.add_argument(
        "--vocab-file",
        default="./data/vocab.json",
    )
    megatron_parser.add_argument(
        "--merge-file",
        default="./data/merges.txt",
    )
    megatron_parser.add_argument("--save-interval", default=500)
    megatron_parser.add_argument("--save", default=None)
    megatron_parser.add_argument(
        "--do-layernorm-bias-weight-decay", action="store_true"
    )
    megatron_parser.add_argument("--no-scaled-init", action="store_true")
    megatron_parser.add_argument("--loss-scale", default=-1)
    megatron_parser.add_argument("--split", default="100,0,0")
    megatron_parser.add_argument("--clip-grad", default=-1)
    megatron_parser.add_argument("--weight-decay", default=0.1)
    megatron_parser.add_argument("--adam-beta1", default=0.9)
    megatron_parser.add_argument("--adam-beta2", default=0.95)
    megatron_parser.add_argument("--init-method-std", default=0.006)
    megatron_parser.add_argument("--log-params-norm", action="store_true")
    megatron_parser.add_argument("--log-num-zeros-in-grad", action="store_true")
    megatron_parser.add_argument(
        "--log-validation-ppl-to-tensorboard", action="store_true"
    )
    megatron_parser.add_argument("--DDP-impl", default="local")
    megatron_parser.add_argument("--tensorboard-dir", default="")
    # megatron_parser.add_argument("--no-query-key-layer-scaling", action="store_true")
    # megatron_parser.add_argument("--no-seq-len-plus-one-tokens", action="store_true")
    megatron_parser.add_argument("--seed", default=0)
    megatron_parser.add_argument("--use-checkpoint-args", action="store_true")
    megatron_parser.add_argument("--load", default=None)
    # Tokenizer args
    megatron_parser.add_argument("--tokenizer-type", default="GPT2BPETokenizer")
    megatron_parser.add_argument("--make-vocab-size-divisible-by", default=128)
    megatron_names = set(vars(parser.parse_known_args()[0]).keys()) - global_names
    args = parser.parse_args()
    megatron_args = dict((k, v) for k, v in vars(args).items() if k in megatron_names)
    return args, megatron_args


scenario_map = {
    "SingleStream": lg.TestScenario.SingleStream,
    "Offline": lg.TestScenario.Offline,
    "Server": lg.TestScenario.Server,
    "MultiStream": lg.TestScenario.MultiStream,
}

gen_args = {
    "max_new_tokens": 128,
    "min_new_tokens": 30,
}


def main():
    args, megatron_args = get_args()
    os.environ["RANK"] = "0"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    gen_kwargs = {
        "early_stopping": True,
        "max_new_tokens": 128,
        "min_new_tokens": 30,
        "top_k": 4,
    }

    sut = get_SUT(
        model_path=args.model_path,
        scenario=args.scenario,
        dtype=args.dtype,
        dataset_path=args.dataset_path,
        max_examples=args.max_examples,
        args=args,
        megatron_args=megatron_args,
        use_gpu=args.gpu,
        gen_kwargs=gen_kwargs,
    )

    settings = lg.TestSettings()
    settings.scenario = scenario_map[args.scenario]
    # Need to update the conf
    settings.FromConfig(args.mlperf_conf, "gptj", args.scenario)
    settings.FromConfig(args.user_conf, "gptj", args.scenario)

    if args.accuracy:
        settings.mode = lg.TestMode.AccuracyOnly
    else:
        settings.mode = lg.TestMode.PerformanceOnly
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

    lg.StartTestWithLogSettings(sut.sut, sut.qsl, settings, log_settings)
    print("Test Done!")

    print("Destroying SUT...")
    lg.DestroySUT(sut.sut)

    print("Destroying QSL...")
    lg.DestroyQSL(sut.qsl)


if __name__ == "__main__":
    main()
