#!/usr/bin/env python3
"""Calculate prefix cache hit rate from multi-hop RAG LLM logs."""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from tokenizers import Tokenizer

BLOCK_SIZE = 16
HF_CACHE = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


def find_token_prefix_len(tokens1, tokens2):
    for i in range(min(len(tokens1), len(tokens2))):
        if tokens1[i] != tokens2[i]:
            return i
    return min(len(tokens1), len(tokens2))


def simulate_prefix_cache(calls, block_size=BLOCK_SIZE):
    previous_tokens = []
    results_by_hop = defaultdict(lambda: {"total_isl": 0, "cached": 0, "count": 0})
    total_isl = 0
    total_cached = 0

    for call in calls:
        tokens = call["_tokens"]
        isl = call["metrics"]["isl"]
        template_overhead = isl - len(tokens)

        best_prefix = 0
        for prev in previous_tokens:
            best_prefix = max(best_prefix, find_token_prefix_len(tokens, prev))

        if not previous_tokens:
            cached = 0
        else:
            cached = min((best_prefix // block_size) * block_size + template_overhead, isl)

        hop = call["hop_count"]
        results_by_hop[hop]["total_isl"] += isl
        results_by_hop[hop]["cached"] += cached
        results_by_hop[hop]["count"] += 1
        total_isl += isl
        total_cached += cached

        previous_tokens.append(tokens)

    return total_isl, total_cached, results_by_hop


MODEL_PATTERNS = {
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "gpt-oss-120b": "openai/gpt-oss-120b",
}


def normalize_model_name(model_name):
    for pattern, hf_name in MODEL_PATTERNS.items():
        if pattern in model_name:
            return hf_name
    return model_name


def resolve_tokenizer(model_name, explicit_path=None):
    if explicit_path:
        return Tokenizer.from_file(explicit_path)
    hf_name = normalize_model_name(model_name)
    cache_dir = HF_CACHE / f"models--{hf_name.replace('/', '--')}"
    if not cache_dir.exists():
        sys.exit(f"Error: tokenizer not found for '{model_name}' (resolved: '{hf_name}') at {cache_dir}\n"
                 f"  Provide explicit path or run: huggingface-cli download {hf_name} tokenizer.json")
    snapshots = cache_dir / "snapshots"
    snapshot = next(snapshots.iterdir())
    tok_file = snapshot / "tokenizer.json"
    if not tok_file.exists():
        sys.exit(f"Error: no tokenizer.json in {snapshot}")
    return Tokenizer.from_file(str(tok_file))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Calculate prefix cache hit rate from multi-hop RAG LLM logs.")
    parser.add_argument("llm_log", help="Path to LLM log JSON file")
    parser.add_argument("--tokenizer-grader", help="Explicit tokenizer path for grader model")
    parser.add_argument("--tokenizer-llm", help="Explicit tokenizer path for LLM model (query/sufficiency/answer)")
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE,
                        help=f"KV cache block size in tokens (default: {BLOCK_SIZE})")
    args = parser.parse_args()

    log_path = args.llm_log
    explicit_20b = args.tokenizer_grader
    explicit_120b = args.tokenizer_llm
    block_size = args.block_size

    with open(log_path) as f:
        data = json.load(f)

    meta = data["experiment_metadata"]
    print(f"Experiment: {meta['experiment_name']}")
    print(f"Models: grader={meta['grader_model']}, "
          f"query={meta['query_model']}, "
          f"sufficiency={meta['sufficiency_checker_model']}")
    print(f"Block size: {block_size}")
    print()

    model_to_tokenizer = {}
    component_models = {
        "evaluate_document_relevance": meta["grader_model"],
        "generate_search_queries": meta["query_model"],
        "check_sufficiency": meta["sufficiency_checker_model"],
        "answer_generator": meta["answer_generator_model"],
    }
    unique_models = list(dict.fromkeys(normalize_model_name(m) for m in component_models.values()))

    for i, model in enumerate(unique_models):
        explicit = [explicit_20b, explicit_120b][min(i, 1)] if (explicit_20b or explicit_120b) else None
        model_to_tokenizer[model] = resolve_tokenizer(model, explicit)

    component_tokenizer = {
        comp: model_to_tokenizer[normalize_model_name(model)]
        for comp, model in component_models.items()
    }

    all_calls = []
    for query in data["queries"]:
        for call in query["llm_calls"]:
            call["query_id"] = query["query_id"]
            all_calls.append(call)

    all_calls.sort(key=lambda x: x["timestamp"])

    for call in all_calls:
        prompt = "\n".join(msg["content"] for msg in call["input"]["messages"])
        tok = component_tokenizer[call["component"]]
        call["_tokens"] = tok.encode(prompt).ids

    by_component = defaultdict(list)
    for call in all_calls:
        by_component[call["component"]].append(call)

    model_map = {
        "evaluate_document_relevance": data["experiment_metadata"]["grader_model"],
        "generate_search_queries": data["experiment_metadata"]["query_model"],
        "check_sufficiency": data["experiment_metadata"]["sufficiency_checker_model"],
        "answer_generator": data["experiment_metadata"]["answer_generator_model"],
    }

    components = ["check_sufficiency", "generate_search_queries",
                  "evaluate_document_relevance", "answer_generator"]

    # --- ISL / OSL table ---
    print("=" * 90)
    print("Table 1: Input/Output Sequence Length Statistics")
    print("-" * 90)
    print(f"{'Component':<30} {'N':>4}   {'ISL (min/avg/max/eff)':^28}   {'OSL (min/avg/max)':^21}")
    print("-" * 90)

    comp_cache_results = {}
    grand_isl = 0
    grand_cached = 0
    endpoint_stats = defaultdict(lambda: {"isl": 0, "cached": 0})

    for comp in components:
        calls = sorted(by_component[comp], key=lambda x: x["timestamp"])
        total_isl, total_cached, by_hop = simulate_prefix_cache(calls, block_size)
        comp_cache_results[comp] = (total_isl, total_cached, by_hop)
        grand_isl += total_isl
        grand_cached += total_cached

        model = normalize_model_name(model_map[comp])
        endpoint_stats[model]["isl"] += total_isl
        endpoint_stats[model]["cached"] += total_cached

        isls = [c["metrics"]["isl"] for c in calls]
        osls = [c["metrics"]["osl"] for c in calls]
        eff_isl = (total_isl - total_cached) // len(calls)

        print(f"{comp:<30} {len(calls):>4}   "
              f"{min(isls):>5} / {sum(isls)//len(isls):>5} / {max(isls):>5} / {eff_isl:>5}   "
              f"{min(osls):>5} / {sum(osls)//len(osls):>5} / {max(osls):>5}")

    print()

    # --- Prefix cache table ---
    print("=" * 75)
    print("Table 2: Prefix Cache Hit Rate (weighted by ISL)")
    print("-" * 75)
    print(f"{'Component':<35} {'Hop 2':>7} {'Hop 3':>7} {'Hop 4':>7} {'Hop 5':>7} {'Avg':>7}")
    print("-" * 75)

    for comp in components:
        total_isl, total_cached, by_hop = comp_cache_results[comp]

        hop_rates = {}
        for hop in sorted(by_hop.keys()):
            h = by_hop[hop]
            hop_rates[hop] = 100 * h["cached"] / h["total_isl"] if h["total_isl"] > 0 else 0

        avg_rate = 100 * total_cached / total_isl if total_isl > 0 else 0
        row = f"{comp:<35}"
        for hop in [2, 3, 4, 5]:
            row += f" {hop_rates.get(hop, 0):>6.2f}%"
        row += f" {avg_rate:>6.2f}%"
        print(row)

    print("-" * 75)
    print(f"{'Overall':<35} {'':>7} {'':>7} {'':>7} {'':>7} {100*grand_cached/grand_isl:>6.2f}%")

    print()
    print("=" * 75)
    print("Table 3: Prefix Cache Hit Rate by Model Endpoint (weighted by ISL)")
    print("-" * 75)
    for model, stats in endpoint_stats.items():
        rate = 100 * stats["cached"] / stats["isl"] if stats["isl"] > 0 else 0
        print(f"  {model:<30} {rate:>6.2f}%  (ISL={stats['isl']:,}, Cached={stats['cached']:,})")
    print(f"  {'Overall':<30} {100*grand_cached/grand_isl:>6.2f}%  (ISL={grand_isl:,}, Cached={grand_cached:,})")


if __name__ == "__main__":
    main()
