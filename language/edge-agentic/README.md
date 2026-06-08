# MLPerf Inference Reference Implementation — Edge Agentic (BFCL v4)

This is the reference implementation for the **Edge Agentic** workload, using
[Berkeley Function Calling Leaderboard v4 (BFCL v4)](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v4.html)
as the accuracy benchmark.

The implementation lives in the
[`mlcommons/endpoints`](https://github.com/mlcommons/endpoints) repository.
The runnable example, configuration files, and a one-script reproducer are at:

> **[`examples/10_Edge_Agentic_Example/`](https://github.com/mlcommons/endpoints/tree/main/examples/10_Edge_Agentic_Example)**

---

## Model

| Property | Value |
| --- | --- |
| Model | `Qwen/Qwen3.6-27B` (Q4\_K\_M GGUF quantization validated) |
| Server | Any OpenAI-compatible endpoint (`/v1/chat/completions`) |
| Validated server | [`llama.cpp llama-server`](https://github.com/ggml-org/llama.cpp) built with CUDA on NVIDIA Jetson Thor |
| HuggingFace ID | [`Qwen/Qwen3.6-27B`](https://huggingface.co/Qwen/Qwen3.6-27B) |

---

## Dataset

| Property | Value |
| --- | --- |
| Source | [`gorilla-llm/gorilla-eval-set`](https://huggingface.co/datasets/gorilla-llm/gorilla-eval-set) (HuggingFace, public) |
| Download | Automatic at runtime — no separate download step required |
| Single-turn subsets | `non_live`, `live`, `hallucination` |
| Multi-turn subsets | `multi_turn_base`, `multi_turn_miss_func`, `multi_turn_miss_param`, `multi_turn_long_context` |

---

## Run Parameters

| Parameter | Value |
| --- | --- |
| `temperature` | `0` (deterministic) |
| `top_p` | TBD |
| `top_k` | TBD |
| `max_tokens` (max\_osl) | TBD |
| `seed` | `42` |
| `tool_choice` | `auto` |
| ST sampling: `non_live` | 20% |
| ST sampling: `live` | 10% (subsets ≤ 25 samples taken in full) |
| ST sampling: `hallucination` | 5% |
| MT sampling | 3% per subset |

---

## Accuracy Targets

| Metric | Reference Score | Threshold (99%) |
| --- | --- | --- |
| Single-turn overall | 87.50% | TBD |
| `non_live` (AST) | 86.98% | TBD |
| `live` | 84.12% | TBD |
| `hallucination` | 94.32% | TBD |
| Multi-turn overall (3% sample) | 45.84% | TBD |
| `multi_turn_base` (full, 200 entries) | 70.00% | TBD |

> Accuracy thresholds (99% of reference score) are TBD pending MLCommons
> working group review.

---

## Reproducing the Results

See the full step-by-step guide and the `run_accuracy.sh` one-script reproducer at:

**[`mlcommons/endpoints — examples/10_Edge_Agentic_Example/`](https://github.com/mlcommons/endpoints/tree/main/examples/10_Edge_Agentic_Example)**

Quick start (≈ 2.5 h on an edge device):

```bash
git clone https://github.com/mlcommons/endpoints.git
cd endpoints
pip install -e ".[bfcl]"
cd examples/10_Edge_Agentic_Example
MODEL=Qwen3.6-27B-Q4_K_M ENDPOINT=http://localhost:8080 bash run_accuracy.sh
```

---

## Scenario

This workload runs in **accuracy-only** mode (offline scenario, single worker,
single connection for deterministic per-sample ordering). There is no
performance (throughput) target for this workload — the benchmark measures
function-calling accuracy, not QPS.
