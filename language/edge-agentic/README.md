# MLPerf Inference Reference Implementation — Edge Agentic (BFCL v4)

This is the reference implementation for the **Edge Agentic** workload, using
[Berkeley Function Calling Leaderboard v4 (BFCL v4)](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v4.html)
as the accuracy benchmark.

The implementation lives in the
[`mlcommons/endpoints`](https://github.com/mlcommons/endpoints) repository.
The runnable example, configuration file, and step-by-step reproducer are at:

> **[`examples/11_Edge_Agentic_Example/`](https://github.com/mlcommons/endpoints/tree/main/examples/11_Edge_Agentic_Example)**

The gated accuracy benchmark is **single-turn only** (3 categories,
per-category sampled to a stable ~995-sample point estimate). Multi-turn remains
available as an optional exploratory run but is **not** part of the accuracy
gate.

---

## Model

| Property | Value |
| --- | --- |
| Model | `Qwen/Qwen3.6-27B` (Q4\_K\_M GGUF quantization validated) |
| Server | Any OpenAI-compatible endpoint (`/v1/chat/completions`) |
| Validated server | [`llama.cpp llama-server`](https://github.com/ggml-org/llama.cpp) (commit `cfff1fc`) built with CUDA on NVIDIA Jetson AGX Thor, `--reasoning off --ctx-size 32768 -np 1` |
| HuggingFace ID | [`Qwen/Qwen3.6-27B`](https://huggingface.co/Qwen/Qwen3.6-27B) |

---

## Dataset

| Property | Value |
| --- | --- |
| Source | [`gorilla-llm/gorilla-eval-set`](https://huggingface.co/datasets/gorilla-llm/gorilla-eval-set) (HuggingFace, public) |
| Download | Automatic at runtime — no separate download step required |
| Gated subsets (single-turn) | `non_live`, `live`, `hallucination` |
| QSL size | ~995 (per-category sampling, see Run Parameters) |
| Multi-turn subsets | `multi_turn_base`, `multi_turn_miss_func`, `multi_turn_miss_param`, `multi_turn_long_context` — **optional, not gated** |

---

## Calibration

Submitters who quantize the reference model themselves calibrate with
[`bfcl_calib.jsonl`](https://github.com/mlcommons/endpoints/blob/main/examples/11_Edge_Agentic_Example/bfcl_calib.jsonl)
— 364 chat/tool-aware BFCL v4 single-turn records, used **only** to estimate
quantizer scales (weight / activation / KV-cache). It is not scored and is not
part of the accuracy gate. See the
[calibration dataset notes](https://github.com/mlcommons/endpoints/tree/main/examples/11_Edge_Agentic_Example#quantization-calibration-dataset)
in the implementation repo for provenance and the overlap disclosure (the set is
sampled from the BFCL v4 domain; ~30% of its prompts also appear in the ~995
accuracy gate).

---

## Run Parameters

| Parameter | Value |
| --- | --- |
| `temperature` | `0` (deterministic, greedy) |
| `top_p` / `top_k` | server default — unused at `temperature 0` |
| `max_tokens` (max\_osl) | `1024` |
| `seed` | `42` |
| `tool_choice` | `auto` |
| ST sampling: `non_live` | 62% (~712 samples) |
| ST sampling: `live` | 10% (subsets ≤ 25 samples taken in full → ~171 samples) |
| ST sampling: `hallucination` | 10% (~112 samples) |
| `subset_floor` | 25 (subsets ≤ 25 taken in full) |

Total ≈ **995** single-turn samples — large enough for a stable point estimate.

---

## Accuracy Targets

The pass/fail criterion is a **3% one-sided band** anchored on the validated
Jetson AGX Thor `Qwen3.6-27B-Q4_K_M` reference: a submission passes if its
single-turn score is **≥ 0.97 × reference**, with no upper bound (a higher score
never fails). Accuracy is hardware-independent (deterministic at `temperature 0`
+ fixed seed), so the same thresholds apply on any device.

| Metric | Reference Score | Pass threshold (0.97 ×) |
| --- | --- | --- |
| Single-turn **overall** (gated) | 86.23% | **≥ 83.64%** |
| Single-turn **non_live-normalized** (gated) | 87.96% | **≥ 85.32%** |
| `non_live` (AST) | 82.59% | not individually gated |
| `live` | 84.12% | not individually gated |
| `hallucination` | 97.16% | not individually gated |

The two gated metrics are encoded in the ruleset at
[`src/inference_endpoint/config/rulesets/mlcommons/models.py`](https://github.com/mlcommons/endpoints/blob/main/src/inference_endpoint/config/rulesets/mlcommons/models.py)
(`Qwen3_6_27B.accuracy_target_settings`).

> The optional multi-turn run is **not gated**. For reference, a single run of
> the full 200-entry `multi_turn_base` (no sampling) scored 140/200 = 70.00%, in
> parity with evalscope.

---

## Reproducing the Results

See the full step-by-step guide at:

**[`mlcommons/endpoints — examples/11_Edge_Agentic_Example/`](https://github.com/mlcommons/endpoints/tree/main/examples/11_Edge_Agentic_Example)**

Quick start — accuracy-only (~3 h on an edge device):

```bash
git clone https://github.com/mlcommons/endpoints.git
cd endpoints
pip install -e ".[bfcl]"
cd examples/11_Edge_Agentic_Example
# Edit model_params.name + endpoint_config.endpoints in the config to match your
# server, then:
inference-endpoint benchmark from-config \
  --config online_edge_full_run.yaml \
  --accuracy-only
```

Drop `--accuracy-only` to run the mandated combined benchmark (performance +
accuracy back-to-back, ~5.5 h) from the same config — this prevents performance
and accuracy from being measured under different settings.

---

## Scenario

The **gated metric is single-turn function-calling accuracy** (offline scenario,
single worker, single connection for deterministic per-sample ordering).

The same `online_edge_full_run.yaml` config also defines a **performance phase**
(single-stream replay of recorded agentic-coding trajectories, scored by an
inline online checker) that runs back-to-back with the accuracy phase. The two
phases share one config so a submission cannot use different settings for
performance and accuracy. Absolute latency/throughput are hardware-specific;
only accuracy is hardware-independent and gated.
