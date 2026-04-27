# MLPerf Inference Anomaly Detector

Experimental tool to detect when a benchmark result deviates significantly from similar historical results.

## Requirements

- Python 3.12
- `pandas==3.0.0`
- `numpy==2.4.2`
- `scipy==1.17.1`
- `dash==4.1.0`
- `plotly==6.7.0`

## Usage

```bash
python3 main.py
```

Opens a Dash web UI at `http://127.0.0.1:8050`.

## How it works

1. **Historical data** is loaded from a CSV of past MLPerf results and grouped by `(Model MLC, Scenario, Accelerator)`.
2. Each group's `Result / Total Accelerators` is modelled as a normal distribution.
3. A new result is submitted via the UI (loadgen log + system JSON + model/scenario selection).
4. The result is tested against its group's distribution using a z-score. Results with `|z| > 2.5` are flagged as anomalies.

## Modules

| File | Description |
|---|---|
| `main.py` | Dash web application — entry point |
| `loader.py` | Loads historical CSV and standardizes accelerator names |
| `statistical_tests.py` | Normal-distribution z-score anomaly test |
| `system_loader.py` | Parses system description JSON files |
| `loadgen_parser.py` | Parses MLPerf loadgen detail log files |
| `config.py` | Constants: accelerator name mappings, result key mappings, thresholds |

## Configuration (`config.py`)

**`SCENARIO_RESULT_KEYS`** — maps scenario name to the loadgen log key used to extract the result:

```python
{
    "Offline":      "result_samples_per_second",
    "SingleStream": "early_stopping_latency_ss",
    "MultiStream":  "early_stopping_latency_ms",
    "Server":       "result_completed_samples_per_sec",
    "Interactive":  "result_completed_samples_per_sec",
}
```

**`MODEL_SCENARIO_RESULT_KEYS`** — overrides for specific `(model, scenario)` pairs:

```python
{
    ("deepseek-r1", "Server"): "result_completed_tokens_per_second",
}
```

**`ACCELERATOR_NAME_MAPPINGS`** — list of `(pattern, canonical_name)` pairs used to normalize accelerator names across marketing variants (e.g. `AMD Instinct MI355X 288GB HBM3e (x87)` → `AMD Instinct MI355X 288GB HBM3e`). First case-insensitive match wins.

**`ANOMALY_Z_THRESHOLD`** — z-score threshold for flagging anomalies (default `2.5`).

**`MIN_SAMPLES`** — minimum historical samples required to run the test (default `3`).

## Input files

### Loadgen detail log (`mlperf_log_detail.txt`)

Standard MLPerf loadgen output. The result value is extracted using the key appropriate for the selected scenario (see `SCENARIO_RESULT_KEYS` above).

### System description JSON

Standard MLPerf system description file. Fields used:

| Field | Description |
|---|---|
| `accelerator_model_name` | Raw accelerator name (standardized via config mapping) |
| `accelerators_per_node` | Number of accelerators per node |
| `number_of_nodes` | Number of nodes |

`total_accelerators = accelerators_per_node × number_of_nodes`

## Data

Sample files are provided in `data/`:

- `Inference v6.0 results - Results.csv` — historical results used as the baseline
- `mlperf_log_detail.txt` — example loadgen detail log
- `8xMI300X_2xEPYC_9575F.json` — example system description
