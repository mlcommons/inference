# Submission directory structure

## Standard submission structure

The following diagram describes the standard submission structure.

```
...
├── closed
│   ├── <submitter_name>                                       
│   │   ├── code
│   │   │   └── <benchmark_name>
│   │   ├── results
│   │   │   ├── <system_desc_id_1>
│   │   │   │   ├── <benchmark_name>
│   │   │   │   │   ├── <scenario>
│   │   │   │   │   │   ├── accuracy
│   │   │   │   │   │   │   ├── accuracy.txt
│   │   │   │   │   │   │   ├── mlperf_log_accuracy.txt
│   │   │   │   │   │   │   ├── mlperf_log_detail.txt
│   │   │   │   │   │   │   └── mlperf_log_summary.txt
│   │   │   │   │   │   ├── performance
│   │   │   │   │   │   │   ├── run_1
│   │   │   │   │   │   │   │   ├── mlperf_log_detail.txt
│   │   │   │   │   │   │   │   └── mlperf_log_summary.txt
│   │   │   │   │   │   │   ├── power (only for power submissions)
│   │   │   │   │   │   │   │   ├── client.json
│   │   │   │   │   │   │   │   ├── client.log
│   │   │   │   │   │   │   │   ├── ptd_log.txt
│   │   │   │   │   │   │   │   ├── server.json
│   │   │   │   │   │   │   │   └── server.log
│   │   │   │   │   │   │   ├── ranging (only for power submissions)
│   │   │   │   │   │   │   │   ├── mlperf_log_detail.txt
│   │   │   │   │   │   │   │   ├── mlperf_log_summary.txt
│   │   │   │   │   │   │   │   └── spl.txt
│   │   │   │   │   │   ├── <TEST0X>
│   │   │   │   │   │   │   ├── accuracy
│   │   │   │   │   │   │   │   ├── accuracy.txt
│   │   │   │   │   │   │   │   ├── baseline_accuracy.txt
│   │   │   │   │   │   │   │   ├── compliance_detail.txt
│   │   │   │   │   │   │   │   └── mlperf_log_accuracy.txt
│   │   │   │   │   │   │   ├── performance
│   │   │   │   │   │   │   │   ├── run_1
│   │   │   │   │   │   │   │   │   ├── mlperf_log_detail.txt
│   │   │   │   │   │   │   │   │   └── mlperf_log_summary.txt
│   │   │   │   │   │   ├── measurements.json
│   │   │   │   │   │   ├── mlperf.conf (optional)
│   │   │   │   │   │   └── user.conf
│   │   │   ├── ...
│   │   │   └── <system_desc_id_n>
│   │   ├── systems
│   │   │   ├── <system_desc_id_1>.json
│   │   │   ├── ...
│   │   │   └── <system_desc_id_n>.json
│   │
├── open
│   ├── <submitter_name>                   
│   │   ├── code
│   │   ├── results
│   │   ├── systems
│   │   └── model_mapping.json (optional)
...
```

## Endpoints submission structure

For endpoints submissions, the `mlperf_log_*.txt` files are replaced by structured JSON and YAML files produced by the endpoint harness. You can provide a performance+accuracy run in the performance folder or one performance run and one accuracy run

```
...
├── closed
│   ├── <submitter_name>
│   │   ├── code
│   │   │   └── <benchmark_name>
│   │   ├── results
│   │   │   ├── <system_desc_id_1>
│   │   │   │   ├── <benchmark_name>
│   │   │   │   │   ├── <scenario>
│   │   │   │   │   │   ├── accuracy (optional)
│   │   │   │   │   │   │   ├── config.yaml
│   │   │   │   │   │   │   ├── results.json
│   │   │   │   │   │   │   └── result_summary.json
│   │   │   │   │   │   ├── performance
│   │   │   │   │   │   │   ├── run_1
│   │   │   │   │   │   │   │   ├── config.yaml
│   │   │   │   │   │   │   │   ├── results.json
│   │   │   │   │   │   │   │   └── result_summary.json
│   │   │   │   │   │   ├── <TEST0X>
│   │   │   │   │   │   │   ├── accuracy
│   │   │   │   │   │   │   │   └── accuracy.txt
│   │   │   │   │   │   └── measurements.json
│   │   │   ├── ...
│   │   │   └── <system_desc_id_n>
│   │   ├── systems
│   │   │   ├── <system_desc_id_1>.json
│   │   │   ├── ...
│   │   │   └── <system_desc_id_n>.json
│   │
├── open
│   ├── <submitter_name>
│   │   ├── code
│   │   ├── results
│   │   ├── systems
│   │   └── model_mapping.json (optional)
...
```

## Description of placeholders

**<submitter_name>:** Name of the submitter
**<system_desc_id_X>:** Descriptive name of the system. 
**<benchmark_name>:** Name of the inference benchmark. E.g llama3.1-405b, rgat.
**<scenario>:** Name of the benchmarking scenario. One of `["SingleStream", "MultiStream", "Offline", "Server", "Interactive"]`
**<TEST0X>:** Compliance test number. E.g TEST01
**model_mapping.json (optional):** Optional file for the open submission that contains a map from the submitters custom names to the official benchmark names.
