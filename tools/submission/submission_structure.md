# Submission directory structure

The following diagram describes the submission structure 

```
├── ...
│   ├── closed                                       
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
│   │   │   │   │   │   │   │   ├── mlperf_log_accuracy.txt
│   │   │   │   │   │   │   │   ├── mlperf_log_detail.txt
│   │   │   │   │   │   │   │   └── mlperf_log_summary.txt
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
│   │   │   │   │   │   ├── mlperf.conf
│   │   │   │   │   │   ├── user.conf
│   │   │   │   │   │   └── accuracy
│   │   │   ├── ...
│   │   │   └── <system_desc_id_n>
│   │   ├── systems
│   │   │   ├── <system_desc_id_1>.json
│   │   │   ├── ...
│   │   │   └── <system_desc_id_n>.json
│   │
│   ├── open                   
│   │   ├── code
│   │   ├── results
│   │   └── systems
...
```

## Description of placeholders

**<system_desc_id_X>:** Descriptive name of the system. 
**<benchmark_name>:** Name of the inference benchmark. E.g llama3.1-405b, rgat.
**<scenario>:** Name of the benchmarking scenario. One of `["SingleStream", "MultiStream", "Offline", "Server", "Interactive"]`
**<TEST0X>:** Compliance test number. E.g TEST01
