# MLPerf Inference Submission Generation Example

## Prerequisites

### Clone the Repo
```bash
git clone -b submission-generation-examples https://github.com/mlcommons/inference.git submission-examples --depth 1
```

### Install mlc-scripts
```bash
pip install mlc-scripts
```

## Generate the Submission Tree

### Basic Command(submission_round_5.1)
```bash
mlc run script --tags=generate,mlperf,inference,submission \
--results_dir=submission-examples/submission_round_4.1/closed \
--run_checker=yes  \
--submission_dir=my_4.1_submissions  \
--quiet \
--submitter=MLCommons \
--division=closed \
--version=v5.1 \
--clean
```

Expected Output:
<details>

```
arjun@arjun-spr:~/tmp$ cm run script --tags=generate,mlperf,inference,submission --results_dir=submission-examples/closed --run_checker=yes  --submission_dir=my_submissions  --quiet --submitter=MLCommons --division=closed --quiet --clean
INFO:root:* cm run script "generate mlperf inference submission"
INFO:root:  * cm run script "get python3"
INFO:root:       ! load /home/arjun/CM/repos/local/cache/301a17a2c7b2405d/cm-cached-state.json
INFO:root:Path to Python: /home/arjun/gh_action/bin/python3
INFO:root:Python version: 3.12.3
INFO:root:  * cm run script "mlcommons inference src"
INFO:root:       ! load /home/arjun/CM/repos/local/cache/bf6768acc4fe4f03/cm-cached-state.json
INFO:root:  * cm run script "get sut system-description"
INFO:root:    * cm run script "detect os"
INFO:root:           ! cd /home/arjun/tmp
INFO:root:           ! call /home/arjun/CM/repos/gateoverflow@mlperf-automations/script/detect-os/run.sh from tmp-run.sh
INFO:root:           ! call "postprocess" from /home/arjun/CM/repos/gateoverflow@mlperf-automations/script/detect-os/customize.py
INFO:root:    * cm run script "detect cpu"
INFO:root:      * cm run script "detect os"
INFO:root:             ! cd /home/arjun/tmp
INFO:root:             ! call /home/arjun/CM/repos/gateoverflow@mlperf-automations/script/detect-os/run.sh from tmp-run.sh
INFO:root:             ! call "postprocess" from /home/arjun/CM/repos/gateoverflow@mlperf-automations/script/detect-os/customize.py
INFO:root:           ! cd /home/arjun/tmp
INFO:root:           ! call /home/arjun/CM/repos/gateoverflow@mlperf-automations/script/detect-cpu/run.sh from tmp-run.sh
INFO:root:           ! call "postprocess" from /home/arjun/CM/repos/gateoverflow@mlperf-automations/script/detect-cpu/customize.py
INFO:root:    * cm run script "get python3"
INFO:root:         ! load /home/arjun/CM/repos/local/cache/301a17a2c7b2405d/cm-cached-state.json
INFO:root:Path to Python: /home/arjun/gh_action/bin/python3
INFO:root:Python version: 3.12.3
INFO:root:    * cm run script "get compiler"
INFO:root:         ! load /home/arjun/CM/repos/local/cache/d7fbbb6d09ed48bb/cm-cached-state.json
INFO:root:    * cm run script "get generic-python-lib _package.dmiparser"
INFO:root:      * cm run script "get python3"
INFO:root:           ! load /home/arjun/CM/repos/local/cache/301a17a2c7b2405d/cm-cached-state.json
INFO:root:Path to Python: /home/arjun/gh_action/bin/python3
INFO:root:Python version: 3.12.3
INFO:root:           ! cd /home/arjun/tmp
INFO:root:           ! call /home/arjun/CM/repos/gateoverflow@mlperf-automations/script/get-generic-python-lib/validate_cache.sh from tmp-run.sh
INFO:root:           ! call "detect_version" from /home/arjun/CM/repos/gateoverflow@mlperf-automations/script/get-generic-python-lib/customize.py
          Detected version: 5.1
INFO:root:      * cm run script "get python3"
INFO:root:           ! load /home/arjun/CM/repos/local/cache/301a17a2c7b2405d/cm-cached-state.json
INFO:root:Path to Python: /home/arjun/gh_action/bin/python3
INFO:root:Python version: 3.12.3
INFO:root:           ! cd /home/arjun/tmp
INFO:root:           ! call /home/arjun/CM/repos/gateoverflow@mlperf-automations/script/get-generic-python-lib/validate_cache.sh from tmp-run.sh
INFO:root:           ! call "detect_version" from /home/arjun/CM/repos/gateoverflow@mlperf-automations/script/get-generic-python-lib/customize.py
          Detected version: 5.1
INFO:root:      * cm run script "get python3"
INFO:root:           ! load /home/arjun/CM/repos/local/cache/301a17a2c7b2405d/cm-cached-state.json
INFO:root:Path to Python: /home/arjun/gh_action/bin/python3
INFO:root:Python version: 3.12.3
INFO:root:         ! load /home/arjun/CM/repos/local/cache/1d10b72ca0374607/cm-cached-state.json
INFO:root:    * cm run script "get cache dir _name.mlperf-inference-sut-descriptions"
INFO:root:         ! load /home/arjun/CM/repos/local/cache/3b3315de9d31469a/cm-cached-state.json
Generating SUT description file for arjun_spr
INFO:root:         ! call "postprocess" from /home/arjun/CM/repos/gateoverflow@mlperf-automations/script/get-mlperf-inference-sut-description/customize.py
INFO:root:  * cm run script "install pip-package for-cmind-python _package.tabulate"
INFO:root:       ! load /home/arjun/CM/repos/local/cache/3e414d6e2b234ca6/cm-cached-state.json
INFO:root:  * cm run script "get mlperf inference utils"
INFO:root:    * cm run script "get mlperf inference src"
INFO:root:         ! load /home/arjun/CM/repos/local/cache/bf6768acc4fe4f03/cm-cached-state.json
INFO:root:         ! call "postprocess" from /home/arjun/CM/repos/gateoverflow@mlperf-automations/script/get-mlperf-inference-utils/customize.py
INFO:root:       ! call "postprocess" from /home/arjun/CM/repos/gateoverflow@mlperf-automations/script/generate-mlperf-inference-submission/customize.py
=================================================
Cleaning my_submissions ...
=================================================
* MLPerf inference submission dir: my_submissions
* MLPerf inference results dir: submission-examples/closed
* MLPerf inference division: closed
* MLPerf inference submitter: MLCommons
The SUT folder name for submission generation is: h13_u1_slim
* MLPerf inference model: resnet50
 * verify_accuracy.txt
 * verify_performance.txt
 * verify_performance.txt
{'accelerator_frequency': '', 'accelerator_host_interconnect': 'PCIe Gen5 16x (32 GT/s)', 'accelerator_interconnect': 'N/A', 'accelerator_interconnect_topology': 'N/A', 'accelerator_memory_capacity': 'disabled', 'accelerator_memory_configuration': 'LPDDR5 64x', 'accelerator_model_name': 'UntetherAI speedAI240 Slim', 'accelerator_on-chip_memories': '238 MB SRAM', 'accelerators_per_node': '1', 'cooling': 'air', 'division': 'closed', 'framework': 'UntetherAI imAIgine SDK v24.07.19', 'host_memory_capacity': '64 GB', 'host_memory_configuration': '4x 16 GB DDR5 (Samsung M321R2GA3PB0-CWMXJ 4800 MT/s)', 'host_network_card_count': '1', 'host_networking': 'integrated', 'host_networking_topology': '1GbE', 'host_processor_caches': 'L1d cache: 512 KiB (16 instances); L1i cache: 512 KiB (16 instances); L2 cache: 16 MiB (16 instances); L3 cache: 64 MiB (4 instances)', 'host_processor_core_count': '16', 'host_processor_frequency': '1500 MHz (min); 3000 MHz (base); 3700 MHz (boost)', 'host_processor_interconnect': 'N/A', 'host_processor_model_name': 'AMD EPYC 9124 16-Core Processor', 'host_processors_per_node': '1', 'host_storage_capacity': '931.5 GB', 'host_storage_type': 'Sabrent Rocket Q NVMe', 'hw_notes': 'SKU: sai240L-F-A-ES', 'number_of_nodes': '1', 'operating_system': 'Ubuntu 22.04.4 LTS (Linux kernel 6.5.0-44-generic #44~22.04.1-Ubuntu SMP PREEMPT_DYNAMIC Tue Jun 18 14:36:16 UTC 2 x86_64 x86_64 x86_64 GNU/Linux)', 'other_software_stack': {'KILT': 'mlperf_4.1', 'Docker': '27.1.0, build 6312585', 'Python': '3.10.12'}, 'status': 'available', 'submitter': 'MLCommons', 'sw_notes': 'Powered by the KRAI X and KILT technologies', 'system_name': 'Supermicro SuperServer H13 (1x speedAI240 Slim)', 'system_type': 'edge', 'system_type_detail': 'workstation', 'host_processor_url': 'https://www.amd.com/en/products/processors/server/epyc/4th-generation-9004-and-8004-series/amd-epyc-9124.html'}
 * mlperf_log_summary.txt
 * mlperf_log_detail.txt
 * verify_performance.txt
 * mlperf_log_summary.txt
 * mlperf_log_accuracy.json
 * mlperf_log_detail.txt
 * accuracy.txt
INFO:MLPerfLog:Sucessfully loaded MLPerf log from submission-examples/closed/h13_u1_slim/resnet50/multistream/performance/run_1/mlperf_log_detail.txt.
INFO:MLPerfLog:Sucessfully loaded MLPerf log from submission-examples/closed/h13_u1_slim/resnet50/multistream/performance/run_1/mlperf_log_detail.txt.
 * verify_accuracy.txt
 * verify_performance.txt
 * verify_performance.txt
{'accelerator_frequency': '', 'accelerator_host_interconnect': 'PCIe Gen5 16x (32 GT/s)', 'accelerator_interconnect': 'N/A', 'accelerator_interconnect_topology': 'N/A', 'accelerator_memory_capacity': 'disabled', 'accelerator_memory_configuration': 'LPDDR5 64x', 'accelerator_model_name': 'UntetherAI speedAI240 Slim', 'accelerator_on-chip_memories': '238 MB SRAM', 'accelerators_per_node': '1', 'cooling': 'air', 'division': 'closed', 'framework': 'UntetherAI imAIgine SDK v24.07.19', 'host_memory_capacity': '64 GB', 'host_memory_configuration': '4x 16 GB DDR5 (Samsung M321R2GA3PB0-CWMXJ 4800 MT/s)', 'host_network_card_count': '1', 'host_networking': 'integrated', 'host_networking_topology': '1GbE', 'host_processor_caches': 'L1d cache: 512 KiB (16 instances); L1i cache: 512 KiB (16 instances); L2 cache: 16 MiB (16 instances); L3 cache: 64 MiB (4 instances)', 'host_processor_core_count': '16', 'host_processor_frequency': '1500 MHz (min); 3000 MHz (base); 3700 MHz (boost)', 'host_processor_interconnect': 'N/A', 'host_processor_model_name': 'AMD EPYC 9124 16-Core Processor', 'host_processors_per_node': '1', 'host_storage_capacity': '931.5 GB', 'host_storage_type': 'Sabrent Rocket Q NVMe', 'hw_notes': 'SKU: sai240L-F-A-ES', 'number_of_nodes': '1', 'operating_system': 'Ubuntu 22.04.4 LTS (Linux kernel 6.5.0-44-generic #44~22.04.1-Ubuntu SMP PREEMPT_DYNAMIC Tue Jun 18 14:36:16 UTC 2 x86_64 x86_64 x86_64 GNU/Linux)', 'other_software_stack': {'KILT': 'mlperf_4.1', 'Docker': '27.1.0, build 6312585', 'Python': '3.10.12'}, 'status': 'available', 'submitter': 'MLCommons', 'sw_notes': 'Powered by the KRAI X and KILT technologies', 'system_name': 'Supermicro SuperServer H13 (1x speedAI240 Slim)', 'system_type': 'edge', 'system_type_detail': 'workstation', 'host_processor_url': 'https://www.amd.com/en/products/processors/server/epyc/4th-generation-9004-and-8004-series/amd-epyc-9124.html'}
 * mlperf_log_summary.txt
 * mlperf_log_detail.txt
 * verify_performance.txt
 * mlperf_log_summary.txt
 * mlperf_log_accuracy.json
 * mlperf_log_detail.txt
 * accuracy.txt
INFO:MLPerfLog:Sucessfully loaded MLPerf log from submission-examples/closed/h13_u1_slim/resnet50/singlestream/performance/run_1/mlperf_log_detail.txt.
INFO:MLPerfLog:Sucessfully loaded MLPerf log from submission-examples/closed/h13_u1_slim/resnet50/singlestream/performance/run_1/mlperf_log_detail.txt.
 * verify_accuracy.txt
 * verify_performance.txt
 * verify_performance.txt
{'accelerator_frequency': '', 'accelerator_host_interconnect': 'PCIe Gen5 16x (32 GT/s)', 'accelerator_interconnect': 'N/A', 'accelerator_interconnect_topology': 'N/A', 'accelerator_memory_capacity': 'disabled', 'accelerator_memory_configuration': 'LPDDR5 64x', 'accelerator_model_name': 'UntetherAI speedAI240 Slim', 'accelerator_on-chip_memories': '238 MB SRAM', 'accelerators_per_node': '1', 'cooling': 'air', 'division': 'closed', 'framework': 'UntetherAI imAIgine SDK v24.07.19', 'host_memory_capacity': '64 GB', 'host_memory_configuration': '4x 16 GB DDR5 (Samsung M321R2GA3PB0-CWMXJ 4800 MT/s)', 'host_network_card_count': '1', 'host_networking': 'integrated', 'host_networking_topology': '1GbE', 'host_processor_caches': 'L1d cache: 512 KiB (16 instances); L1i cache: 512 KiB (16 instances); L2 cache: 16 MiB (16 instances); L3 cache: 64 MiB (4 instances)', 'host_processor_core_count': '16', 'host_processor_frequency': '1500 MHz (min); 3000 MHz (base); 3700 MHz (boost)', 'host_processor_interconnect': 'N/A', 'host_processor_model_name': 'AMD EPYC 9124 16-Core Processor', 'host_processors_per_node': '1', 'host_storage_capacity': '931.5 GB', 'host_storage_type': 'Sabrent Rocket Q NVMe', 'hw_notes': 'SKU: sai240L-F-A-ES', 'number_of_nodes': '1', 'operating_system': 'Ubuntu 22.04.4 LTS (Linux kernel 6.5.0-44-generic #44~22.04.1-Ubuntu SMP PREEMPT_DYNAMIC Tue Jun 18 14:36:16 UTC 2 x86_64 x86_64 x86_64 GNU/Linux)', 'other_software_stack': {'KILT': 'mlperf_4.1', 'Docker': '27.1.0, build 6312585', 'Python': '3.10.12'}, 'status': 'available', 'submitter': 'MLCommons', 'sw_notes': 'Powered by the KRAI X and KILT technologies', 'system_name': 'Supermicro SuperServer H13 (1x speedAI240 Slim)', 'system_type': 'edge', 'system_type_detail': 'workstation', 'host_processor_url': 'https://www.amd.com/en/products/processors/server/epyc/4th-generation-9004-and-8004-series/amd-epyc-9124.html'}
 * mlperf_log_summary.txt
 * mlperf_log_detail.txt
 * verify_performance.txt
 * mlperf_log_summary.txt
 * mlperf_log_accuracy.json
 * mlperf_log_detail.txt
 * accuracy.txt
INFO:MLPerfLog:Sucessfully loaded MLPerf log from submission-examples/closed/h13_u1_slim/resnet50/offline/performance/run_1/mlperf_log_detail.txt.
INFO:MLPerfLog:Sucessfully loaded MLPerf log from submission-examples/closed/h13_u1_slim/resnet50/offline/performance/run_1/mlperf_log_detail.txt.
+----------+--------------+----------+------------+-----------------+---------------------------------+--------+--------+
|  Model   |   Scenario   | Accuracy | Throughput | Latency (in ms) | Power Efficiency (in samples/J) | TEST01 | TEST04 |
+----------+--------------+----------+------------+-----------------+---------------------------------+--------+--------+
| resnet50 | multistream  |  75.912  | 28070.175  |      0.285      |                                 | passed | passed |
| resnet50 | singlestream |  75.912  |  8264.463  |      0.121      |                                 | passed | passed |
| resnet50 |   offline    |  75.912  |  56277.1   |        -        |                                 | passed | passed |
+----------+--------------+----------+------------+-----------------+---------------------------------+--------+--------+
The SUT folder name for submission generation is: R760xa_L40Sx4_TRT
* MLPerf inference model: resnet50
 * verify_accuracy.txt
 * verify_performance.txt
 * verify_performance.txt
{'accelerator_frequency': '', 'accelerator_host_interconnect': 'PCIe 4.0 x16', 'accelerator_interconnect': 'PCIe 4.0 x16', 'accelerator_interconnect_topology': '', 'accelerator_memory_capacity': '48 GB', 'accelerator_memory_configuration': 'GDDR6', 'accelerator_model_name': 'NVIDIA L40S', 'accelerator_on-chip_memories': '', 'accelerators_per_node': 4, 'cooling': 'air-cooled', 'division': 'closed', 'framework': 'TensorRT 10.2.0, CUDA 12.4', 'host_memory_capacity': '512 GB', 'host_memory_configuration': '16x 32GB 3200 MT/s', 'host_network_card_count': '2x 1GbE', 'host_networking': 'Ethernet', 'host_networking_topology': 'N/A', 'host_processor_caches': 'L1d cache: 1.1 MiB (24 instances), L1i cache: 768 KiB (24 instances), L2 cache: 48 MiB (24 instances), L3 cache: 45 MiB (1 instance)', 'host_processor_core_count': 64, 'host_processor_frequency': '4800.0000', 'host_processor_interconnect': '', 'host_processor_model_name': 'Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz', 'host_processors_per_node': 2, 'host_storage_capacity': '3.5 TB', 'host_storage_type': 'SSD', 'hw_notes': '', 'number_of_nodes': 1, 'operating_system': 'Rocky Linux 9.1', 'other_software_stack': 'TensorRT 10.2.0, CUDA 12.4, cuDNN 8.9.7, Driver 550.54', 'status': 'available', 'submitter': 'MLCommons', 'sw_notes': '', 'system_name': 'Dell PowerEdge R760xa (4x L40S, TensorRT)', 'system_type': 'datacenter', 'system_type_detail': 'N/A'}
 * mlperf_log_summary.txt
 * mlperf_log_accuracy.json
 * mlperf_log_detail.txt
 * verify_performance.txt
 * mlperf_log_summary.txt
 * mlperf_log_accuracy.json
 * mlperf_log_detail.txt
 * accuracy.txt
INFO:MLPerfLog:Sucessfully loaded MLPerf log from submission-examples/closed/R760xa_L40Sx4_TRT/resnet50/Server/performance/run_1/mlperf_log_detail.txt.
INFO:MLPerfLog:Sucessfully loaded MLPerf log from submission-examples/closed/R760xa_L40Sx4_TRT/resnet50/Server/performance/run_1/mlperf_log_detail.txt.
 * verify_accuracy.txt
 * verify_performance.txt
 * verify_performance.txt
{'accelerator_frequency': '', 'accelerator_host_interconnect': 'PCIe 4.0 x16', 'accelerator_interconnect': 'PCIe 4.0 x16', 'accelerator_interconnect_topology': '', 'accelerator_memory_capacity': '48 GB', 'accelerator_memory_configuration': 'GDDR6', 'accelerator_model_name': 'NVIDIA L40S', 'accelerator_on-chip_memories': '', 'accelerators_per_node': 4, 'cooling': 'air-cooled', 'division': 'closed', 'framework': 'TensorRT 10.2.0, CUDA 12.4', 'host_memory_capacity': '512 GB', 'host_memory_configuration': '16x 32GB 3200 MT/s', 'host_network_card_count': '2x 1GbE', 'host_networking': 'Ethernet', 'host_networking_topology': 'N/A', 'host_processor_caches': 'L1d cache: 1.1 MiB (24 instances), L1i cache: 768 KiB (24 instances), L2 cache: 48 MiB (24 instances), L3 cache: 45 MiB (1 instance)', 'host_processor_core_count': 64, 'host_processor_frequency': '4800.0000', 'host_processor_interconnect': '', 'host_processor_model_name': 'Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz', 'host_processors_per_node': 2, 'host_storage_capacity': '3.5 TB', 'host_storage_type': 'SSD', 'hw_notes': '', 'number_of_nodes': 1, 'operating_system': 'Rocky Linux 9.1', 'other_software_stack': 'TensorRT 10.2.0, CUDA 12.4, cuDNN 8.9.7, Driver 550.54', 'status': 'available', 'submitter': 'MLCommons', 'sw_notes': '', 'system_name': 'Dell PowerEdge R760xa (4x L40S, TensorRT)', 'system_type': 'datacenter', 'system_type_detail': 'N/A'}
 * mlperf_log_summary.txt
 * mlperf_log_accuracy.json
 * mlperf_log_detail.txt
 * verify_performance.txt
 * mlperf_log_summary.txt
 * mlperf_log_accuracy.json
 * mlperf_log_detail.txt
 * accuracy.txt
INFO:MLPerfLog:Sucessfully loaded MLPerf log from submission-examples/closed/R760xa_L40Sx4_TRT/resnet50/Offline/performance/run_1/mlperf_log_detail.txt.
INFO:MLPerfLog:Sucessfully loaded MLPerf log from submission-examples/closed/R760xa_L40Sx4_TRT/resnet50/Offline/performance/run_1/mlperf_log_detail.txt.
* MLPerf inference model: retinanet
 * verify_accuracy.txt
 * verify_performance.txt
{'accelerator_frequency': '', 'accelerator_host_interconnect': 'PCIe 4.0 x16', 'accelerator_interconnect': 'PCIe 4.0 x16', 'accelerator_interconnect_topology': '', 'accelerator_memory_capacity': '48 GB', 'accelerator_memory_configuration': 'GDDR6', 'accelerator_model_name': 'NVIDIA L40S', 'accelerator_on-chip_memories': '', 'accelerators_per_node': 4, 'cooling': 'air-cooled', 'division': 'closed', 'framework': 'TensorRT 10.2.0, CUDA 12.4', 'host_memory_capacity': '512 GB', 'host_memory_configuration': '16x 32GB 3200 MT/s', 'host_network_card_count': '2x 1GbE', 'host_networking': 'Ethernet', 'host_networking_topology': 'N/A', 'host_processor_caches': 'L1d cache: 1.1 MiB (24 instances), L1i cache: 768 KiB (24 instances), L2 cache: 48 MiB (24 instances), L3 cache: 45 MiB (1 instance)', 'host_processor_core_count': 64, 'host_processor_frequency': '4800.0000', 'host_processor_interconnect': '', 'host_processor_model_name': 'Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz', 'host_processors_per_node': 2, 'host_storage_capacity': '3.5 TB', 'host_storage_type': 'SSD', 'hw_notes': '', 'number_of_nodes': 1, 'operating_system': 'Rocky Linux 9.1', 'other_software_stack': 'TensorRT 10.2.0, CUDA 12.4, cuDNN 8.9.7, Driver 550.54', 'status': 'available', 'submitter': 'MLCommons', 'sw_notes': '', 'system_name': 'Dell PowerEdge R760xa (4x L40S, TensorRT)', 'system_type': 'datacenter', 'system_type_detail': 'N/A'}
 * mlperf_log_summary.txt
 * mlperf_log_accuracy.json
 * mlperf_log_detail.txt
 * verify_performance.txt
 * mlperf_log_summary.txt
 * mlperf_log_accuracy.json
 * mlperf_log_detail.txt
 * accuracy.txt
INFO:MLPerfLog:Sucessfully loaded MLPerf log from submission-examples/closed/R760xa_L40Sx4_TRT/retinanet/Server/performance/run_1/mlperf_log_detail.txt.
INFO:MLPerfLog:Sucessfully loaded MLPerf log from submission-examples/closed/R760xa_L40Sx4_TRT/retinanet/Server/performance/run_1/mlperf_log_detail.txt.
 * verify_accuracy.txt
 * verify_performance.txt
{'accelerator_frequency': '', 'accelerator_host_interconnect': 'PCIe 4.0 x16', 'accelerator_interconnect': 'PCIe 4.0 x16', 'accelerator_interconnect_topology': '', 'accelerator_memory_capacity': '48 GB', 'accelerator_memory_configuration': 'GDDR6', 'accelerator_model_name': 'NVIDIA L40S', 'accelerator_on-chip_memories': '', 'accelerators_per_node': 4, 'cooling': 'air-cooled', 'division': 'closed', 'framework': 'TensorRT 10.2.0, CUDA 12.4', 'host_memory_capacity': '512 GB', 'host_memory_configuration': '16x 32GB 3200 MT/s', 'host_network_card_count': '2x 1GbE', 'host_networking': 'Ethernet', 'host_networking_topology': 'N/A', 'host_processor_caches': 'L1d cache: 1.1 MiB (24 instances), L1i cache: 768 KiB (24 instances), L2 cache: 48 MiB (24 instances), L3 cache: 45 MiB (1 instance)', 'host_processor_core_count': 64, 'host_processor_frequency': '4800.0000', 'host_processor_interconnect': '', 'host_processor_model_name': 'Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz', 'host_processors_per_node': 2, 'host_storage_capacity': '3.5 TB', 'host_storage_type': 'SSD', 'hw_notes': '', 'number_of_nodes': 1, 'operating_system': 'Rocky Linux 9.1', 'other_software_stack': 'TensorRT 10.2.0, CUDA 12.4, cuDNN 8.9.7, Driver 550.54', 'status': 'available', 'submitter': 'MLCommons', 'sw_notes': '', 'system_name': 'Dell PowerEdge R760xa (4x L40S, TensorRT)', 'system_type': 'datacenter', 'system_type_detail': 'N/A'}
 * mlperf_log_summary.txt
 * mlperf_log_accuracy.json
 * mlperf_log_detail.txt
 * verify_performance.txt
 * mlperf_log_summary.txt
 * mlperf_log_accuracy.json
 * mlperf_log_detail.txt
 * accuracy.txt
INFO:MLPerfLog:Sucessfully loaded MLPerf log from submission-examples/closed/R760xa_L40Sx4_TRT/retinanet/Offline/performance/run_1/mlperf_log_detail.txt.
INFO:MLPerfLog:Sucessfully loaded MLPerf log from submission-examples/closed/R760xa_L40Sx4_TRT/retinanet/Offline/performance/run_1/mlperf_log_detail.txt.
+-----------+----------+----------+------------+-----------------+---------------------------------+--------+--------+
|   Model   | Scenario | Accuracy | Throughput | Latency (in ms) | Power Efficiency (in samples/J) | TEST01 | TEST04 |
+-----------+----------+----------+------------+-----------------+---------------------------------+--------+--------+
| resnet50  |  Server  |  76.078  |  181231.0  |        -        |                                 | passed | passed |
| resnet50  | Offline  |  76.078  |  180613.0  |        -        |                                 | passed | passed |
| retinanet |  Server  |  37.296  |  3152.43   |        -        |                                 | passed |        |
| retinanet | Offline  |  37.352  |  3345.71   |        -        |                                 | passed |        |
+-----------+----------+----------+------------+-----------------+---------------------------------+--------+--------+
The SUT folder name for submission generation is: XE9680_H200_SXM_141GBx8_TRT
* MLPerf inference model: resnet50
 * verify_accuracy.txt
 * verify_performance.txt
 * verify_performance.txt
{'accelerator_frequency': '', 'accelerator_host_interconnect': 'PCIe Gen5 x16', 'accelerator_interconnect': 'NVLINK', 'accelerator_interconnect_topology': 'NVLINK Switch', 'accelerator_memory_capacity': '141 GB', 'accelerator_memory_configuration': 'HBM3e', 'accelerator_model_name': 'NVIDIA H200-SXM-141GB', 'accelerator_on-chip_memories': '', 'accelerators_per_node': 8, 'cooling': 'air-cooled', 'division': 'closed', 'framework': 'TensorRT 10.2.0, CUDA 12.4', 'host_memory_capacity': '2 TB', 'host_memory_configuration': '32x 64GB DDR5', 'host_network_card_count': '10x 400Gb Infiniband', 'host_networking': 'Infiniband:Data bandwidth for GPU-PCIe: 504GB/s; PCIe-NIC: 500GB/s', 'host_networking_topology': 'Ethernet/Infiniband on switching network', 'host_processor_caches': 'L1d cache: 1.1 MiB (24 instances), L1i cache: 768 KiB (24 instances), L2 cache: 48 MiB (24 instances), L3 cache: 45 MiB (1 instance)', 'host_processor_core_count': 52, 'host_processor_frequency': '4800.0000', 'host_processor_interconnect': '', 'host_processor_model_name': 'Intel(R) Xeon(R) Platinum 8470', 'host_processors_per_node': 2, 'host_storage_capacity': '7.68 TB', 'host_storage_type': 'NVMe SSD', 'hw_notes': 'H200 TGP 700W', 'number_of_nodes': 1, 'operating_system': 'Ubuntu 22.04.3', 'other_software_stack': 'TensorRT 10.2.0, CUDA 12.5, cuDNN 8.9.7, Driver 555.42.06', 'status': 'available', 'submitter': 'MLCommons', 'sw_notes': '', 'system_name': 'Dell PowerEdge XE9680 (8x H200-SXM-141GB, TensorRT)', 'system_type': 'datacenter', 'system_type_detail': 'N/A', 'disk_controllers': 'NVMe'}
 * mlperf_log_summary.txt
 * mlperf_log_accuracy.json
 * mlperf_log_detail.txt
 * verify_performance.txt
 * mlperf_log_summary.txt
 * mlperf_log_accuracy.json
 * mlperf_log_detail.txt
 * accuracy.txt
INFO:MLPerfLog:Sucessfully loaded MLPerf log from submission-examples/closed/XE9680_H200_SXM_141GBx8_TRT/resnet50/Server/performance/run_1/mlperf_log_detail.txt.
INFO:MLPerfLog:Sucessfully loaded MLPerf log from submission-examples/closed/XE9680_H200_SXM_141GBx8_TRT/resnet50/Server/performance/run_1/mlperf_log_detail.txt.
 * verify_accuracy.txt
 * verify_performance.txt
 * verify_performance.txt
{'accelerator_frequency': '', 'accelerator_host_interconnect': 'PCIe Gen5 x16', 'accelerator_interconnect': 'NVLINK', 'accelerator_interconnect_topology': 'NVLINK Switch', 'accelerator_memory_capacity': '141 GB', 'accelerator_memory_configuration': 'HBM3e', 'accelerator_model_name': 'NVIDIA H200-SXM-141GB', 'accelerator_on-chip_memories': '', 'accelerators_per_node': 8, 'cooling': 'air-cooled', 'division': 'closed', 'framework': 'TensorRT 10.2.0, CUDA 12.4', 'host_memory_capacity': '2 TB', 'host_memory_configuration': '32x 64GB DDR5', 'host_network_card_count': '10x 400Gb Infiniband', 'host_networking': 'Infiniband:Data bandwidth for GPU-PCIe: 504GB/s; PCIe-NIC: 500GB/s', 'host_networking_topology': 'Ethernet/Infiniband on switching network', 'host_processor_caches': 'L1d cache: 1.1 MiB (24 instances), L1i cache: 768 KiB (24 instances), L2 cache: 48 MiB (24 instances), L3 cache: 45 MiB (1 instance)', 'host_processor_core_count': 52, 'host_processor_frequency': '4800.0000', 'host_processor_interconnect': '', 'host_processor_model_name': 'Intel(R) Xeon(R) Platinum 8470', 'host_processors_per_node': 2, 'host_storage_capacity': '7.68 TB', 'host_storage_type': 'NVMe SSD', 'hw_notes': 'H200 TGP 700W', 'number_of_nodes': 1, 'operating_system': 'Ubuntu 22.04.3', 'other_software_stack': 'TensorRT 10.2.0, CUDA 12.5, cuDNN 8.9.7, Driver 555.42.06', 'status': 'available', 'submitter': 'MLCommons', 'sw_notes': '', 'system_name': 'Dell PowerEdge XE9680 (8x H200-SXM-141GB, TensorRT)', 'system_type': 'datacenter', 'system_type_detail': 'N/A', 'disk_controllers': 'NVMe'}
 * mlperf_log_summary.txt
 * mlperf_log_accuracy.json
 * mlperf_log_detail.txt
 * verify_performance.txt
 * mlperf_log_summary.txt
 * mlperf_log_accuracy.json
 * mlperf_log_detail.txt
 * accuracy.txt
INFO:MLPerfLog:Sucessfully loaded MLPerf log from submission-examples/closed/XE9680_H200_SXM_141GBx8_TRT/resnet50/Offline/performance/run_1/mlperf_log_detail.txt.
INFO:MLPerfLog:Sucessfully loaded MLPerf log from submission-examples/closed/XE9680_H200_SXM_141GBx8_TRT/resnet50/Offline/performance/run_1/mlperf_log_detail.txt.
* MLPerf inference model: retinanet
 * verify_accuracy.txt
 * verify_performance.txt
{'accelerator_frequency': '', 'accelerator_host_interconnect': 'PCIe Gen5 x16', 'accelerator_interconnect': 'NVLINK', 'accelerator_interconnect_topology': 'NVLINK Switch', 'accelerator_memory_capacity': '141 GB', 'accelerator_memory_configuration': 'HBM3e', 'accelerator_model_name': 'NVIDIA H200-SXM-141GB', 'accelerator_on-chip_memories': '', 'accelerators_per_node': 8, 'cooling': 'air-cooled', 'division': 'closed', 'framework': 'TensorRT 10.2.0, CUDA 12.4', 'host_memory_capacity': '2 TB', 'host_memory_configuration': '32x 64GB DDR5', 'host_network_card_count': '10x 400Gb Infiniband', 'host_networking': 'Infiniband:Data bandwidth for GPU-PCIe: 504GB/s; PCIe-NIC: 500GB/s', 'host_networking_topology': 'Ethernet/Infiniband on switching network', 'host_processor_caches': 'L1d cache: 1.1 MiB (24 instances), L1i cache: 768 KiB (24 instances), L2 cache: 48 MiB (24 instances), L3 cache: 45 MiB (1 instance)', 'host_processor_core_count': 52, 'host_processor_frequency': '4800.0000', 'host_processor_interconnect': '', 'host_processor_model_name': 'Intel(R) Xeon(R) Platinum 8470', 'host_processors_per_node': 2, 'host_storage_capacity': '7.68 TB', 'host_storage_type': 'NVMe SSD', 'hw_notes': 'H200 TGP 700W', 'number_of_nodes': 1, 'operating_system': 'Ubuntu 22.04.3', 'other_software_stack': 'TensorRT 10.2.0, CUDA 12.5, cuDNN 8.9.7, Driver 555.42.06', 'status': 'available', 'submitter': 'MLCommons', 'sw_notes': '', 'system_name': 'Dell PowerEdge XE9680 (8x H200-SXM-141GB, TensorRT)', 'system_type': 'datacenter', 'system_type_detail': 'N/A', 'disk_controllers': 'NVMe'}
 * mlperf_log_summary.txt
 * mlperf_log_accuracy.json
 * mlperf_log_detail.txt
 * verify_performance.txt
 * mlperf_log_summary.txt
 * mlperf_log_accuracy.json
 * mlperf_log_detail.txt
 * accuracy.txt
INFO:MLPerfLog:Sucessfully loaded MLPerf log from submission-examples/closed/XE9680_H200_SXM_141GBx8_TRT/retinanet/Server/performance/run_1/mlperf_log_detail.txt.
INFO:MLPerfLog:Sucessfully loaded MLPerf log from submission-examples/closed/XE9680_H200_SXM_141GBx8_TRT/retinanet/Server/performance/run_1/mlperf_log_detail.txt.
 * verify_accuracy.txt
 * verify_performance.txt
{'accelerator_frequency': '', 'accelerator_host_interconnect': 'PCIe Gen5 x16', 'accelerator_interconnect': 'NVLINK', 'accelerator_interconnect_topology': 'NVLINK Switch', 'accelerator_memory_capacity': '141 GB', 'accelerator_memory_configuration': 'HBM3e', 'accelerator_model_name': 'NVIDIA H200-SXM-141GB', 'accelerator_on-chip_memories': '', 'accelerators_per_node': 8, 'cooling': 'air-cooled', 'division': 'closed', 'framework': 'TensorRT 10.2.0, CUDA 12.4', 'host_memory_capacity': '2 TB', 'host_memory_configuration': '32x 64GB DDR5', 'host_network_card_count': '10x 400Gb Infiniband', 'host_networking': 'Infiniband:Data bandwidth for GPU-PCIe: 504GB/s; PCIe-NIC: 500GB/s', 'host_networking_topology': 'Ethernet/Infiniband on switching network', 'host_processor_caches': 'L1d cache: 1.1 MiB (24 instances), L1i cache: 768 KiB (24 instances), L2 cache: 48 MiB (24 instances), L3 cache: 45 MiB (1 instance)', 'host_processor_core_count': 52, 'host_processor_frequency': '4800.0000', 'host_processor_interconnect': '', 'host_processor_model_name': 'Intel(R) Xeon(R) Platinum 8470', 'host_processors_per_node': 2, 'host_storage_capacity': '7.68 TB', 'host_storage_type': 'NVMe SSD', 'hw_notes': 'H200 TGP 700W', 'number_of_nodes': 1, 'operating_system': 'Ubuntu 22.04.3', 'other_software_stack': 'TensorRT 10.2.0, CUDA 12.5, cuDNN 8.9.7, Driver 555.42.06', 'status': 'available', 'submitter': 'MLCommons', 'sw_notes': '', 'system_name': 'Dell PowerEdge XE9680 (8x H200-SXM-141GB, TensorRT)', 'system_type': 'datacenter', 'system_type_detail': 'N/A', 'disk_controllers': 'NVMe'}
 * mlperf_log_summary.txt
 * mlperf_log_accuracy.json
 * mlperf_log_detail.txt
 * verify_performance.txt
 * mlperf_log_summary.txt
 * mlperf_log_accuracy.json
 * mlperf_log_detail.txt
 * accuracy.txt
INFO:MLPerfLog:Sucessfully loaded MLPerf log from submission-examples/closed/XE9680_H200_SXM_141GBx8_TRT/retinanet/Offline/performance/run_1/mlperf_log_detail.txt.
INFO:MLPerfLog:Sucessfully loaded MLPerf log from submission-examples/closed/XE9680_H200_SXM_141GBx8_TRT/retinanet/Offline/performance/run_1/mlperf_log_detail.txt.
+-----------+----------+----------+------------+-----------------+---------------------------------+--------+--------+
|   Model   | Scenario | Accuracy | Throughput | Latency (in ms) | Power Efficiency (in samples/J) | TEST01 | TEST04 |
+-----------+----------+----------+------------+-----------------+---------------------------------+--------+--------+
| resnet50  |  Server  |  76.078  |  630226.0  |        -        |                                 | passed | passed |
| resnet50  | Offline  |  76.078  |  768235.0  |        -        |                                 | passed | passed |
| retinanet |  Server  |  37.328  |  13603.8   |        -        |                                 | passed |        |
| retinanet | Offline  |  37.329  |  14760.1   |        -        |                                 | passed |        |
+-----------+----------+----------+------------+-----------------+---------------------------------+--------+--------+
INFO:root:* cm run script "accuracy truncate mlc"
INFO:root:  * cm run script "get python3"
INFO:root:       ! load /home/arjun/CM/repos/local/cache/301a17a2c7b2405d/cm-cached-state.json
INFO:root:Path to Python: /home/arjun/gh_action/bin/python3
INFO:root:Python version: 3.12.3
INFO:root:  * cm run script "get mlcommons inference src"
INFO:root:       ! load /home/arjun/CM/repos/local/cache/bf6768acc4fe4f03/cm-cached-state.json
INFO:root:       ! cd /home/arjun/tmp
INFO:root:       ! call /home/arjun/CM/repos/gateoverflow@mlperf-automations/script/truncate-mlperf-inference-accuracy-log/run.sh from tmp-run.sh
python3 '/home/arjun/CM/repos/local/cache/12e5ddb36c4e4ee0/inference/tools/submission/truncate_accuracy_log.py' --input 'my_submissions' --submitter 'MLCommons' --backup 'my_submissions_logs'
INFO:main:closed/MLCommons/results/h13_u1_slim/resnet50/multistream/accuracy already has hash and size seems truncated
INFO:main:closed/MLCommons/results/h13_u1_slim/resnet50/multistream/accuracy already has hash and size seems truncated
INFO:main:closed/MLCommons/results/h13_u1_slim/resnet50/singlestream/accuracy already has hash and size seems truncated
INFO:main:closed/MLCommons/results/h13_u1_slim/resnet50/singlestream/accuracy already has hash and size seems truncated
INFO:main:closed/MLCommons/results/h13_u1_slim/resnet50/offline/accuracy already has hash and size seems truncated
INFO:main:closed/MLCommons/results/h13_u1_slim/resnet50/offline/accuracy already has hash and size seems truncated
INFO:main:closed/MLCommons/results/R760xa_L40Sx4_TRT/resnet50/Server/accuracy already has hash and size seems truncated
INFO:main:closed/MLCommons/results/R760xa_L40Sx4_TRT/resnet50/Server/accuracy already has hash and size seems truncated
INFO:main:closed/MLCommons/results/R760xa_L40Sx4_TRT/resnet50/Offline/accuracy already has hash and size seems truncated
INFO:main:closed/MLCommons/results/R760xa_L40Sx4_TRT/resnet50/Offline/accuracy already has hash and size seems truncated
INFO:main:closed/MLCommons/results/R760xa_L40Sx4_TRT/retinanet/Server/accuracy already has hash and size seems truncated
INFO:main:closed/MLCommons/results/R760xa_L40Sx4_TRT/retinanet/Server/accuracy already has hash and size seems truncated
INFO:main:closed/MLCommons/results/R760xa_L40Sx4_TRT/retinanet/Offline/accuracy already has hash and size seems truncated
INFO:main:closed/MLCommons/results/R760xa_L40Sx4_TRT/retinanet/Offline/accuracy already has hash and size seems truncated
INFO:main:closed/MLCommons/results/XE9680_H200_SXM_141GBx8_TRT/resnet50/Server/accuracy already has hash and size seems truncated
INFO:main:closed/MLCommons/results/XE9680_H200_SXM_141GBx8_TRT/resnet50/Server/accuracy already has hash and size seems truncated
INFO:main:closed/MLCommons/results/XE9680_H200_SXM_141GBx8_TRT/resnet50/Offline/accuracy already has hash and size seems truncated
INFO:main:closed/MLCommons/results/XE9680_H200_SXM_141GBx8_TRT/resnet50/Offline/accuracy already has hash and size seems truncated
INFO:main:closed/MLCommons/results/XE9680_H200_SXM_141GBx8_TRT/retinanet/Server/accuracy already has hash and size seems truncated
INFO:main:closed/MLCommons/results/XE9680_H200_SXM_141GBx8_TRT/retinanet/Server/accuracy already has hash and size seems truncated
INFO:main:closed/MLCommons/results/XE9680_H200_SXM_141GBx8_TRT/retinanet/Offline/accuracy already has hash and size seems truncated
INFO:main:closed/MLCommons/results/XE9680_H200_SXM_141GBx8_TRT/retinanet/Offline/accuracy already has hash and size seems truncated
INFO:main:closed/MLCommons/compliance/h13_u1_slim/resnet50/multistream/TEST01/accuracy already has hash and size seems truncated
INFO:main:closed/MLCommons/compliance/h13_u1_slim/resnet50/singlestream/TEST01/accuracy already has hash and size seems truncated
INFO:main:closed/MLCommons/compliance/h13_u1_slim/resnet50/offline/TEST01/accuracy already has hash and size seems truncated
INFO:main:closed/MLCommons/compliance/R760xa_L40Sx4_TRT/resnet50/Server/TEST01/accuracy already has hash and size seems truncated
INFO:main:closed/MLCommons/compliance/R760xa_L40Sx4_TRT/resnet50/Offline/TEST01/accuracy already has hash and size seems truncated
INFO:main:closed/MLCommons/compliance/R760xa_L40Sx4_TRT/retinanet/Server/TEST01/accuracy already has hash and size seems truncated
INFO:main:closed/MLCommons/compliance/R760xa_L40Sx4_TRT/retinanet/Offline/TEST01/accuracy already has hash and size seems truncated
INFO:main:closed/MLCommons/compliance/XE9680_H200_SXM_141GBx8_TRT/resnet50/Server/TEST01/accuracy already has hash and size seems truncated
INFO:main:closed/MLCommons/compliance/XE9680_H200_SXM_141GBx8_TRT/resnet50/Offline/TEST01/accuracy already has hash and size seems truncated
INFO:main:closed/MLCommons/compliance/XE9680_H200_SXM_141GBx8_TRT/retinanet/Server/TEST01/accuracy already has hash and size seems truncated
INFO:main:closed/MLCommons/compliance/XE9680_H200_SXM_141GBx8_TRT/retinanet/Offline/TEST01/accuracy already has hash and size seems truncated
INFO:main:Make sure you keep a backup of my_submissions_logs in case mlperf wants to see the original accuracy logs
INFO:root:* cm run script "submission inference checker mlc"
INFO:root:  * cm run script "get python3"
INFO:root:       ! load /home/arjun/CM/repos/local/cache/301a17a2c7b2405d/cm-cached-state.json
INFO:root:Path to Python: /home/arjun/gh_action/bin/python3
INFO:root:Python version: 3.12.3
INFO:root:  * cm run script "get mlcommons inference src"
INFO:root:       ! load /home/arjun/CM/repos/local/cache/bf6768acc4fe4f03/cm-cached-state.json
INFO:root:  * cm run script "get generic-python-lib _xlsxwriter"
INFO:root:    * cm run script "get python3"
INFO:root:         ! load /home/arjun/CM/repos/local/cache/301a17a2c7b2405d/cm-cached-state.json
INFO:root:Path to Python: /home/arjun/gh_action/bin/python3
INFO:root:Python version: 3.12.3
INFO:root:         ! cd /home/arjun/tmp
INFO:root:         ! call /home/arjun/CM/repos/gateoverflow@mlperf-automations/script/get-generic-python-lib/validate_cache.sh from tmp-run.sh
INFO:root:         ! call "detect_version" from /home/arjun/CM/repos/gateoverflow@mlperf-automations/script/get-generic-python-lib/customize.py
        Detected version: 3.2.0
INFO:root:    * cm run script "get python3"
INFO:root:         ! load /home/arjun/CM/repos/local/cache/301a17a2c7b2405d/cm-cached-state.json
INFO:root:Path to Python: /home/arjun/gh_action/bin/python3
INFO:root:Python version: 3.12.3
INFO:root:       ! load /home/arjun/CM/repos/local/cache/55113e99701848a6/cm-cached-state.json
INFO:root:  * cm run script "get generic-python-lib _package.pyarrow"
INFO:root:    * cm run script "get python3"
INFO:root:         ! load /home/arjun/CM/repos/local/cache/301a17a2c7b2405d/cm-cached-state.json
INFO:root:Path to Python: /home/arjun/gh_action/bin/python3
INFO:root:Python version: 3.12.3
INFO:root:         ! cd /home/arjun/tmp
INFO:root:         ! call /home/arjun/CM/repos/gateoverflow@mlperf-automations/script/get-generic-python-lib/validate_cache.sh from tmp-run.sh
INFO:root:         ! call "detect_version" from /home/arjun/CM/repos/gateoverflow@mlperf-automations/script/get-generic-python-lib/customize.py
        Detected version: 18.1.0
INFO:root:    * cm run script "get python3"
INFO:root:         ! load /home/arjun/CM/repos/local/cache/301a17a2c7b2405d/cm-cached-state.json
INFO:root:Path to Python: /home/arjun/gh_action/bin/python3
INFO:root:Python version: 3.12.3
INFO:root:       ! load /home/arjun/CM/repos/local/cache/50ff969e56f447fc/cm-cached-state.json
INFO:root:  * cm run script "get generic-python-lib _pandas"
INFO:root:    * cm run script "get python3"
INFO:root:         ! load /home/arjun/CM/repos/local/cache/301a17a2c7b2405d/cm-cached-state.json
INFO:root:Path to Python: /home/arjun/gh_action/bin/python3
INFO:root:Python version: 3.12.3
INFO:root:         ! cd /home/arjun/tmp
INFO:root:         ! call /home/arjun/CM/repos/gateoverflow@mlperf-automations/script/get-generic-python-lib/validate_cache.sh from tmp-run.sh
INFO:root:         ! call "detect_version" from /home/arjun/CM/repos/gateoverflow@mlperf-automations/script/get-generic-python-lib/customize.py
        Detected version: 2.2.3
INFO:root:    * cm run script "get python3"
INFO:root:         ! load /home/arjun/CM/repos/local/cache/301a17a2c7b2405d/cm-cached-state.json
INFO:root:Path to Python: /home/arjun/gh_action/bin/python3
INFO:root:Python version: 3.12.3
INFO:root:       ! load /home/arjun/CM/repos/local/cache/cbd16ae062564423/cm-cached-state.json
/home/arjun/gh_action/bin/python3 '/home/arjun/CM/repos/local/cache/12e5ddb36c4e4ee0/inference/tools/submission/submission_checker.py' --input 'my_submissions' --submitter 'MLCommons'
INFO:root:       ! cd /home/arjun/tmp
INFO:root:       ! call /home/arjun/CM/repos/gateoverflow@mlperf-automations/script/run-mlperf-inference-submission-checker/run.sh from tmp-run.sh
/home/arjun/gh_action/bin/python3 '/home/arjun/CM/repos/local/cache/12e5ddb36c4e4ee0/inference/tools/submission/submission_checker.py' --input 'my_submissions' --submitter 'MLCommons'
[2025-01-07 21:50:42,630 submission_checker.py:2597 WARNING] closed/MLCommons/results/h13_u1_slim, field host_processor_url is unknown
[2025-01-07 21:50:42,634 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/results/h13_u1_slim/resnet50/multistream/accuracy/mlperf_log_detail.txt.
[2025-01-07 21:50:42,635 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/results/h13_u1_slim/resnet50/multistream/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,636 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/results/h13_u1_slim/resnet50/multistream/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,636 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/compliance/h13_u1_slim/resnet50/multistream/TEST01/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,636 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/compliance/h13_u1_slim/resnet50/multistream/TEST01/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,637 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/compliance/h13_u1_slim/resnet50/multistream/TEST04/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,637 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/compliance/h13_u1_slim/resnet50/multistream/TEST04/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,640 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/results/h13_u1_slim/resnet50/singlestream/accuracy/mlperf_log_detail.txt.
[2025-01-07 21:50:42,641 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/results/h13_u1_slim/resnet50/singlestream/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,641 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/results/h13_u1_slim/resnet50/singlestream/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,642 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/compliance/h13_u1_slim/resnet50/singlestream/TEST01/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,642 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/compliance/h13_u1_slim/resnet50/singlestream/TEST01/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,642 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/compliance/h13_u1_slim/resnet50/singlestream/TEST04/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,643 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/compliance/h13_u1_slim/resnet50/singlestream/TEST04/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,646 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/results/h13_u1_slim/resnet50/offline/accuracy/mlperf_log_detail.txt.
[2025-01-07 21:50:42,647 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/results/h13_u1_slim/resnet50/offline/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,647 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/results/h13_u1_slim/resnet50/offline/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,647 submission_checker.py:1439 INFO] Target latency: None, Latency: 650182629071, Scenario: Offline
[2025-01-07 21:50:42,649 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/compliance/h13_u1_slim/resnet50/offline/TEST01/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,650 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/compliance/h13_u1_slim/resnet50/offline/TEST01/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,650 submission_checker.py:1439 INFO] Target latency: None, Latency: 650181895367, Scenario: Offline
[2025-01-07 21:50:42,650 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/compliance/h13_u1_slim/resnet50/offline/TEST04/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,651 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/compliance/h13_u1_slim/resnet50/offline/TEST04/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,651 submission_checker.py:1439 INFO] Target latency: None, Latency: 650182917418, Scenario: Offline
[2025-01-07 21:50:42,654 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/results/R760xa_L40Sx4_TRT/resnet50/Server/accuracy/mlperf_log_detail.txt.
[2025-01-07 21:50:42,655 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/results/R760xa_L40Sx4_TRT/resnet50/Server/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,656 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/results/R760xa_L40Sx4_TRT/resnet50/Server/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,656 submission_checker.py:1420 INFO] Target latency: 15000000, Early Stopping Latency: 15000000, Scenario: Server
[2025-01-07 21:50:42,656 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/compliance/R760xa_L40Sx4_TRT/resnet50/Server/TEST01/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,657 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/compliance/R760xa_L40Sx4_TRT/resnet50/Server/TEST01/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,657 submission_checker.py:1420 INFO] Target latency: 15000000, Early Stopping Latency: 15000000, Scenario: Server
[2025-01-07 21:50:42,657 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/compliance/R760xa_L40Sx4_TRT/resnet50/Server/TEST04/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,657 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/compliance/R760xa_L40Sx4_TRT/resnet50/Server/TEST04/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,657 submission_checker.py:1420 INFO] Target latency: 15000000, Early Stopping Latency: 15000000, Scenario: Server
[2025-01-07 21:50:42,660 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/results/R760xa_L40Sx4_TRT/resnet50/Offline/accuracy/mlperf_log_detail.txt.
[2025-01-07 21:50:42,661 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/results/R760xa_L40Sx4_TRT/resnet50/Offline/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,661 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/results/R760xa_L40Sx4_TRT/resnet50/Offline/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,661 submission_checker.py:1439 INFO] Target latency: None, Latency: 994899614302, Scenario: Offline
[2025-01-07 21:50:42,662 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/compliance/R760xa_L40Sx4_TRT/resnet50/Offline/TEST01/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,662 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/compliance/R760xa_L40Sx4_TRT/resnet50/Offline/TEST01/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,662 submission_checker.py:1439 INFO] Target latency: None, Latency: 995475274544, Scenario: Offline
[2025-01-07 21:50:42,663 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/compliance/R760xa_L40Sx4_TRT/resnet50/Offline/TEST04/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,663 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/compliance/R760xa_L40Sx4_TRT/resnet50/Offline/TEST04/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,663 submission_checker.py:1439 INFO] Target latency: None, Latency: 1205591233193, Scenario: Offline
[2025-01-07 21:50:42,679 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/results/R760xa_L40Sx4_TRT/retinanet/Server/accuracy/mlperf_log_detail.txt.
[2025-01-07 21:50:42,682 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/results/R760xa_L40Sx4_TRT/retinanet/Server/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,682 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/results/R760xa_L40Sx4_TRT/retinanet/Server/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,682 submission_checker.py:1420 INFO] Target latency: 100000000, Early Stopping Latency: 100000000, Scenario: Server
[2025-01-07 21:50:42,683 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/compliance/R760xa_L40Sx4_TRT/retinanet/Server/TEST01/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,683 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/compliance/R760xa_L40Sx4_TRT/retinanet/Server/TEST01/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,683 submission_checker.py:1420 INFO] Target latency: 100000000, Early Stopping Latency: 100000000, Scenario: Server
[2025-01-07 21:50:42,697 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/results/R760xa_L40Sx4_TRT/retinanet/Offline/accuracy/mlperf_log_detail.txt.
[2025-01-07 21:50:42,699 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/results/R760xa_L40Sx4_TRT/retinanet/Offline/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,699 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/results/R760xa_L40Sx4_TRT/retinanet/Offline/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,699 submission_checker.py:1439 INFO] Target latency: None, Latency: 820218532577, Scenario: Offline
[2025-01-07 21:50:42,699 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/compliance/R760xa_L40Sx4_TRT/retinanet/Offline/TEST01/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,699 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/compliance/R760xa_L40Sx4_TRT/retinanet/Offline/TEST01/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,699 submission_checker.py:1439 INFO] Target latency: None, Latency: 820219447799, Scenario: Offline
[2025-01-07 21:50:42,702 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/results/XE9680_H200_SXM_141GBx8_TRT/resnet50/Server/accuracy/mlperf_log_detail.txt.
[2025-01-07 21:50:42,703 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/results/XE9680_H200_SXM_141GBx8_TRT/resnet50/Server/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,704 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/results/XE9680_H200_SXM_141GBx8_TRT/resnet50/Server/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,704 submission_checker.py:1420 INFO] Target latency: 15000000, Early Stopping Latency: 15000000, Scenario: Server
[2025-01-07 21:50:42,704 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/compliance/XE9680_H200_SXM_141GBx8_TRT/resnet50/Server/TEST01/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,705 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/compliance/XE9680_H200_SXM_141GBx8_TRT/resnet50/Server/TEST01/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,705 submission_checker.py:1420 INFO] Target latency: 15000000, Early Stopping Latency: 15000000, Scenario: Server
[2025-01-07 21:50:42,705 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/compliance/XE9680_H200_SXM_141GBx8_TRT/resnet50/Server/TEST04/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,705 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/compliance/XE9680_H200_SXM_141GBx8_TRT/resnet50/Server/TEST04/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,705 submission_checker.py:1420 INFO] Target latency: 15000000, Early Stopping Latency: 15000000, Scenario: Server
[2025-01-07 21:50:42,709 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/results/XE9680_H200_SXM_141GBx8_TRT/resnet50/Offline/accuracy/mlperf_log_detail.txt.
[2025-01-07 21:50:42,710 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/results/XE9680_H200_SXM_141GBx8_TRT/resnet50/Offline/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,710 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/results/XE9680_H200_SXM_141GBx8_TRT/resnet50/Offline/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,710 submission_checker.py:1439 INFO] Target latency: None, Latency: 612424167382, Scenario: Offline
[2025-01-07 21:50:42,711 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/compliance/XE9680_H200_SXM_141GBx8_TRT/resnet50/Offline/TEST01/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,711 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/compliance/XE9680_H200_SXM_141GBx8_TRT/resnet50/Offline/TEST01/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,711 submission_checker.py:1439 INFO] Target latency: None, Latency: 612110080396, Scenario: Offline
[2025-01-07 21:50:42,712 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/compliance/XE9680_H200_SXM_141GBx8_TRT/resnet50/Offline/TEST04/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,712 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/compliance/XE9680_H200_SXM_141GBx8_TRT/resnet50/Offline/TEST04/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,712 submission_checker.py:1439 INFO] Target latency: None, Latency: 1174322867511, Scenario: Offline
[2025-01-07 21:50:42,725 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/results/XE9680_H200_SXM_141GBx8_TRT/retinanet/Server/accuracy/mlperf_log_detail.txt.
[2025-01-07 21:50:42,727 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/results/XE9680_H200_SXM_141GBx8_TRT/retinanet/Server/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,727 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/results/XE9680_H200_SXM_141GBx8_TRT/retinanet/Server/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,727 submission_checker.py:1420 INFO] Target latency: 100000000, Early Stopping Latency: 100000000, Scenario: Server
[2025-01-07 21:50:42,727 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/compliance/XE9680_H200_SXM_141GBx8_TRT/retinanet/Server/TEST01/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,728 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/compliance/XE9680_H200_SXM_141GBx8_TRT/retinanet/Server/TEST01/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,728 submission_checker.py:1420 INFO] Target latency: 100000000, Early Stopping Latency: 100000000, Scenario: Server
[2025-01-07 21:50:42,742 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/results/XE9680_H200_SXM_141GBx8_TRT/retinanet/Offline/accuracy/mlperf_log_detail.txt.
[2025-01-07 21:50:42,744 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/results/XE9680_H200_SXM_141GBx8_TRT/retinanet/Offline/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,744 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/results/XE9680_H200_SXM_141GBx8_TRT/retinanet/Offline/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,744 submission_checker.py:1439 INFO] Target latency: None, Latency: 646544882332, Scenario: Offline
[2025-01-07 21:50:42,745 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/compliance/XE9680_H200_SXM_141GBx8_TRT/retinanet/Offline/TEST01/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,745 log_parser.py:59 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/compliance/XE9680_H200_SXM_141GBx8_TRT/retinanet/Offline/TEST01/performance/run_1/mlperf_log_detail.txt.
[2025-01-07 21:50:42,745 submission_checker.py:1439 INFO] Target latency: None, Latency: 649661405172, Scenario: Offline
[2025-01-07 21:50:42,745 submission_checker.py:3071 INFO] ---
[2025-01-07 21:50:42,745 submission_checker.py:3075 INFO] Results closed/MLCommons/results/R760xa_L40Sx4_TRT/resnet50/Offline 180613.0
[2025-01-07 21:50:42,745 submission_checker.py:3075 INFO] Results closed/MLCommons/results/R760xa_L40Sx4_TRT/resnet50/Server 181231.0
[2025-01-07 21:50:42,745 submission_checker.py:3075 INFO] Results closed/MLCommons/results/R760xa_L40Sx4_TRT/retinanet/Offline 3345.71
[2025-01-07 21:50:42,745 submission_checker.py:3075 INFO] Results closed/MLCommons/results/R760xa_L40Sx4_TRT/retinanet/Server 3152.43
[2025-01-07 21:50:42,745 submission_checker.py:3075 INFO] Results closed/MLCommons/results/XE9680_H200_SXM_141GBx8_TRT/resnet50/Offline 768235.0
[2025-01-07 21:50:42,745 submission_checker.py:3075 INFO] Results closed/MLCommons/results/XE9680_H200_SXM_141GBx8_TRT/resnet50/Server 630226.0
[2025-01-07 21:50:42,745 submission_checker.py:3075 INFO] Results closed/MLCommons/results/XE9680_H200_SXM_141GBx8_TRT/retinanet/Offline 14760.1
[2025-01-07 21:50:42,745 submission_checker.py:3075 INFO] Results closed/MLCommons/results/XE9680_H200_SXM_141GBx8_TRT/retinanet/Server 13603.8
[2025-01-07 21:50:42,745 submission_checker.py:3075 INFO] Results closed/MLCommons/results/h13_u1_slim/resnet50/multistream 0.285447
[2025-01-07 21:50:42,745 submission_checker.py:3075 INFO] Results closed/MLCommons/results/h13_u1_slim/resnet50/offline 56277.1
[2025-01-07 21:50:42,745 submission_checker.py:3075 INFO] Results closed/MLCommons/results/h13_u1_slim/resnet50/singlestream 0.121029
[2025-01-07 21:50:42,745 submission_checker.py:3077 INFO] ---
[2025-01-07 21:50:42,745 submission_checker.py:3166 INFO] ---
[2025-01-07 21:50:42,745 submission_checker.py:3167 INFO] Results=11, NoResults=0, Power Results=0
[2025-01-07 21:50:42,745 submission_checker.py:3174 INFO] ---
[2025-01-07 21:50:42,745 submission_checker.py:3175 INFO] Closed Results=11, Closed Power Results=0

[2025-01-07 21:50:42,745 submission_checker.py:3180 INFO] Open Results=0, Open Power Results=0

[2025-01-07 21:50:42,745 submission_checker.py:3185 INFO] Network Results=0, Network Power Results=0

[2025-01-07 21:50:42,745 submission_checker.py:3190 INFO] ---
[2025-01-07 21:50:42,745 submission_checker.py:3192 INFO] Systems=3, Power Systems=0
[2025-01-07 21:50:42,745 submission_checker.py:3196 INFO] Closed Systems=3, Closed Power Systems=0
[2025-01-07 21:50:42,745 submission_checker.py:3201 INFO] Open Systems=0, Open Power Systems=0
[2025-01-07 21:50:42,745 submission_checker.py:3206 INFO] Network Systems=0, Network Power Systems=0
[2025-01-07 21:50:42,745 submission_checker.py:3211 INFO] ---
[2025-01-07 21:50:42,746 submission_checker.py:3216 INFO] SUMMARY: submission looks OK
/home/arjun/gh_action/bin/python3 '/home/arjun/CM/repos/local/cache/12e5ddb36c4e4ee0/inference/tools/submission/generate_final_report.py' --input summary.csv
=========================================================
Searching for summary.csv ...
Converting to json ...

                                                                          0   ...                                                 10
Organization                                                       MLCommons  ...                                          MLCommons
Availability                                                       available  ...                                          available
Division                                                              closed  ...                                             closed
SystemType                                                              edge  ...                                         datacenter
SystemName                   Supermicro SuperServer H13 (1x speedAI240 Slim)  ...  Dell PowerEdge XE9680 (8x H200-SXM-141GB, Tens...
Platform                                                         h13_u1_slim  ...                        XE9680_H200_SXM_141GBx8_TRT
Model                                                               resnet50  ...                                          retinanet
MlperfModel                                                           resnet  ...                                          retinanet
Scenario                                                         MultiStream  ...                                            Offline
Result                                                              0.285447  ...                                            14760.1
Accuracy                                                         acc: 75.912  ...                                        mAP: 37.329
number_of_nodes                                                            1  ...                                                  1
host_processor_model_name                    AMD EPYC 9124 16-Core Processor  ...                     Intel(R) Xeon(R) Platinum 8470
host_processors_per_node                                                   1  ...                                                  2
host_processor_core_count                                                 16  ...                                                 52
accelerator_model_name                            UntetherAI speedAI240 Slim  ...                              NVIDIA H200-SXM-141GB
accelerators_per_node                                                      1  ...                                                  8
Location                   closed/MLCommons/results/h13_u1_slim/resnet50/...  ...  closed/MLCommons/results/XE9680_H200_SXM_141GB...
framework                                  UntetherAI imAIgine SDK v24.07.19  ...                         TensorRT 10.2.0, CUDA 12.4
operating_system           Ubuntu 22.04.4 LTS (Linux kernel 6.5.0-44-gene...  ...                                     Ubuntu 22.04.3
notes                      SKU: sai240L-F-A-ES. Powered by the KRAI X and...  ...                                      H200 TGP 700W
compliance                                                                 1  ...                                                  1
errors                                                                     0  ...                                                  0
version                                                                 v5.1  ...                                               v5.1
inferred                                                                   0  ...                                                  0
has_power                                                              False  ...                                              False
Units                                                           Latency (ms)  ...                                          Samples/s
weight_data_types                                                     float8  ...                                               int8

[28 rows x 11 columns]

=========================================================
INFO:root:       ! call "postprocess" from /home/arjun/CM/repos/gateoverflow@mlperf-automations/script/run-mlperf-inference-submission-checker/customize.py
```
</details>

The example submissions used for testing were selectively obtained from:
- [MLPerf Inference v5.1 Results](https://github.com/mlcommons/inference_results_v5.1)
These submissions were used to demonstrate the submission generation process and validate the functionality of the mlc-scripts for `submission generation` and `submission checker`. 
Special thanks to **[MLCommons](https://github.com/mlcommons)** and the **submitting organizations** for making their results publicly available.
