### Clone the Repo
```
git clone -b submission-generation-examples https://github.com/mlcommons/inference.git submission-generation-examples --depth 1
```
### Install cm4mlops
```
pip install cm4mlops
```

### Generate the submission tree
```
cm run script --tags=generate,mlperf,inference,submission \
--results_dir=submission-examples/closed \
--run_checker=yes  \
--submission_dir=my_submissions  \
--quiet \
--submitter=MLCommons \
--division=closed
--clean
```

Expected Output:
<details>

```
INFO:root:* cm run script "generate mlperf inference submission"
INFO:root:  * cm run script "get python3"
INFO:root:       ! load /home/arjun/CM/repos/local/cache/548fbb5fbd2247a3/cm-cached-state.json
INFO:root:Path to Python: /home/arjun/cm/bin/python3
INFO:root:Python version: 3.12.3
INFO:root:  * cm run script "mlcommons inference src"
INFO:root:       ! load /home/arjun/CM/repos/local/cache/329bf723391d457a/cm-cached-state.json
INFO:root:  * cm run script "get sut system-description"
INFO:root:    * cm run script "detect os"
INFO:root:           ! cd /home/arjun
INFO:root:           ! call /home/arjun/CM/repos/gateoverflow@cm4mlops/script/detect-os/run.sh from tmp-run.sh
INFO:root:           ! call "postprocess" from /home/arjun/CM/repos/gateoverflow@cm4mlops/script/detect-os/customize.py
INFO:root:    * cm run script "detect cpu"
INFO:root:      * cm run script "detect os"
INFO:root:             ! cd /home/arjun
INFO:root:             ! call /home/arjun/CM/repos/gateoverflow@cm4mlops/script/detect-os/run.sh from tmp-run.sh
INFO:root:             ! call "postprocess" from /home/arjun/CM/repos/gateoverflow@cm4mlops/script/detect-os/customize.py
INFO:root:           ! cd /home/arjun
INFO:root:           ! call /home/arjun/CM/repos/gateoverflow@cm4mlops/script/detect-cpu/run.sh from tmp-run.sh
INFO:root:           ! call "postprocess" from /home/arjun/CM/repos/gateoverflow@cm4mlops/script/detect-cpu/customize.py
INFO:root:    * cm run script "get python3"
INFO:root:         ! load /home/arjun/CM/repos/local/cache/548fbb5fbd2247a3/cm-cached-state.json
INFO:root:Path to Python: /home/arjun/cm/bin/python3
INFO:root:Python version: 3.12.3
INFO:root:    * cm run script "get compiler"
INFO:root:         ! load /home/arjun/CM/repos/local/cache/ad72b0b0eafe4731/cm-cached-state.json
INFO:root:    * cm run script "get generic-python-lib _package.dmiparser"
INFO:root:         ! load /home/arjun/CM/repos/local/cache/884af9f84a4844b1/cm-cached-state.json
INFO:root:    * cm run script "get cache dir _name.mlperf-inference-sut-descriptions"
INFO:root:         ! load /home/arjun/CM/repos/local/cache/635118be6ba047c9/cm-cached-state.json
Generating SUT description file for phoenix_Amd_Am5
INFO:root:         ! call "postprocess" from /home/arjun/CM/repos/gateoverflow@cm4mlops/script/get-mlperf-inference-sut-description/customize.py
INFO:root:  * cm run script "install pip-package for-cmind-python _package.tabulate"
INFO:root:       ! load /home/arjun/CM/repos/local/cache/4cbf8fe4ffa14112/cm-cached-state.json
INFO:root:  * cm run script "get mlperf inference utils"
INFO:root:    * cm run script "get mlperf inference src"
INFO:root:         ! load /home/arjun/CM/repos/local/cache/329bf723391d457a/cm-cached-state.json
INFO:root:         ! call "postprocess" from /home/arjun/CM/repos/gateoverflow@cm4mlops/script/get-mlperf-inference-utils/customize.py
INFO:root:       ! call "postprocess" from /home/arjun/CM/repos/gateoverflow@cm4mlops/script/generate-mlperf-inference-submission/customize.py
=================================================
Cleaning plaa ...
=================================================
* MLPerf inference submission dir: plaa
* MLPerf inference results dir: submission-tests/case-3
* MLPerf inference division: closed
* MLPerf inference submitter: MLCommons
sut info completely filled from submission-tests/case-3/H200-SXM-141GBx8_TRT_MaxQ/cm-sut-info.json!
* MLPerf inference model: retinanet
 * verify_performance.txt
 * mlperf_log_accuracy.json
 * mlperf_log_summary.txt
 * mlperf_log_detail.txt
 * accuracy.txt
 * mlperf_log_accuracy.json
 * mlperf_log_summary.txt
 * mlperf_log_detail.txt
 * verify_performance.txt
 * verify_accuracy.txt
INFO:MLPerfLog:Sucessfully loaded MLPerf log from submission-tests/case-3/H200-SXM-141GBx8_TRT_MaxQ/retinanet/Offline/performance/run_1/mlperf_log_detail.txt.
INFO:MLPerfLog:Sucessfully loaded MLPerf log from submission-tests/case-3/H200-SXM-141GBx8_TRT_MaxQ/retinanet/Offline/performance/run_1/mlperf_log_detail.txt.
 * verify_performance.txt
 * mlperf_log_accuracy.json
 * mlperf_log_summary.txt
 * mlperf_log_detail.txt
 * accuracy.txt
 * mlperf_log_accuracy.json
 * mlperf_log_summary.txt
 * mlperf_log_detail.txt
 * verify_performance.txt
 * verify_accuracy.txt
INFO:MLPerfLog:Sucessfully loaded MLPerf log from submission-tests/case-3/H200-SXM-141GBx8_TRT_MaxQ/retinanet/Server/performance/run_1/mlperf_log_detail.txt.
INFO:MLPerfLog:Sucessfully loaded MLPerf log from submission-tests/case-3/H200-SXM-141GBx8_TRT_MaxQ/retinanet/Server/performance/run_1/mlperf_log_detail.txt.
+-----------+----------+----------+------------+-----------------+---------------------------------+--------+--------+
|   Model   | Scenario | Accuracy | Throughput | Latency (in ms) | Power Efficiency (in samples/J) | TEST01 | TEST05 |
+-----------+----------+----------+------------+-----------------+---------------------------------+--------+--------+
| retinanet | Offline  |  37.268  |  10802.5   |        -        |                                 | passed | passed |
| retinanet |  Server  |  37.32   |  9603.47   |        -        |                                 | passed | passed |
+-----------+----------+----------+------------+-----------------+---------------------------------+--------+--------+
INFO:root:* cm run script "accuracy truncate mlc"
INFO:root:  * cm run script "get python3"
INFO:root:       ! load /home/arjun/CM/repos/local/cache/548fbb5fbd2247a3/cm-cached-state.json
INFO:root:Path to Python: /home/arjun/cm/bin/python3
INFO:root:Python version: 3.12.3
INFO:root:  * cm run script "get mlcommons inference src"
INFO:root:       ! load /home/arjun/CM/repos/local/cache/329bf723391d457a/cm-cached-state.json
INFO:root:       ! cd /home/arjun
INFO:root:       ! call /home/arjun/CM/repos/gateoverflow@cm4mlops/script/truncate-mlperf-inference-accuracy-log/run.sh from tmp-run.sh
python3 '/home/arjun/CM/repos/local/cache/ab43ae81b215428f/inference/tools/submission/truncate_accuracy_log.py' --input 'plaa' --submitter 'MLCommons' --backup 'plaa_logs'
INFO:main:closed/MLCommons/results/H200-SXM-141GBx8-nvidia-cuda-tensorrt-default/retinanet/Offline/accuracy already has hash and size seems truncated
INFO:main:closed/MLCommons/results/H200-SXM-141GBx8-nvidia-cuda-tensorrt-default/retinanet/Offline/accuracy already has hash and size seems truncated
INFO:main:closed/MLCommons/results/H200-SXM-141GBx8-nvidia-cuda-tensorrt-default/retinanet/Server/accuracy already has hash and size seems truncated
INFO:main:closed/MLCommons/results/H200-SXM-141GBx8-nvidia-cuda-tensorrt-default/retinanet/Server/accuracy already has hash and size seems truncated
INFO:main:closed/MLCommons/compliance/H200-SXM-141GBx8-nvidia-cuda-tensorrt-default/retinanet/Offline/TEST01/accuracy already has hash and size seems truncated
INFO:main:closed/MLCommons/compliance/H200-SXM-141GBx8-nvidia-cuda-tensorrt-default/retinanet/Server/TEST01/accuracy already has hash and size seems truncated
INFO:main:Make sure you keep a backup of plaa_logs in case mlperf wants to see the original accuracy logs
INFO:root:* cm run script "submission inference checker mlc"
INFO:root:  * cm run script "get python3"
INFO:root:       ! load /home/arjun/CM/repos/local/cache/548fbb5fbd2247a3/cm-cached-state.json
INFO:root:Path to Python: /home/arjun/cm/bin/python3
INFO:root:Python version: 3.12.3
INFO:root:  * cm run script "get mlcommons inference src"
INFO:root:       ! load /home/arjun/CM/repos/local/cache/329bf723391d457a/cm-cached-state.json
INFO:root:  * cm run script "get generic-python-lib _xlsxwriter"
INFO:root:       ! load /home/arjun/CM/repos/local/cache/bfa737dc1acb4e15/cm-cached-state.json
INFO:root:  * cm run script "get generic-python-lib _package.pyarrow"
INFO:root:       ! load /home/arjun/CM/repos/local/cache/04da54023ce24d97/cm-cached-state.json
INFO:root:  * cm run script "get generic-python-lib _pandas"
INFO:root:       ! load /home/arjun/CM/repos/local/cache/21ef9332af354f8f/cm-cached-state.json
INFO:root:  * cm run script "get generic-python-lib _numpy"
INFO:root:       ! load /home/arjun/CM/repos/local/cache/f81454e7084d4370/cm-cached-state.json
/home/arjun/cm/bin/python3 '/home/arjun/CM/repos/local/cache/ab43ae81b215428f/inference/tools/submission/submission_checker.py' --input 'plaa' --submitter 'MLCommons'
INFO:root:       ! cd /home/arjun
INFO:root:       ! call /home/arjun/CM/repos/gateoverflow@cm4mlops/script/run-mlperf-inference-submission-checker/run.sh from tmp-run.sh
/home/arjun/cm/bin/python3 '/home/arjun/CM/repos/local/cache/ab43ae81b215428f/inference/tools/submission/submission_checker.py' --input 'plaa' --submitter 'MLCommons'
[2024-11-12 14:19:40,371 log_parser.py:50 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/results/H200-SXM-141GBx8-nvidia-cuda-tensorrt-default/retinanet/Offline/accuracy/mlperf_log_detail.txt.
[2024-11-12 14:19:40,373 log_parser.py:50 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/results/H200-SXM-141GBx8-nvidia-cuda-tensorrt-default/retinanet/Offline/performance/run_1/mlperf_log_detail.txt.
[2024-11-12 14:19:40,373 log_parser.py:50 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/results/H200-SXM-141GBx8-nvidia-cuda-tensorrt-default/retinanet/Offline/performance/run_1/mlperf_log_detail.txt.
[2024-11-12 14:19:40,373 submission_checker.py:1168 INFO] Target latency: None, Latency: 604877953388, Scenario: Offline
[2024-11-12 14:19:40,373 log_parser.py:50 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/compliance/H200-SXM-141GBx8-nvidia-cuda-tensorrt-default/retinanet/Offline/TEST01/performance/run_1/mlperf_log_detail.txt.
[2024-11-12 14:19:40,373 log_parser.py:50 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/compliance/H200-SXM-141GBx8-nvidia-cuda-tensorrt-default/retinanet/Offline/TEST01/performance/run_1/mlperf_log_detail.txt.
[2024-11-12 14:19:40,373 submission_checker.py:1168 INFO] Target latency: None, Latency: 605071581356, Scenario: Offline
[2024-11-12 14:19:40,374 log_parser.py:50 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/compliance/H200-SXM-141GBx8-nvidia-cuda-tensorrt-default/retinanet/Offline/TEST05/performance/run_1/mlperf_log_detail.txt.
[2024-11-12 14:19:40,374 log_parser.py:50 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/compliance/H200-SXM-141GBx8-nvidia-cuda-tensorrt-default/retinanet/Offline/TEST05/performance/run_1/mlperf_log_detail.txt.
[2024-11-12 14:19:40,374 submission_checker.py:1168 INFO] Target latency: None, Latency: 606583600879, Scenario: Offline
[2024-11-12 14:19:40,387 log_parser.py:50 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/results/H200-SXM-141GBx8-nvidia-cuda-tensorrt-default/retinanet/Server/accuracy/mlperf_log_detail.txt.
[2024-11-12 14:19:40,388 log_parser.py:50 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/results/H200-SXM-141GBx8-nvidia-cuda-tensorrt-default/retinanet/Server/performance/run_1/mlperf_log_detail.txt.
[2024-11-12 14:19:40,389 log_parser.py:50 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/results/H200-SXM-141GBx8-nvidia-cuda-tensorrt-default/retinanet/Server/performance/run_1/mlperf_log_detail.txt.
[2024-11-12 14:19:40,389 submission_checker.py:1150 INFO] Target latency: 100000000, Early Stopping Latency: 100000000, Scenario: Server
[2024-11-12 14:19:40,389 log_parser.py:50 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/compliance/H200-SXM-141GBx8-nvidia-cuda-tensorrt-default/retinanet/Server/TEST01/performance/run_1/mlperf_log_detail.txt.
[2024-11-12 14:19:40,389 log_parser.py:50 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/compliance/H200-SXM-141GBx8-nvidia-cuda-tensorrt-default/retinanet/Server/TEST01/performance/run_1/mlperf_log_detail.txt.
[2024-11-12 14:19:40,389 submission_checker.py:1150 INFO] Target latency: 100000000, Early Stopping Latency: 100000000, Scenario: Server
[2024-11-12 14:19:40,390 log_parser.py:50 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/compliance/H200-SXM-141GBx8-nvidia-cuda-tensorrt-default/retinanet/Server/TEST05/performance/run_1/mlperf_log_detail.txt.
[2024-11-12 14:19:40,390 log_parser.py:50 INFO] Sucessfully loaded MLPerf log from closed/MLCommons/compliance/H200-SXM-141GBx8-nvidia-cuda-tensorrt-default/retinanet/Server/TEST05/performance/run_1/mlperf_log_detail.txt.
[2024-11-12 14:19:40,390 submission_checker.py:1150 INFO] Target latency: 100000000, Early Stopping Latency: 100000000, Scenario: Server
[2024-11-12 14:19:40,390 submission_checker.py:2680 INFO] ---
[2024-11-12 14:19:40,390 submission_checker.py:2684 INFO] Results closed/MLCommons/results/H200-SXM-141GBx8-nvidia-cuda-tensorrt-default/retinanet/Offline 10802.5
[2024-11-12 14:19:40,390 submission_checker.py:2684 INFO] Results closed/MLCommons/results/H200-SXM-141GBx8-nvidia-cuda-tensorrt-default/retinanet/Server 9602.95
[2024-11-12 14:19:40,390 submission_checker.py:2686 INFO] ---
[2024-11-12 14:19:40,390 submission_checker.py:2771 INFO] ---
[2024-11-12 14:19:40,390 submission_checker.py:2772 INFO] Results=2, NoResults=0, Power Results=0
[2024-11-12 14:19:40,390 submission_checker.py:2779 INFO] ---
[2024-11-12 14:19:40,390 submission_checker.py:2780 INFO] Closed Results=2, Closed Power Results=0

[2024-11-12 14:19:40,390 submission_checker.py:2785 INFO] Open Results=0, Open Power Results=0

[2024-11-12 14:19:40,390 submission_checker.py:2790 INFO] Network Results=0, Network Power Results=0

[2024-11-12 14:19:40,390 submission_checker.py:2795 INFO] ---
[2024-11-12 14:19:40,390 submission_checker.py:2797 INFO] Systems=1, Power Systems=0
[2024-11-12 14:19:40,390 submission_checker.py:2798 INFO] Closed Systems=1, Closed Power Systems=0
[2024-11-12 14:19:40,390 submission_checker.py:2803 INFO] Open Systems=0, Open Power Systems=0
[2024-11-12 14:19:40,390 submission_checker.py:2808 INFO] Network Systems=0, Network Power Systems=0
[2024-11-12 14:19:40,390 submission_checker.py:2813 INFO] ---
[2024-11-12 14:19:40,390 submission_checker.py:2818 INFO] SUMMARY: submission looks OK
/home/arjun/cm/bin/python3 '/home/arjun/CM/repos/local/cache/ab43ae81b215428f/inference/tools/submission/generate_final_report.py' --input summary.csv
=========================================================
Searching for summary.csv ...
Converting to json ...

                                                                           0                                                  1
Organization                                                       MLCommons                                          MLCommons
Availability                                                       available                                          available
Division                                                              closed                                             closed
SystemType                                                        datacenter                                         datacenter
SystemName                   NVIDIA H200 (8x H200-SXM-141GB, MaxQ, TensorRT)    NVIDIA H200 (8x H200-SXM-141GB, MaxQ, TensorRT)
Platform                       H200-SXM-141GBx8-nvidia-cuda-tensorrt-default      H200-SXM-141GBx8-nvidia-cuda-tensorrt-default
Model                                                              retinanet                                          retinanet
MlperfModel                                                        retinanet                                          retinanet
Scenario                                                             Offline                                             Server
Result                                                               10802.5                                            9602.95
Accuracy                                                              37.268                                              37.32
number_of_nodes                                                            1                                                  1
host_processor_model_name                    Intel(R) Xeon(R) Platinum 8480C                    Intel(R) Xeon(R) Platinum 8480C
host_processors_per_node                                                   2                                                  2
host_processor_core_count                                                 56                                                 56
accelerator_model_name                                 NVIDIA H200-SXM-141GB                              NVIDIA H200-SXM-141GB
accelerators_per_node                                                      8                                                  8
Location                   closed/MLCommons/results/H200-SXM-141GBx8-nvid...  closed/MLCommons/results/H200-SXM-141GBx8-nvid...
framework                                         TensorRT 10.2.0, CUDA 12.4                         TensorRT 10.2.0, CUDA 12.4
operating_system                                              Ubuntu 22.04.4                                     Ubuntu 22.04.4
notes                      H200 TGP 700W. Automated by MLCommons CM v3.4.1.   H200 TGP 700W. Automated by MLCommons CM v3.4.1.
compliance                                                                 1                                                  1
errors                                                                     0                                                  0
version                                                                 v4.1                                               v4.1
inferred                                                                   0                                                  0
has_power                                                              False                                              False
Units                                                              Samples/s                                          Queries/s
```
</details>

Description of test cases:
<details>
  
**Case-1**: model_maping.json in SUT folder

**Case-2**: model_mapping.json in individual folder

**Case-3**: model_mapping.json not present but model name is matching with the official one in submission checker

**Case-4**: model_mapping.json is not present but model name is mapped to official model name in submission checker. Example: resnet50 to resnet

**Case-5**: Case-1 to Case-4 is not satisfied. The gh action will be successfull if the submission generation fails.

**Case-6**: Case-2 but model_mapping.json is not present in any of the folders. The gh action will be successfull if the submission generation fails.

**Case-7**: sut_info.json is not completely filled but the SUT folder name is in required format(hardware_name-implementation-device-framework-run_config)

**Case-8**: system_meta.json absent in results folder

**Closed**:

**Closed-no-compliance**:

**closed-power**:

**closed-failed-power-logs**:

</details>
