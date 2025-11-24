# Building the LoadGen {#ReadmeBuild}

## Prerequisites

    sudo apt-get install libglib2.0-dev python-pip python3-pip

## Quick Start
### Installation - Python
If you need to clone the repo (e.g., because you are a MLPerf Inference developer), you
can build and install the `mlperf-loadgen` package via:

    git clone --recurse-submodules https://github.com/mlcommons/inference.git mlperf_inference
    cd mlperf_inference/loadgen
    pip install -e .  # Install in the editable mode because you are a developer.

If you don't need to clone the repo (e.g., you just want to install `mlperf-loadgen`
from the latest commit of the `master` branch):

    pip install git+https://github.com/mlcommons/inference.git#subdirectory=loadgen

This will fetch the loadgen source, then build and install the loadgen as a python module.

Alternatively, we provide wheels for several python versions and operating system that can be installed using pip directly.

    pip install mlperf-loadgen

**NOTE:** Take into account that we only update the published wheels after an official release, they may not include the latest changes.

### Testing your Installation
The following command will run a simple end-to-end demo:

    python mlperf_inference/loadgen/demos/py_demo_single_stream.py

A summary of the test results can be found in the *"mlperf_log_summary.txt"* logfile.

For a timeline visualization of what happened during the test, open the *"mlperf_log_trace.json"* file in Chrome:
* Type “chrome://tracing” in the address bar, then drag-n-drop the json.
* This may be useful for SUT performance tuning and understanding + debugging the loadgen.

### Installation - C++
To build the loadgen as a C++ library, rather than a python module:

    git clone https://github.com/mlcommons/inference.git mlperf_inference
    cd mlperf_inference
    mkdir loadgen/build/ && cd loadgen/build/
    cmake .. && cmake --build .
    cp libmlperf_loadgen.a ..

## Quick start: Loadgen Over the Network

Refer to [LON demo](demos/lon/README.md) for a basic example.
