# Building the LoadGen {#ReadmeBuild}

## Prerequisites

    sudo apt-get install libglib2.0-dev python-pip python3-pip
    pip2 install absl-py numpy
    pip3 install absl-py numpy

## Quick Start

    pip install absl-py numpy
    git clone --recurse-submodules https://github.com/mlcommons/inference.git mlperf_inference
    cd mlperf_inference/loadgen
    CFLAGS="-std=c++14 -O3" python setup.py bdist_wheel
    pip install --force-reinstall dist/mlperf_loadgen-0.5a0-cp36-cp36m-linux_x86_64.whl
    python demos/py_demo_single_stream.py

This will fetch the loadgen source, build and install the loadgen as a python module, and run a simple end-to-end demo. The exact *.whl filename may differ on your system, but there should only be one resulting whl file for you to use.

A summary of the test results can be found in the *"mlperf_log_summary.txt"* logfile.

For a timeline visualization of what happened during the test, open the *"mlperf_log_trace.json"* file in Chrome:
* Type “chrome://tracing” in the address bar, then drag-n-drop the json.
* This may be useful for SUT performance tuning and understanding + debugging the loadgen.

To build the loadgen as a C++ library, rather than a python module:

    git clone https://github.com/mlcommons/inference.git mlperf_inference
    cd mlperf_inference
    mkdir loadgen/build/ && cd loadgen/build/
    cmake .. && cmake --build .
    cp libmlperf_loadgen.a ..

## Loadgen Over the Network

In this mode, the client is a separate node running LoadGen, QSL, and QDL instances.
The SUT is another node which runs the server

Prerequisites:

* Install python packages:

```sh
pip install absl-py numpy wheel flask requests
```

Clone, as above:

```sh
git clone --recurse-submodules https://github.com/mlcommons/inference.git mlperf_inference
```

Build:

```sh
cd mlperf_inference/loadgen
CFLAGS="-std=c++14 -O3" python setup.py bdist_wheel
cd ..; pip install --force-reinstall loadgen/dist/`ls -r loadgen/dist/ | head -n1` ; cd -
```

Run demo (single machine):

Open a sut dummy server (run this at a separate terminal):

```sh
python demos/flask_app.py --port 8000
```

Start the test:

```sh
python demos/py_demo_server_lon.py --sut_server http://localhost:8000
```

The demo brings up a dummy SUT server, and one client that send queries simultaneously to the server.

Run the demo (over the network):
To run it over a network - run the flask app (i.e., the dummy SUT) over on a different machine, and replace `localhost` with its IP.

