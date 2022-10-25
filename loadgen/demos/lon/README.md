# Demo

## Loadgen Over the Network

### Overview

In this mode, the client is a separate node running LoadGen, QSL, and QDL instances.
The SUT is another node which runs the server.

The demo brings up a dummy SUT server and one client that send queries to the server.

### Setup

Install python packages:

```sh
pip install absl-py numpy wheel flask requests
```

Clone:

```sh
git clone --recurse-submodules https://github.com/mlcommons/inference.git mlperf_inference
```

Build:

```sh
cd mlperf_inference/loadgen
CFLAGS="-std=c++14 -O3" python setup.py bdist_wheel
cd ..; pip install --force-reinstall loadgen/dist/`ls -r loadgen/dist/ | head -n1` ; cd -
```

### Run the demo (single machine)

Start a dummy SUT server (run this at a separate terminal):

```sh
python demos/lon/lon_flask_app_sut.py --port 8000
```

Start the test - client:

```sh
python demos/lon/py_demo_server_lon.py --sut_server http://localhost:8000
```

### Run the demo (over the network)

To run over a network - simply run the flask app (i.e., the dummy SUT) over on a different machine. \
Then, when running the client, replace `localhost` with the correct IP.
