# MLPerf Inference - Image Classification - ShuffleNet using [MLModelScope](MLModelScope.org)

[![MLModelScope](https://img.shields.io/badge/mlmodelscope-mlperf-green.svg)](https://mlmodelscope.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

To maintain the SW stack and to guarantee isolation, evaluation occurs within docker containers. One can introspect the SW stack using exsiting docker tools that allows querying a image’s SW environment, its metadata, and perform diffs between container images.

Evaluations are traced by MLModelScope tracer and traces can be published to a database for later analysis.
MLModelScope comes with reporting commands built in to aggreate and analyze the evaluation output.

## Installation

### Docker

Install docker following the instructions at [docs.mlmodelscope.org](https://docs.mlmodelscope.org/installation/source/external_services/) using

```bash
curl -fsSL get.docker.com -o get-docker.sh | sudo sh
sudo usermod -aG docker $USER
```

### Helper Tools

Start the tracing and database server using docker by

```
docker run -d -e COLLECTOR_ZIPKIN_HTTP_PORT=9411 -p5775:5775/udp -p6831:6831/udp -p6832:6832/udp \
  -p5778:5778 -p16686:16686 -p14268:14268 -p9411:9411 jaegertracing/all-in-one:latest
docker run -p 27017:27017 --restart always -d mongo:3.0
```

### MlModelScope Configration

You must have a carml config file called `.carml_config.yml` under your home directory. Please refer to [carml_config.yml](https://docs.mlmodelscope.org/installation/configuration/).

## Evaluation

1. [Benchmark ShuffleNet via Caffe2](caffe2/README.md)
