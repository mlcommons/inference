# MLPerf Inference - Object Classification - ShuffleNet using [MLModelScope](MLModelScope.org) [![MLModelScope](https://img.shields.io/badge/mlmodelscope-mlperf-green.svg)](https://mlmodelscope.org) [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# MLPerf Inference - Image Classification - ShuffleNet

## Installation

Install docker following the instructions at [docs.mlmodelscope.org](https://docs.mlmodelscope.org/installation/source/external_services/) using

```bash
curl -fsSL get.docker.com -o get-docker.sh | sudo sh
sudo usermod -aG docker $USER
```

and start the tracing, registry, and database server

```
docker run -d -e COLLECTOR_ZIPKIN_HTTP_PORT=9411 -p5775:5775/udp -p6831:6831/udp -p6832:6832/udp \
  -p5778:5778 -p16686:16686 -p14268:14268 -p9411:9411 jaegertracing/all-in-one:latest
docker run -p 8500:8500 -p 8600:8600 -d consul
docker run -p 27017:27017 --restart always -d mongo:3.0
```

## Evaluating

1. [Benchmark ShuffleNet via MXNet](mxnet/README.md)
2. [Benchmark ShuffleNet via Caffe2](caffe2/README.md)
