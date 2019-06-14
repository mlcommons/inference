
## Build docker
```
  docker build -t inference/language_modeling .
```

## Run docker
```
docker run -it --rm -v $(pwd):/mlperf inference/language_modeling python3 /mlperf/benchmark.py
```

## Run without docker
To install run (ideally in a new virtual environment):
```
pip3 install -r requirements.txt
```

To run:
```
python3 benchmark.py
```

For help:
```
python3 benchmark.py -h
```
