# Myrtle Deep Speech

A PyTorch implementation of [DeepSpeech](https://arxiv.org/abs/1412.5567) and
[DeepSpeech2](https://arxiv.org/abs/1512.02595).

This repository is intended as an evolving baseline for other implementations
to compare their training performance against.

Current roadmap:
1. ~Pre-trained weights for both networks and full performance statistics.~
    - See v0.1 release: https://github.com/MyrtleSoftware/deepspeech/releases/tag/v0.1
1. Mixed-precision training.

## Running

Build the Docker image:

```
make build
```

Run the Docker container (here using
[nvidia-docker](https://github.com/NVIDIA/nvidia-docker)), ensuring to publish
the port of the JupyterLab session to the host:

```
sudo docker run --runtime=nvidia --shm-size 512M -p 9999:9999 deepspeech
```

The JupyterLab session can be accessed via `localhost:9999`.

This Python package will accessible in the running Docker container and is
accessible through either the command line interface:

```
deepspeech --help
```

or as a Python package:

```python
import deepspeech
```

## Example Training:

`deepspeech --help` will print the configurable parameters (batch size,
learning rate, log location, number of epochs...) - it aims to have reasonably
sensible defaults. A Deep Speech training run can be started by the following
command, adding flags as necessary:

```
deepspeech ds1
```

By default the experimental data and logs are output to
`/tmp/experiments/year_month_date-hour_minute_second_microsecond`.

## Dataset

The package contains code to download and use the [LibriSpeech ASR
corpus](http://www.openslr.org/12/).

## WER

The word error rate (WER) is computed using the formula that is widely used in
many open-source speech-to-text systems (Kaldi, PaddlePaddle, Mozilla
DeepSpeech). In pseudocode, where `N` is the number of validation or test
samples:

```
sum_edits = sum([edit_distance(target, predict)
                 for target, predict in zip(targets, predictions)])
sum_lens = sum([len(target) for target in targets])
WER = (1.0/N) * (sum_edits / sum_lens)
```

This reduces the impact on the WER of errors in short sentences. Toy example:

| Target                         | Prediction                    | Edit Distance | Label Length |
|--------------------------------|-------------------------------|---------------|--------------|
| lectures                       | lectured                      | 1             | 1            |
| i'm afraid he said             | i am afraid he said           | 2             | 4            |
| nice to see you mister meeking | nice to see your mister makin | 2             | 6            |

The mean WER of each sample considered individually is:

```
>>> (1.0/3) * ((1.0/1) + (2.0/4) + (2.0/6))
0.611111111111111
```

Compared to the pseudocode version given above:

```
>>> (1.0/3) * ((1.0 + 2 + 2) / (1.0 + 4 + 6))
0.1515151515151515
```

## Maintainer

Please contact `mlperf at myrtle dot ai`.
