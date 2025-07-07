*Check [CM MLPerf docs](https://docs.mlcommons.org/inference) for more details.*

## Host platform

* OS version: Linux-5.10.0-34-cloud-amd64-x86_64-with-glibc2.35
* CPU version: x86_64
* Python version: 3.10.12 (main, May 27 2025, 17:12:29) [GCC 11.4.0]
* MLC version: unknown

## CM Run Command

See [CM installation guide](https://docs.mlcommons.org/inference/install/).

```bash
pip install -U mlcflow

mlc rm cache -f

mlc pull repo anandhu-eng@mlperf-automations --checkout=b994f10dc088cbd2d826163027c864facd6a6cfa


```
*Note that if you want to use the [latest automation recipes](https://docs.mlcommons.org/inference) for MLPerf,
 you should simply reload anandhu-eng@mlperf-automations without checkout and clean MLC cache as follows:*

```bash
mlc rm repo anandhu-eng@mlperf-automations
mlc pull repo anandhu-eng@mlperf-automations
mlc rm cache -f

```