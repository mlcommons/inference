# Installation
We use MLCommons CM Automation framework to run MLPerf inference benchmarks.

CM needs `git`, `python3-pip` and `python3-venv` installed on your system. If any of these are absent, please follow the [official CM installation page](https://docs.mlcommons.org/ck/install) to install them. Once the dependencies are installed, do the following

## Activate a VENV for CM
```bash
   python3 -m venv cm
   source cm/bin/activate
```

## Install CM and pulls any needed repositories

```bash
   pip install cm4mlops
```


Now, you are ready to use the `cm` commands to run MLPerf inference as given in the [benchmarks](../benchmarks/index.md) page
