---
hide:
  - toc
---

# Installation
We use MLCommons CM Automation framework to run MLPerf inference benchmarks.

CM needs `git`, `python3-pip` and `python3-venv` installed on your system. If any of these are absent, please follow the [official CM installation page](https://docs.mlcommons.org/ck/install) to install them. Once the dependencies are installed, do the following

## Activate a Virtual ENV for CM
This step is not mandatory as CM can use separate virtual environment for MLPerf inference. But the latest `pip` install requires this or else will need the `--break-system-packages` flag while installing `cm4mlops`.

```bash
python3 -m venv cm
source cm/bin/activate
```

## Install CM and pulls any needed repositories
=== "Use the default fork of CM MLOps repository"
    ```bash
     pip install cm4mlops
    ```

=== "Use custom fork/branch of the CM MLOps repository"
    ```bash
     pip install cmind && cm init --quiet --repo=mlcommons@cm4mlops --branch=mlperf-inference
    ```
    Here, `repo` is in the format `githubUsername@githubRepo`.

Now, you are ready to use the `cm` commands to run MLPerf inference as given in the [benchmarks](../index.md) page
