---
hide:
  - toc
---

# Installation
We use MLCommons MLC Automation framework to run MLPerf inference benchmarks.

MLC needs `git`, `python3-pip` and `python3-venv` installed on your system. Once the dependencies are installed, do the following

## Activate a Virtual ENV for MLCFlow
This step is not mandatory as MLC can use separate virtual environment for MLPerf inference. But the latest `pip` install requires this or else will need the `--break-system-packages` flag while installing `mlc-scripts`.

```bash
python3 -m venv mlc
source mlc/bin/activate
```

## Install MLC and pulls any needed repositories
=== "Use the default fork of MLC-Scripts repository"
    ```bash
     pip install mlc-scripts
    ```

=== "Use custom fork/branch of the MLC-Scripts repository"
    ```bash
     pip install mlcflow && mlc pull repo --url=mlcommons@mlperf-automations --branch=dev
    ```
    Here, `repo` is in the format `githubUsername@githubRepo` or you can give any URL

Now, you are ready to use the `mlcr` commands to run MLPerf inference as given in the [benchmarks](../index.md) page
