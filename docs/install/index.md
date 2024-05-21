# Installation
We use MLCommons CM Automation framework to run MLPerf inference benchmarks.

## CM Install

We have successfully tested CM on

* Ubuntu 18.x, 20.x, 22.x , 23.x, 
* RedHat 8, RedHat 9, CentOS 8
* macOS
* Wndows 10, Windows 11
 
=== "Ubuntu"
    ### Ubuntu, Debian


    ```bash
       sudo apt update && sudo apt upgrade
       sudo apt install python3 python3-pip python3-venv git wget curl
    ```

    **Note that you must set up virtual env on Ubuntu 23+ before using any Python project:**
    ```bash
       python3 -m venv cm
       source cm/bin/activate
    ```

    You can now install CM via PIP:

    ```bash
       python3 -m pip install cmind
    ```

    You might need to do the following command to update the `PATH` to include the BIN paths from pip installs

    ```bash
       source $HOME/.profile
    ```

    You can check that CM is available by checking the `cm` command


=== "Red Hat"
    ### Red Hat

    ```bash
       sudo dnf update
       sudo dnf install python3 python-pip git wget curl
       python3 -m pip install cmind --user
    ```

=== "macOS"
    ### macOS

    *Note that CM currently does not work with Python installed from the Apple Store.
     Please install Python via brew as described below.*

    If `brew` package manager is not installed, please install it as follows (see details [here](https://brew.sh/)):
    ```bash
       /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    ```

    Don't forget to add brew to PATH environment as described in the end of the installation output.

    Then install python, pip, git and wget:

    ```bash
       brew install python3 git wget curl
       python3 -m pip install cmind
    ```

=== "Windows"

    ### Windows
    * Configure Windows 10+ to [support long paths](https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=registry#enable-long-paths-in-windows-10-version-1607-and-later) from command line as admin:
      <small>
      ```bash
         reg add "HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f
      ```
      </small>
    * Download and install Git from [git-for-windows.github.io](https://git-for-windows.github.io).
      * Configure Git to accept long file names: `git config --system core.longpaths true`
    * Download and install Python 3+ from [www.python.org/downloads/windows](https://www.python.org/downloads/windows).
      * Don't forget to select option to add Python binaries to PATH environment!
      * Configure Windows to accept long fie names during Python installation!

    * Install CM via PIP:

    ```bash
       python -m pip install cmind
    ```

    *Note that we [have reports](https://github.com/mlcommons/ck/issues/844) 
     that CM does not work when Python was first installed from the Microsoft Store.
     If CM fails to run, you can find a fix [here](https://stackoverflow.com/questions/57485491/python-python3-executes-in-command-prompt-but-does-not-run-correctly)*.

Please visit the [official CM installation page](https://github.com/mlcommons/ck/blob/master/docs/installation.md) for more details

## Download the CM MLOps Repository

```bash
   cm pull repo gateoverflow@cm4mlops
```

Now, you are ready to use the `cm` commands to run MLPerf inference as given in the [benchmarks](../benchmarks/index.md) page
