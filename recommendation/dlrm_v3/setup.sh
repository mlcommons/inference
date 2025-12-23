#!/usr/bin/env bash

conda create --name dlrmv3 python=3.13
conda activate dlrmv3
pip install -r requirements.txt
git_dir=$(git rev-parse --show-toplevel)
pip install $git_dir/loadgen
