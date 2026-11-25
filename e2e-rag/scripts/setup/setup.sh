#!/bin/bash

cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
pip install -r requirements.txt
