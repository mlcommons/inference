#!/bin/bash

cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
pip install -r requirements.txt
apt-get update
apt-get install -y --no-install-recommends \
    wkhtmltopdf