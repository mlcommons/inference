#!/bin/bash

pip install -r requirements.txt

git clone https://huggingface.co/datasets/livecodebench/code_generation_lite data/lcb
python3 data/fetch_all.py --output_path data/accuracy_eval_raw.pkl --lcb_folder data/lcb
python3 harmonize_inputs.py --data-file data/accuracy_eval_raw.pkl --output-file data/accuracy_eval_tokenized.pkl --reasoning-effort high --num-processes 32
