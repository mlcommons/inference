#! /usr/bin/env python3

#
# Copyright 2021 The MLPerf Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import json
import sys
import os
from pathlib import Path

# Usage -
# python3 verify_accuracy_baseline.py <compliance test root dir> <benchmark name> <target accuracy>

root_path = Path(sys.argv[1])
model     = sys.argv[2].lower()
accuracy  = sys.argv[3]


baseline_file   = os.path.join(root_path,"accuracy","baseline_accuracy.txt")
compliance_file = os.path.join(root_path,"accuracy","compliance_accuracy.txt")
verify_file = os.path.join(root_path,"verify_accuracy.txt")

baseline   = open(baseline_file, "r").read()
compliance = open(compliance_file, "r").read()

vf = open(verify_file, "a")

if accuracy != '99' and accuracy != '99.9':
    raise Exception('Incorrect target accuracy. Must be 99 or 99.9')

accuracy = float(accuracy)

if model == 'bert':

    start = baseline.find('"f1": ')+6
    end = baseline.find('}', start)
    baseline_val = baseline[start: end]

    start = compliance.find('"f1": ')+6
    end = compliance.find('}', start)
    compliance_val = compliance[start: end]

    match = 100.0*(1-(float(baseline_val) - float(compliance_val))/float(baseline_val))

    vf.write("Verifying accuracy via baseline comparison...\n")
    vf.write(f"baseline accuracy: {baseline_val} (100.00%)\n")
    vf.write(f"compliance accuracy: {compliance_val} ({match:.2f}%)\n")
    vf.write(f"threshold accuracy: {accuracy}%\n")

    if match < accuracy:
        vf.write("TEST FAIL\n")
    else:
        vf.write("TEST PASS\n")

else:
  #TODO add other models as required.
  raise Exception('This model is not currently handled.')
