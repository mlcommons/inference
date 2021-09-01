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
from pathlib import Path

input_path1 = Path(sys.argv[1])
input_path2 = Path(sys.argv[2])

f1 = open(input_path1,)
f2 = open(input_path2,)
  
input1 = json.load(f1)
input2 = json.load(f2)

idxs = []
for elem in input2:
  idxs.append(elem["qsl_idx"])

idxs.sort()

output = []
for idx in idxs:
  for elem in input1:
    if elem["qsl_idx"] == idx:
      output.append(elem)

outname = input_path1.stem + "_baseline" + input_path1.suffix

print("Created a baseline accuracy file:", outname)

with open(outname, 'w', encoding='utf-8') as f:
  json.dump(output, f, ensure_ascii=False, indent=2, separators=(',', ':'))
