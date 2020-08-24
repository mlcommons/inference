# Copyright 2019 The MLPerf Authors. All Rights Reserved.
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

import os
import sys
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('log')

    return parser.parse_args()

def main(args):

    print('Load log from {0}'.format(args.log))
    with open(args.log, 'r') as f:
        results = json.load(f)

    print('Processing log entries')
    rmap = {}
    truncated_results = []
    for j in results:
        idx = j['qsl_idx']
        if idx in rmap and rmap[idx] == j['data']:
            continue
        else:
            truncated_results.append(j)
            if idx not in rmap:
                rmap[idx] = j['data']
    print('original: {0} => truncated: {1}'.format(len(results), len(truncated_results)))
    
    print('Write truncated log to {0}.new'.format(args.log))
    with open(args.log+'.new', 'w') as f:
        json.dump(truncated_results, f, indent=4)
            
if __name__ == '__main__':
    main(parse_args())
