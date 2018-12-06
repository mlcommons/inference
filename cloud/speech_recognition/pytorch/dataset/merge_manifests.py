from __future__ import print_function

import argparse
import io
import os

import subprocess

from utils import update_progress

parser = argparse.ArgumentParser(description='Merges all manifest CSV files in specified folder.')
parser.add_argument('--merge_dir', default='manifests/', help='Path to all manifest files you want to merge')
parser.add_argument('--min_duration', default=-1, type=int,
                    help='Optionally prunes any samples shorter than the min duration (given in seconds, default off)')
parser.add_argument('--max_duration', default=-1, type=int,
                    help='Optionally prunes any samples longer than the max duration (given in seconds, default off)')
parser.add_argument('--output_path', default='merged_manifest.csv', help='Output path to merged manifest')

args = parser.parse_args()

files = []
for file_name in os.listdir(args.merge_dir):
    if file_name.endswith(".csv"):
        with open(os.path.join(args.merge_dir, file_name), 'r') as fh:
            files += fh.readlines()

prune_min = args.min_duration >= 0
prune_max = args.max_duration >= 0
if prune_min:
    print("Pruning files with minimum duration %d" % args.min_duration)
if prune_max:
    print("Pruning files with  maximum duration of %d" % args.max_duration)

new_files = []
size = len(files)
for x in range(size):
    file_path = files[x]
    file_path = file_path.split(',')[0]
    output = subprocess.check_output(
        ['soxi -D \"%s\"' % file_path.strip()],
        shell=True
    )
    duration = float(output)
    if prune_min or prune_max:
        if not ((prune_min and duration < args.min_duration) or
                (prune_max and duration > args.max_duration)):
            new_files.append((files[x], duration))
    else:
        new_files.append((files[x], duration))
    update_progress(x / float(size))

print("\nSorting files by length...")

new_files.sort(key=lambda element: element[1])

print("Saving new manifest...")

with io.FileIO(args.output_path, 'w') as f:
    for file_path in new_files:
        sample = file_path[0].strip() + '\n'
        f.write(sample.encode('utf-8'))
