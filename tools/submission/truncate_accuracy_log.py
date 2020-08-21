"""
Tool to truncate the mlperf_log_accuracy.json
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import hashlib
import logging
import os
import re
import sys
import shutil


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")

MAX_ACCURACY_LOG_SIZE = 10 * 1024
VIEWABLE_SIZE = 4096


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="submission directory")
    parser.add_argument("--submitter", required=True, help="filter to submitter")
    parser.add_argument("--backup", required=True, help="directory to store the original accuacy log")
    args = parser.parse_args()
    return args


def list_dir(*path):
    path = os.path.join(*path)
    return [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]


def list_files(*path):
    path = os.path.join(*path)
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


def split_path(m):
    return m.replace("\\", "/").split("/")


def get_hash(fname):
    """Return hash for file."""
    m = hashlib.sha256()
    with open(fname, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            m.update(byte_block)
    return m.hexdigest()


def truncate_file(src_name, dst_name):
    """Truncate file to 4K from start and 4K from end."""
    size = os.stat(src_name).st_size
    with open(src_name, "r") as src:
        start = src.read(VIEWABLE_SIZE)
        src.seek(size - VIEWABLE_SIZE, 0)
        end = src.read(VIEWABLE_SIZE)
    with open(dst_name, "w") as dst:
        dst.write(start)
        dst.write("\n\n...\n\n")
        dst.write(end)


def truncate_results_dir(filter_submitter, backup):
    """Walk result dir and 
       write a hash of mlperf_log_accuracy.json to accuracy.txt
       copy mlperf_log_accuracy.json to a backup location
       truncate mlperf_log_accuracy.
    """
    for division in list_dir("."):
        if division not in ["closed", "open"]:
            continue

        for submitter in list_dir(division):
            if filter_submitter and submitter != filter_submitter:
                continue
            results_path = os.path.join(division, submitter, "results")
            if not os.path.exists(results_path):
                log.error("no submission in %s", results_path)
                continue

            for system_desc in list_dir(results_path):
                for model in list_dir(results_path, system_desc):
                    for scenario in list_dir(results_path, system_desc, model):
                        name = os.path.join(results_path, system_desc, model, scenario)

                        hash_val = None
                        acc_path = os.path.join(name, "accuracy")
                        acc_log = os.path.join(acc_path, "mlperf_log_accuracy.json")
                        acc_txt = os.path.join(acc_path, "accuracy.txt")
                        if not os.path.exists(acc_log):
                            log.error("%s missing", acc_log)
                            continue
                        if not os.path.exists(acc_txt):
                            log.error("%s missing, generate to continue", acc_txt)
                            continue
                        with open(acc_txt, "r") as f:
                            for line in f:
                                m = re.match(r"^hash=([\w\d]+)$", line)
                                if m:
                                    hash_val = m.group(1)
                                    break
                        size = os.stat(acc_log).st_size
                        if hash_val and size < MAX_ACCURACY_LOG_SIZE:
                            log.info("%s already has hash and size seems truncated", acc_path)
                            continue

                        backup_dir = os.path.join(backup, name, "accuracy")
                        os.makedirs(backup_dir, exist_ok=True)
                        dst = os.path.join(backup, name, "mlperf_log_accuracy.json")
                        if os.path.exists(dst):
                            log.error("not processing %s because %s already exist", acc_log, dst)
                            continue

                        # get to work
                        shutil.copy(acc_log, dst)
                        hash_val = get_hash(acc_log)
                        with open(acc_txt, "a") as f:
                            f.write("hash={}\n".format(hash_val))
                        truncate_file(dst, acc_log)
                        log.info("%s truncated", acc_log)


def main():
    args = get_args()

    os.chdir(args.input)
    # truncate results directory
    truncate_results_dir(args.submitter, args.backup)
    return 0


if __name__ == "__main__":
    sys.exit(main())
