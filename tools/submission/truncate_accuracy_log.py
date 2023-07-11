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

HELP_TEXT = """
You can run this tool in 2 ways:

1. pick an existing submission directory and create a brand new submission tree with the trucated
    mlperf_log_accuracy.json files. The original submission directory is not modified.

    python tools/submission/truncate_accuracy_log.py --input ORIGINAL_SUBMISSION_DIRECTORY --submitter MY_ORG \\
        --output NEW_SUBMISSION_DIRECTORY

2. pick a existing submission directory and a backup location for files that are going to be modified.
    The tool will copy files that are modified into the backup directory and than modify the existing
    submission directory.

    python tools/submission/truncate_accuracy_log.py --input ROOT_OF_SUBMISSION_DIRECTORY --submitter MY_ORG \\
        --backup MY_SUPER_SAFE_STORAGE 
"""

def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser(description="Truncate mlperf_log_accuracy.json files.",
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=HELP_TEXT)
    parser.add_argument("--input", required=True, help="orignal submission directory")
    parser.add_argument("--output", help="new submission directory")
    parser.add_argument("--submitter", required=True, help="filter to submitter")
    parser.add_argument("--backup", help="directory to store the original accuacy log")

    args = parser.parse_args()
    if not args.output and not args.backup:
        parser.print_help()
        sys.exit(1)

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


def truncate_file(fname):
    """Truncate file to 4K from start and 4K from end."""
    size = os.stat(fname).st_size
    if size < VIEWABLE_SIZE:
       return
    with open(fname, "r") as src:
        start = src.read(VIEWABLE_SIZE)
        src.seek(size - VIEWABLE_SIZE, 0)
        end = src.read(VIEWABLE_SIZE)
    with open(fname, "w") as dst:
        dst.write(start)
        dst.write("\n\n...\n\n")
        dst.write(end)


def copy_submission_dir(src, dst, filter_submitter):
    for division in list_dir(src):
        if division not in ["closed", "open", "network"]:
            continue
        for submitter in list_dir(os.path.join(src, division)):
            if filter_submitter and submitter != filter_submitter:
                continue
            shutil.copytree(os.path.join(src, division, submitter),
                            os.path.join(dst, division, submitter))


def truncate_results_dir(filter_submitter, backup):
    """Walk result dir and 
       write a hash of mlperf_log_accuracy.json to accuracy.txt
       copy mlperf_log_accuracy.json to a backup location
       truncate mlperf_log_accuracy.
    """
    for division in list_dir("."):
        # we are looking at ./$division, ie ./closed
        if division not in ["closed", "open", "network"]:
            continue

        for submitter in list_dir(division):
            # we are looking at ./$division/$submitter, ie ./closed/mlperf_org
            if filter_submitter and submitter != filter_submitter:
                continue

            # process results
            for directory in ["results", "compliance"]:

                log_path = os.path.join(division, submitter, directory)
                if not os.path.exists(log_path):
                    log.error("no submission in %s", log_path)
                    continue

                for system_desc in list_dir(log_path):
                    for model in list_dir(log_path, system_desc):
                        for scenario in list_dir(log_path, system_desc, model):
                            for test in list_dir(log_path, system_desc, model, scenario):

                                name = os.path.join(log_path, system_desc, model, scenario)
                                if directory == "compliance":
                                    name = os.path.join(log_path, system_desc, model, scenario, test)

                                hash_val = None
                                acc_path = os.path.join(name, "accuracy")
                                acc_log = os.path.join(acc_path, "mlperf_log_accuracy.json")
                                acc_txt = os.path.join(acc_path, "accuracy.txt")

                                # only TEST01 has an accuracy log
                                if directory == "compliance" and test != "TEST01":
                                    continue
                                if not os.path.exists(acc_log):
                                    log.error("%s missing", acc_log)
                                    continue
                                if not os.path.exists(acc_txt) and directory == "compliance":
                                    # compliance test directory will not have an accuracy.txt file by default
                                    log.info("no accuracy.txt in compliance directory %s", acc_path)
                                else:
                                    if not os.path.exists(acc_txt):
                                        log.error("%s missing, generate to continue", acc_txt)
                                        continue
                                    with open(acc_txt, "r", encoding="utf-8") as f:
                                        for line in f:
                                            m = re.match(r"^hash=([\w\d]+)$", line)
                                            if m:
                                                hash_val = m.group(1)
                                                break
                                size = os.stat(acc_log).st_size
                                if hash_val and size < MAX_ACCURACY_LOG_SIZE:
                                    log.info("%s already has hash and size seems truncated", acc_path)
                                    continue

                                if backup:
                                    backup_dir = os.path.join(backup, name, "accuracy")
                                    os.makedirs(backup_dir, exist_ok=True)
                                    dst = os.path.join(backup, name, "accuracy", "mlperf_log_accuracy.json")
                                    if os.path.exists(dst):
                                        log.error("not processing %s because %s already exist", acc_log, dst)
                                        continue
                                    shutil.copy(acc_log, dst)

                                # get to work
                                hash_val = get_hash(acc_log)
                                with open(acc_txt, "a", encoding="utf-8") as f:
                                    f.write("hash={0}\n".format(hash_val))
                                truncate_file(acc_log)
                                log.info("%s truncated", acc_log)

                                # No need to iterate on compliance test subdirectories in the results folder
                                if directory == "results":
                                    break


def main():
    args = get_args()

    src_dir = args.input
    if args.output:
        if os.path.exists(args.output):
            print("output directory already exists")
            sys.exit(1)
        os.makedirs(args.output)
        copy_submission_dir(args.input, args.output, args.submitter)
        src_dir = args.output

    os.chdir(src_dir)
    # truncate results directory
    truncate_results_dir(args.submitter, args.backup)

    backup_location = args.output or args.backup
    log.info("Make sure you keep a backup of %s in case mlperf wants to see the original accuracy logs", backup_location)

    return 0


if __name__ == "__main__":
    sys.exit(main())
