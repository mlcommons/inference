# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
import json
import logging
import os
import re
import sys

# pylint: disable=missing-docstring

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(filename)s:%(lineno)d %(levelname)s] %(message)s",
)


class MLPerfLog:
    def __init__(self, log_path, strict=True):
        """
        Helper class to parse the detail logs.
        log_path: path to the detail log.
        strict: whether to ignore lines with :::MLLOG prefix but with invalid JSON format.
        """
        self.marker = ":::MLLOG"
        self.logger = logging.getLogger("MLPerfLog")
        self.messages = []
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip()
                if line.find(self.marker) == 0:
                    try:
                        self.messages.append(
                            json.loads(line[len(self.marker):]))
                    except BaseException:
                        if strict:
                            raise RuntimeError(
                                "Encountered invalid line: {:}".format(line)
                            )
                        else:
                            self.logger.warning(
                                "Skipping invalid line: {:}".format(line)
                            )
        self.keys = set()
        for message in self.messages:
            self.keys.add(message["key"])
        self.logger.info(
            "Sucessfully loaded MLPerf log from {:}.".format(log_path))

    def __getitem__(self, key):
        """
        Get the value of the message with the specific key. If a key appears multiple times, the first one is used.
        """
        if key not in self.keys:
            return None
        results = []
        for message in self.messages:
            if message["key"] == key:
                results.append(message)
        if len(results) != 1:
            self.logger.warning(
                "There are multiple messages with key {:} in the log. Emprically choosing the first one.".format(
                    key
                )
            )
        return results[0]["value"]

    def get(self, key):
        """
        Get all the messages with specific key in the log.
        """
        results = []
        if key in self.keys:
            for message in self.messages:
                if message["key"] == key:
                    results.append(message)
        return results

    def get_messages(self):
        """
        Get all the messages in the log.
        """
        return self.messages

    def get_keys(self):
        """
        Get all the keys in the log.
        """
        return self.keys

    def get_dict(self):
        """
        Get a dict representing the log. If a key appears multiple times, the first one is used.
        """
        result = {}
        for message in self.messages:
            if message["key"] not in result:
                result[message["key"]] = message["value"]
            else:
                self.logger.warning(
                    "There are multiple messages with key {:} in the log. Emprically choosing the first one.".format(
                        key
                    )
                )

    def dump(self, output_path):
        """
        Dump the entire log as a json file.
        """
        with open(log_path, "w") as f:
            json.dump(self.messages, f, indent=4)

    def num_messages(self):
        """Get number of messages (including errors and warnings) in the log."""
        return len(self.messages)

    def num_errors(self):
        """Get number of errors in the log."""
        count = 0
        for message in self.messages:
            if message["metadata"]["is_error"]:
                count += 1
        return count

    def num_warnings(self):
        """Get number of warning in the log."""
        count = 0
        for message in self.messages:
            if message["metadata"]["is_warning"]:
                count += 1
        return count

    def has_error(self):
        """Check if the log contains any errors."""
        return self.num_errors() != 0

    def has_warning(self):
        """Check if the log contains any warnings."""
        return self.num_warnings() != 0

    def get_errors(self):
        """
        Get all the error messages in the log.
        """
        results = []
        for message in self.messages:
            if message["metadata"]["is_error"]:
                results.append(message)
        return results

    def get_warnings(self):
        """
        Get all the warning messages in the log.
        """
        results = []
        for message in self.messages:
            if message["metadata"]["is_warning"]:
                results.append(message)
        return results


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="path to the detail log")
    parser.add_argument(
        "--ignore_invalid_lines",
        action="store_true",
        help="whether to stop if there are lines with invalid formats",
    )
    args = parser.parse_args()
    return args


def main():
    """
    Inspect a detailed log.
    """
    args = get_args()
    mlperf_log = MLPerfLog(args.input, strict=not args.ignore_invalid_lines)
    logger = logging.getLogger("main")
    logger.info("Details of the log:")
    logger.info("- Number of messages: {:d}".format(mlperf_log.num_messages()))
    logger.info("- Number of errors: {:d}".format(mlperf_log.num_errors()))
    logger.info("- Number of warnings: {:d}".format(mlperf_log.num_warnings()))
    logger.info("- Contents:")
    messages = mlperf_log.get_messages()
    for message in messages:
        logger.info('"{:}": {:}'.format(message["key"], message["value"]))
    logger.info("Done!")


if __name__ == "__main__":
    sys.exit(main())
