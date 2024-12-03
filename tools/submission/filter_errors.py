"""
Tool to remove manually verified ERRORs from the log file in the v0.7 submission.
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import sys

ignore = [
    "ERROR:main:invalid division in input dir .vscode",
    "ERROR : Loadgen built with uncommitted changes!",
    "is missing code_dir",
    "is missing A100-PCIex4_TRT_AEP*.json",
    "is missing T4x16_TRT_AEP*.json",
    "is missing macbook_pro_2019*.json",
    "is missing n2-highcpu-16*.json",
    "measurement_dir has issues",
    "Required minimum Query Count not met by user config, Expected=270336, Found=90112",
    "no compliance dir for closed/dividiti/results/firefly-armnn-v20.08-neon/resnet50/offline",
    "no compliance dir for closed/dividiti/results/firefly-armnn-v20.08-opencl/resnet50/offline",
    "no compliance dir for closed/dividiti/results/firefly-tflite-v2.2.0-ruy/resnet50/offline",
    "compliance dir closed/dividiti/compliance/firefly-tflite-v2.2.0-ruy/resnet50/singlestream has issues",
    "no compliance dir for closed/dividiti/results/rpi4-armnn-v20.08-neon/resnet50/offline",
    "no compliance dir for closed/dividiti/results/rpi4-tflite-v2.2.0-ruy/resnet50/offline",
    "no compliance dir for closed/dividiti/results/rpi4coral-armnn-v20.08-neon/resnet50/offline",
    "compliance dir closed/dividiti/compliance/rpi4coral-armnn-v20.08-neon/resnet50/singlestream has issues",
    "no compliance dir for closed/dividiti/results/rpi4coral-tflite-v2.2.0-ruy/resnet50/offline",
    "compliance dir closed/dividiti/compliance/rpi4coral-tflite-v2.2.0-ruy/resnet50/singlestream has issues",
    "no compliance dir for closed/dividiti/results/rpi4coral-tflite-v2.2.0-ruy/ssd-mobilenet-non-quantized/offline",
    "no compliance dir for closed/dividiti/results/xavier-armnn-v20.08-neon/resnet50/offline",
    "compliance dir closed/dividiti/compliance/xavier-armnn-v20.08-neon/resnet50/singlestream has issues",
    "compliance dir closed/dividiti/compliance/xavier-tensorrt-v6.0/resnet50/offline has issues",
    "compliance dir closed/dividiti/compliance/xavier-tensorrt-v6.0/ssd-mobilenet/offline has issues",
    "compliance dir closed/dividiti/compliance/xavier-tensorrt-v6.0/ssd-mobilenet/singlestream has issues",
    "no compliance dir for closed/dividiti/results/xavier-tflite-v2.2.0-ruy/resnet50/offline",
    "no compliance dir for closed/dividiti/results/xavier-tflite-v2.3.0-ruy/resnet50/offline",
]


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="orignal submission directory")
    parser.add_argument("--output", help="new submission directory")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    ignored = 0
    with open(args.input, "r", encoding="utf-8") as inp:
        with open(args.output, "w", encoding="utf-8") as outp:
            for line in inp:
                keep = True
                if "ERROR" in line:
                    for e in ignore:
                        if e in line.replace("\\", "/"):
                            keep = False
                            ignored += 1
                            break
                if keep:
                    outp.write(line)
    print("ignored", ignored)
    return 0


if __name__ == "__main__":
    sys.exit(main())
