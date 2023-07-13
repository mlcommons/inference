"""
Tool to infer scenario results and cleanup submission tree
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import os
import sys
import shutil
import json

import submission_checker as checker


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")


HELP_TEXT = """
pick an existing submission directory and create a brand new submission tree with
    possible results being inferred from already measured ones. The original submission directory is not modified.

    cd tools/submission
    python3 preprocess_submission.py --input ORIGINAL_SUBMISSION_DIRECTORY --submitter MY_ORG \\
        --output NEW_SUBMISSION_DIRECTORY

"""

def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser(description="Infer scenario results",
                    formatter_class=argparse.RawDescriptionHelpFormatter, epilog=HELP_TEXT)
    parser.add_argument("--input", required=True, help="orignal submission directory")
    parser.add_argument("--output", help="new submission directory")
    parser.add_argument("--noinfer_low_accuracy_results",
        help="do not infer low accuracy results if a high accuracy result is present",
        default=False, action="store_true")
    parser.add_argument("--nodelete_empty_dirs",
        help="do not delete empty dirs in submission tree",
        default=False, action="store_true")
    parser.add_argument(
        "--version",
        default="v3.1",
        choices=list(checker.MODEL_CONFIG.keys()),
        help="mlperf version")
    parser.add_argument("--submitter", help="filter to submitter")
    parser.add_argument(
        "--extra-model-benchmark-map",
        help="File containing extra custom model mapping.\
        It is assumed to be inside the folder open/<submitter>",
        default="model_mapping.json")

    args = parser.parse_args()
    if not args.output:
        parser.print_help()
        sys.exit(1)

    return args


def list_dir(*path):
    """
    Filters only directories from a given path
    """
    path = os.path.join(*path)
    return next(os.walk(path))[1]

def delete_empty_dirs(src):
    """
    Deletes any empty directory in the src tree
    """
    if not os.path.isdir(src):
        return False

    if all ([delete_empty_dirs(os.path.join(src, file)) for file in os.listdir(src)]):
        log.info("Removing empty dir: (%s)", src)
        os.rmdir(src)
        return True

    return False

def copy_submission_dir(src, dst, filter_submitter):
    """
    Copies the submission tree to output directory for processing
    """
    for division in next(os.walk(src))[1]:
        if division not in ["closed", "open", "network"]:
            continue
        for submitter in next(os.walk(os.path.join(src, division)))[1]:
            if filter_submitter and submitter != filter_submitter:
                continue
            shutil.copytree(os.path.join(src, division, submitter),
                            os.path.join(dst, division, submitter))


def infer_scenario_results(filter_submitter, noinfer_low_accuracy_results, config):
    """Walk result dir and check for singlestream (SS) folders and \
               corresponding offline and multistream (MS) ones.
       If SS exists and offline and MS are not existing, \
               SS folder is copied to MS and offline folders.
       If SS and offline exists and MS is not existing, MS is inferred from SS.
       If SS and MS exists and offline is not existing, offline is inferred from MS.
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
            for directory in ["results", "measurements"] + \
                (["compliance"] if division == "closed" else []):

                log_path = os.path.join(division, submitter, directory)
                if not os.path.exists(log_path):
                    log.error("no submission in %s", log_path)
                    continue

                for system_desc in list_dir(log_path):
                    system_id_json = os.path.join(division, submitter, "systems",
                                      system_desc + ".json")
                    if not os.path.exists(system_id_json):
                        log.error("no system_desc for %s/%s/%s", division, submitter,
                            system_desc)
                        continue

                    with open(system_id_json) as system_info:
                        system_json = json.load(system_info)
                    system_type = system_json.get("system_type")
                    valid_system_types = ["datacenter", "edge", \
                            "datacenter,edge", "edge,datacenter"]
                    if system_type not in valid_system_types:
                        log.error("Division %s, submitter %s, "\
                                "system %s has invalid system type (%s)", \
                                division, submitter, system_id_json, system_type)
                    config.set_type(system_type)

                    for model in list_dir(log_path, system_desc):
                        extra_model_mapping = None
                        if division == "open":
                            model_mapping_path = (
                                f"{division}/{submitter}/{config.extra_model_benchmark_map}"
                            )
                            if os.path.exists(model_mapping_path):
                                with open(model_mapping_path) as fp:
                                    extra_model_mapping = json.load(fp)

                        mlperf_model = config.get_mlperf_model(model, extra_model_mapping)
                        if not mlperf_model:
                            log.error("Division %s, submitter %s, system %s has "\
                                    "invalid model (%s)", division, submitter, \
                                    system_id_json, model)
                            continue

                        if mlperf_model not in config.required:
                            log.error("Division %s, submitter %s, system %s has invalid "\
                                    "MLPerf model (%s) corresponding to given model (%s). "\
                                    "Valid ones for MLPerf inference version (%s) in (%s) "\
                                    "category are [%s]", division, submitter, system_id_json,\
                                    mlperf_model, model, config.version, system_type, \
                                    config.required.keys())
                            continue


                        required_scenarios = config.get_required(model)
                        all_scenarios = set(
                            list(required_scenarios) +
                            list(config.get_optional(mlperf_model)))

                        for scenario in list_dir(log_path, system_desc, model):

                            scenario_path = os.path.join(log_path, system_desc, model, scenario)

                            if scenario.lower() == "singlestream":
                                tobeinferredpaths = []
                                offline_scenario_path =  os.path.join(log_path, system_desc, \
                                        model, "offline")
                                multistream_scenario_path =  os.path.join(log_path, system_desc, \
                                        model, "multistream")
                                if not os.path.exists(multistream_scenario_path) and \
                                        not os.path.exists(offline_scenario_path):
                                    #infer both the scenarios from SS
                                    tobeinferredpaths = [ offline_scenario_path ]
                                    if "MultiStream" in all_scenarios:
                                        tobeinferredpaths.append(multistream_scenario_path)

                                    for tobeinferredpath in tobeinferredpaths:
                                        inferred_scenario = os.path.basename(tobeinferredpath)
                                        log.info("Division %s, submitter %s, system %s, " \
                                            "model %s: \
                                            inferring %s results from %s", \
                                            division, submitter, system_desc, model, \
                                            inferred_scenario, "singlestream")
                                        shutil.copytree(scenario_path, tobeinferredpath)

                                elif not os.path.exists(multistream_scenario_path) and \
                                        "MultiStream" in all_scenarios:
                                    #infer MS from SS
                                    for tobeinferredpath in [ multistream_scenario_path ]:
                                        log.info("Division %s, submitter %s, system %s, model %s: \
                                            inferring %s results from %s", division, submitter, \
                                            system_desc, model, "multistream", "singlestream")
                                        shutil.copytree(scenario_path, multistream_scenario_path)
                                elif not os.path.exists(offline_scenario_path):
                                    '''we have both MS and SS results. Inferring from MS is \
                                        expected to be better \
                                    '''
                                    pass

                            elif scenario.lower() == "multistream":
                                offline_scenario_path =  os.path.join(log_path, system_desc, \
                                        model, "offline")
                                '''Need to check if MS is indeed a measured result and not infeered.\
                                But if MS is indeed inferred from SS, offline scenario will also be \
                                inferred already by the inferring code above \
                                '''
                                for tobeinferredpath in [offline_scenario_path]:
                                    if not os.path.exists(tobeinferredpath):
                                        log.info("Division %s, submitter %s, system %s, model %s: \
                                                inferring %s results from %s", division, submitter,\
                                                system_desc, model, "offline", "multistream")
                                        shutil.copytree(scenario_path, tobeinferredpath)

                if not noinfer_low_accuracy_results:
                    for system_desc in list_dir(log_path):
                        for model in list_dir(log_path, system_desc):
                            if model.endswith("-99.9"):
                                low_accuracy_model =model[:-2]
                                if low_accuracy_model not in config.required:
                                    continue
                                high_accuracy_model_path = os.path.join(log_path, \
                                        system_desc, model)
                                low_accuracy_model_path = os.path.join(log_path, system_desc, \
                                        low_accuracy_model)
                                if not os.path.exists(low_accuracy_model_path):
                                    log.info("Division %s, submitter %s, system %s: \
                                            copying %s results to %s", division, submitter, \
                                            system_desc, model, low_accuracy_model)

                                    shutil.copytree(high_accuracy_model_path, \
                                            low_accuracy_model_path)



def main():
    """
    Tool to infer scenario results and cleanup submission tree
    """
    args = get_args()

    src_dir = args.input

    if os.path.exists(args.output):
        print("output directory already exists")
        sys.exit(1)
    os.makedirs(args.output)
    copy_submission_dir(args.input, args.output, args.submitter)
    src_dir = args.output

    config = checker.Config(
      args.version,
      args.extra_model_benchmark_map)

    if not args.nodelete_empty_dirs:
        delete_empty_dirs(os.path.join(src_dir))

    os.chdir(src_dir)

    infer_scenario_results(args.submitter, args.noinfer_low_accuracy_results, config)

    return 0


if __name__ == "__main__":
    sys.exit(main())
