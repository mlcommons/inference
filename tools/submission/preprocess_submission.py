"""
Tool to infer scenario results and cleanup submission tree
"""

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
    parser.add_argument(
        "--input",
        required=True,
        help="orignal submission directory")
    parser.add_argument("--output", help="new submission directory")
    parser.add_argument("--noinfer-low-accuracy-results",
                        help="do not infer low accuracy results if a high accuracy result is present",
                        default=False, action="store_true")
    parser.add_argument("--nodelete-empty-dirs",
                        help="do not delete empty dirs in submission tree",
                        default=False, action="store_true")
    parser.add_argument("--nomove-failed-to-open",
                        help="do not move failed results to open division",
                        default=False, action="store_true")
    parser.add_argument("--nodelete-failed",
                        help="do not delete failed results (submission checker will fail)",
                        default=False, action="store_true")

    parser.add_argument(
        "--version",
        default="v4.1",
        choices=list(checker.MODEL_CONFIG.keys()),
        help="mlperf version",
    )
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

    if all([delete_empty_dirs(os.path.join(src, file))
           for file in os.listdir(src)]):
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
            shutil.copytree(
                os.path.join(src, division, submitter),
                os.path.join(dst, division, submitter),
            )


def change_first_directory_to_open(path):
    # Split the path into components
    parts = path.split(os.path.sep)

    # Modify the first directory in the path
    for i, part in enumerate(parts):
        if part:  # Skip empty parts to handle initial slashes in absolute paths
            parts[i] = "open"
            break

    # Join the parts back into a modified path
    modified_path = os.path.join(*parts)
    return modified_path


def change_folder_name_in_path(path, old_folder_name, new_folder_name):
    # Split the path into components
    path_parts = path.split(os.path.sep)

    # Replace the old folder name with the new one
    path_parts = [new_folder_name if part ==
                  old_folder_name else part for part in path_parts]

    # Reassemble the path
    new_path = os.path.join(*path_parts)
    return new_path


def clean_model_dir(model_results_dir):
    model_measurements_dir = change_folder_name_in_path(
        model_results_dir, "results", "measurements")
    model_compliance_dir = change_folder_name_in_path(
        model_results_dir, "results", "compliance")

    print(f"rmtree {model_results_dir}")
    shutil.rmtree(model_results_dir)
    shutil.rmtree(model_measurements_dir)
    shutil.rmtree(model_compliance_dir)
    sut_results_dir = os.path.dirname(model_results_dir)
    if not os.listdir(sut_results_dir):
        # clean sut dir
        sut = os.path.basename(sut_results_dir)
        log.info(
            f"No benchmark results remaining for {sut}. rmtree {sut_results_dir}")
        shutil.rmtree(sut_results_dir)
        shutil.rmtree(os.path.dirname(model_measurements_dir))
        shutil.rmtree(os.path.dirname(model_compliance_dir))


def clean_invalid_results(args, log_path, config, system_desc, system_json,
                          model, mlperf_model, division, system_id_json, is_closed_or_network):
    # cleanup invalid results
    for scenario in list_dir(log_path, system_desc, model):
        scenario_fixed = checker.SCENARIO_MAPPING.get(
            scenario, scenario)
        scenario_path = os.path.join(log_path, system_desc, model, scenario)
        if not os.path.exists(
                scenario_path):  # can happen since scenario results may be moved to open division on failure
            continue
        acc_path = os.path.join(scenario_path, "accuracy")
        try:
            accuracy_is_valid, acc = checker.check_accuracy_dir(
                config,
                mlperf_model,
                acc_path,
                is_closed_or_network,
            )
        except Exception as e:
            log.warning(e)
            accuracy_is_valid = False
        perf_path = os.path.join(scenario_path, "performance", "run_1")
        try:
            perf_is_valid, r, is_inferred = checker.check_performance_dir(
                config,
                mlperf_model,
                perf_path,
                scenario_fixed,
                division,
                system_json,
            )
        except Exception as e:
            log.warning(e)
            perf_is_valid = False
        compliance_is_valid = False
        if perf_is_valid:
            power_path = os.path.join(scenario_path, "performance", "power")
            has_power = os.path.exists(power_path)
            if has_power:
                ranging_path = os.path.join(
                    scenario_path, "performance", "ranging")
                try:
                    ranging_r = get_performance_metric(
                        config,
                        mlperf_model,
                        ranging_path,
                        scenario_fixed,
                    )
                    (
                        power_is_valid,
                        power_metric,
                        power_efficiency,
                    ) = check_power_dir(
                        power_path,
                        ranging_path,
                        perf_path,
                        scenario_fixed,
                        ranging_r,
                        r,
                        config,
                    )
                except Exception as e:
                    power_is_valid = False
                if not power_is_valid:
                    log.warning(
                        f"Power result is invalid for {system_desc}: {model} {scenario} scenario in {division} division. Removing...")
                    shutil.rmtree(power_path)
                    shutil.rmtree(ranging_path)
                    shutil.rmtree(os.path.join(perf_path, "spl.txt"))

            compliance_is_valid = True
            if is_closed_or_network:
                compliance_dir = change_folder_name_in_path(
                    scenario_path, "results", "compliance")
                if not checker.check_compliance_dir(
                    compliance_dir,
                    mlperf_model,
                    scenario_fixed,
                    config,
                    division,
                    system_json,
                    os.path.dirname(scenario_path)
                ):
                    compliance_is_valid = False

        is_valid = accuracy_is_valid and perf_is_valid and compliance_is_valid
        if not is_valid:  # Remove the scenario result
            scenario_measurements_path = change_folder_name_in_path(
                scenario_path, "results", "measurements")
            if scenario in [
                    "Offline", "MultiStream"] and (not accuracy_is_valid or not perf_is_valid) or division == "open":  # they can be inferred
                scenario_compliance_path = change_folder_name_in_path(
                    scenario_path, "results", "compliance")
                log.warning(
                    f"{scenario} scenario result is invalid for {system_desc}: {model} in {division} division. Accuracy: {accuracy_is_valid}, Performance: {perf_is_valid}. Removing...")
                shutil.rmtree(scenario_path)
                shutil.rmtree(scenario_measurements_path)
                shutil.rmtree(scenario_compliance_path)
            elif division in ["closed", "network"]:
                model_results_path = os.path.dirname(scenario_path)
                model_measurements_path = change_folder_name_in_path(
                    model_results_path, "results", "measurements")
                model_compliance_path = change_folder_name_in_path(
                    model_results_path, "results", "compliance")
                model_code_path = os.path.join(
                    change_folder_name_in_path(
                        log_path, "results", "code"), model)
                if not args.nomove_failed_to_open:
                    target_code_path = change_first_directory_to_open(
                        model_code_path)
                    target_results_path = change_first_directory_to_open(
                        model_results_path)
                    target_measurements_path = change_first_directory_to_open(
                        model_measurements_path)
                    target_system_json = change_first_directory_to_open(
                        system_id_json)
                    # if only accuracy or compliance failed, result is valid
                    # for open
                    if not perf_is_valid:
                        log.warning(
                            f"{scenario} scenario result is invalid for {system_desc}: {model} in {division} and open divisions. Accuracy: {accuracy_is_valid}, Performance: {perf_is_valid}. Removing it...")
                        shutil.rmtree(scenario_path)
                        scenario_measurements_path = change_folder_name_in_path(
                            scenario_path, "results", "measurements")
                        shutil.rmtree(scenario_measurements_path)
                    if not os.path.exists(target_results_path):
                        shutil.copytree(
                            model_results_path, target_results_path)
                    if not os.path.exists(target_measurements_path):
                        shutil.copytree(
                            model_measurements_path,
                            target_measurements_path)
                    if not os.path.exists(target_code_path):
                        shutil.copytree(model_code_path, target_code_path)
                    if not os.path.exists(target_system_json):
                        dst_dir = os.path.dirname(target_system_json)
                        if not os.path.exists(dst_dir):
                            os.makedirs(dst_dir, exist_ok=True)
                        import copy
                        target_system_json_contents = copy.deepcopy(
                            system_json)
                        target_system_json_contents['division'] = 'open'
                        with open(target_system_json, 'w') as f:
                            json.dump(target_system_json_contents, f, indent=2)
                    if perf_is_valid:
                        log.warning(f"{scenario} scenario result is invalid for {system_desc}: {model} in {division} division. Accuracy: {accuracy_is_valid}, Performance: {perf_is_valid}. Compliance: {compliance_is_valid}. Moving {model} results to open...")
                    else:
                        log.warning(f"{scenario} scenario result is invalid for {system_desc}: {model} in {division} division. Accuracy: {accuracy_is_valid}, Performance: {perf_is_valid}. Compliance: {compliance_is_valid}. Moving other scenario results of {model} to open...")
                else:
                    log.warning(f"{scenario} scenario result is invalid for {system_desc}: {model} in {division} division. Accuracy: {accuracy_is_valid}, Performance: {perf_is_valid}. Removing all dependent scenario results...")
                clean_model_dir(model_results_path)
            else:  # delete this result
                # delete other scenario results too
                shutil.rmtree(scenario_path)
                # delete other scenario results too
                shutil.rmtree(scenario_measurements_path)
                log.warning(
                    f"{scenario} scenario result is invalid for {system_desc}: {model} in {division} division. Accuracy: {accuracy_is_valid}, Performance: {perf_is_valid}. Removing it...")


def infer_scenario_results(args, config):
    """Walk result dir and check for singlestream (SS) folders and \
            corresponding offline and multistream (MS) ones.
       If SS exists and offline and MS are not existing, \
               SS folder is copied to MS and offline folders.
       If SS and offline exists and MS is not existing, MS is inferred from SS.
       If SS and MS exists and offline is not existing, offline is inferred from MS.
    """
    filter_submitter = args.submitter
    noinfer_low_accuracy_results = args.noinfer_low_accuracy_results

    for division in sorted(
            list_dir(".")):  # process closed and network before open
        # we are looking at ./$division, ie ./closed
        if division not in ["closed", "open", "network"]:
            continue
        is_closed_or_network = division in ["closed", "network"]

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
                    valid_system_types = ["datacenter", "edge",
                                          "datacenter,edge", "edge,datacenter"]
                    if system_type not in valid_system_types:
                        log.error("Division %s, submitter %s, "
                                  "system %s has invalid system type (%s)",
                                  division, submitter, system_id_json, system_type)

                    config.set_type(system_type)

                    for model in list_dir(log_path, system_desc):
                        extra_model_mapping = None
                        if division == "open":
                            model_mapping_path = f"{division}/{submitter}/{config.extra_model_benchmark_map}"
                            if os.path.exists(model_mapping_path):
                                with open(model_mapping_path) as fp:
                                    extra_model_mapping = json.load(fp)

                        mlperf_model = config.get_mlperf_model(
                            model, extra_model_mapping)
                        if not mlperf_model:
                            log.error("Division %s, submitter %s, system %s has "
                                      "invalid model (%s)", division, submitter,
                                      system_id_json, model)
                            continue

                        if mlperf_model not in config.required:
                            log.error("Division %s, submitter %s, system %s has invalid "
                                      "MLPerf model (%s) corresponding to given model (%s). "
                                      "Valid ones for MLPerf inference version (%s) in (%s) "
                                      "category are [%s]", division, submitter, system_id_json,
                                      mlperf_model, model, config.version, system_type,
                                      config.required.keys())
                            continue

                        required_scenarios = config.get_required(mlperf_model)
                        all_scenarios = set(
                            list(required_scenarios)
                            + list(config.get_optional(mlperf_model))
                        )

                        if directory == "results":
                            clean_invalid_results(
                                args,
                                log_path,
                                config,
                                system_desc,
                                system_json,
                                model,
                                mlperf_model,
                                division,
                                system_id_json,
                                is_closed_or_network)
                            if not os.path.exists(os.path.join(
                                    log_path, system_desc, model)):
                                continue

                        for scenario in list_dir(log_path, system_desc, model):

                            scenario_path = os.path.join(
                                log_path, system_desc, model, scenario)

                            if scenario.lower() == "singlestream":
                                tobeinferredpaths = []
                                offline_scenario_path = os.path.join(log_path, system_desc,
                                                                     model, "offline")
                                multistream_scenario_path = os.path.join(log_path, system_desc,
                                                                         model, "multistream")
                                if not os.path.exists(multistream_scenario_path) and \
                                        not os.path.exists(offline_scenario_path):

                                    # infer both the scenarios from SS
                                    tobeinferredpaths = [offline_scenario_path]
                                    if "MultiStream" in all_scenarios:
                                        tobeinferredpaths.append(
                                            multistream_scenario_path)

                                    for tobeinferredpath in tobeinferredpaths:
                                        inferred_scenario = os.path.basename(
                                            tobeinferredpath)
                                        log.info("Division %s, submitter %s, system %s, "
                                                 "model %s: \
                                                inferring %s results from %s",
                                                 division, submitter, system_desc, model,
                                                 inferred_scenario, "singlestream")
                                        shutil.copytree(
                                            scenario_path, tobeinferredpath)

                                elif not os.path.exists(multistream_scenario_path) and \
                                        "MultiStream" in all_scenarios:
                                    # infer MS from SS
                                    for tobeinferredpath in [
                                            multistream_scenario_path]:
                                        log.info("Division %s, submitter %s, system %s, model %s: \
                                                inferring %s results from %s", division, submitter,
                                                 system_desc, model, "multistream", "singlestream")
                                        shutil.copytree(
                                            scenario_path, multistream_scenario_path)
                                elif not os.path.exists(offline_scenario_path):
                                    '''we have both MS and SS results. Inferring from MS is \
                                            expected to be better \
                                            '''
                                    pass

                            elif scenario.lower() == "multistream":
                                offline_scenario_path = os.path.join(log_path, system_desc,
                                                                     model, "offline")
                                '''Need to check if MS is indeed a measured result and not infeered.\
                                        But if MS is indeed inferred from SS, offline scenario will also be \
                                        inferred already by the inferring code above \
                                        '''
                                for tobeinferredpath in [
                                        offline_scenario_path]:
                                    if not os.path.exists(tobeinferredpath):
                                        log.info("Division %s, submitter %s, system %s, model %s: \
                                                inferring %s results from %s", division, submitter,
                                                 system_desc, model, "offline", "multistream")

                                        shutil.copytree(
                                            scenario_path, tobeinferredpath)

                if not noinfer_low_accuracy_results:
                    for system_desc in list_dir(log_path):
                        for model in list_dir(log_path, system_desc):
                            if model.endswith("-99.9"):
                                low_accuracy_model = model[:-2]
                                if low_accuracy_model not in config.required:
                                    continue
                                high_accuracy_model_path = os.path.join(log_path,
                                                                        system_desc, model)
                                low_accuracy_model_path = os.path.join(log_path, system_desc,
                                                                       low_accuracy_model)
                                if not os.path.exists(low_accuracy_model_path):
                                    log.info("Division %s, submitter %s, system %s: \
                                            copying %s results to %s", division, submitter,
                                             system_desc, model, low_accuracy_model)

                                    shutil.copytree(high_accuracy_model_path,
                                                    low_accuracy_model_path)
                                high_accuracy_model_code_path = os.path.join(log_path, "..",
                                                                             "code", model)
                                low_accuracy_model_code_path = os.path.join(log_path, "..",
                                                                            "code", low_accuracy_model)
                                if not os.path.exists(
                                        low_accuracy_model_code_path):
                                    shutil.copytree(high_accuracy_model_code_path,
                                                    low_accuracy_model_code_path)


def main():
    """
    Tool to infer scenario results and cleanup submission tree
    """
    args = get_args()

    src_dir = args.input

    if os.path.exists(args.output):
        log.error(f"output directory {args.output} already exists")
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

    infer_scenario_results(args, config)

    if not args.nodelete_empty_dirs:
        delete_empty_dirs(os.path.join(src_dir))

    return 0


if __name__ == "__main__":
    sys.exit(main())
