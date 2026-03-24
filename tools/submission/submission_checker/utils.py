import os
from .constants import *
from .parsers.loadgen_parser import LoadgenParser


def list_dir(*path):
    path = os.path.join(*path)
    return sorted([f for f in os.listdir(
        path) if os.path.isdir(os.path.join(path, f))])


def list_files(*path):
    path = os.path.join(*path)
    return sorted([f for f in os.listdir(
        path) if os.path.isfile(os.path.join(path, f))])


def list_empty_dirs_recursively(*path):
    path = os.path.join(*path)
    return [dirpath for dirpath, dirs, files in os.walk(
        path) if not dirs and not files]


def list_dirs_recursively(*path):
    path = os.path.join(*path)
    return [dirpath for dirpath, dirs, files in os.walk(path)]


def list_files_recursively(*path):
    path = os.path.join(*path)
    return [
        os.path.join(dirpath, file)
        for dirpath, dirs, files in os.walk(path)
        for file in files
    ]


def files_diff(list1, list2, optional=None):
    """returns a list of files that are missing or added."""
    if not optional:
        optional = []
    optional = optional + ["mlperf_log_trace.json", "results.json", ".gitkeep"]
    return set(list1).symmetric_difference(set(list2)) - set(optional)


def check_extra_files(path, target_files):
    missing_files = []
    check_pass = True
    folders = list_dir(path)
    for dir in target_files.keys():
        if dir not in folders:
            check_pass = False
            missing_files.append(os.path.join(path, dir))
        else:
            files = [f.split(".")[0]
                     for f in list_files(os.path.join(path, dir))]
            for target_file in target_files[dir]:
                if target_file not in files:
                    check_pass = False
                    missing_files.append(
                        f"{os.path.join(path, dir, target_file)}.png")
            if "captions" not in files:
                missing_files.append(
                    f"{os.path.join(path, dir, 'captions.txt')}")
    return check_pass, missing_files


def split_path(m):
    return m.replace("\\", "/").split("/")


def get_boolean(s):
    if s is None:
        return False
    elif isinstance(s, bool):
        return s
    elif isinstance(s, str):
        return s.lower() == "true"
    elif isinstance(s, int):
        return bool(s)
    else:
        raise TypeError(
            f"Variable should be bool, string or int, got {type(s)} instead"
        )


def merge_two_dict(x, y):
    z = x.copy()
    for key in y:
        if key not in z:
            z[key] = y[key]
        else:
            z[key] += y[key]
    return z


def sum_dict_values(x):
    count = 0
    for key in x:
        count += x[key]
    return count


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    
def lower_list(l):
    return [str(e).lower() for e in l]

def contains_list(l1, l2):
    # Check if l1 contains all elements of l2
    missing = []
    for e in l2:
        if e not in l1:
            missing.append(e)
    return missing, len(missing) == 0 


def get_performance_metric(
        config, model, path, scenario_fixed):
    # Assumes new logging format
    version = config.version

    fname = os.path.join(path, "mlperf_log_detail.txt")
    mlperf_log = LoadgenParser(fname)
    if (
        "result_validity" in mlperf_log.get_keys()
        and mlperf_log["result_validity"] == "VALID"
    ):
        is_valid = True
    scenario = mlperf_log["effective_scenario"]

    res = float(mlperf_log[RESULT_FIELD_NEW[version][scenario]])
    if (
        version in RESULT_FIELD_BENCHMARK_OVERWRITE
        and model in RESULT_FIELD_BENCHMARK_OVERWRITE[version]
        and scenario in RESULT_FIELD_BENCHMARK_OVERWRITE[version][model]
    ):
        res = float(
            mlperf_log[RESULT_FIELD_BENCHMARK_OVERWRITE[version]
                       [model][scenario]]
        )

    inferred = False
    if scenario_fixed != scenario:
        inferred, res, _ = get_inferred_result(
            scenario_fixed, scenario, res, mlperf_log, config, False
        )

    return res


def get_inferred_result(
    scenario_fixed, scenario, res, mlperf_log, config, log_error=False
):

    inferred = False
    is_valid = True
    # Check if current scenario (and version) uses early stopping
    uses_early_stopping = config.uses_early_stopping(scenario)

    latency_mean = mlperf_log["result_mean_latency_ns"]
    if scenario in ["MultiStream"]:
        latency_99_percentile = mlperf_log[
            "result_99.00_percentile_per_query_latency_ns"
        ]
        latency_mean = mlperf_log["result_mean_query_latency_ns"]
    samples_per_query = mlperf_log["effective_samples_per_query"]
    if scenario == "SingleStream":
        # qps_wo_loadgen_overhead is only used for inferring Offline from
        # SingleStream; only for old submissions
        qps_wo_loadgen_overhead = mlperf_log["result_qps_without_loadgen_overhead"]

    # special case for results inferred from different scenario
    if scenario_fixed in ["Offline"] and scenario in ["SingleStream"]:
        inferred = True
        res = qps_wo_loadgen_overhead

    if (scenario_fixed in ["Offline"]) and scenario in ["MultiStream"]:
        inferred = True
        res = samples_per_query * S_TO_MS / (latency_mean / MS_TO_NS)

    if (scenario_fixed in ["MultiStream"]) and scenario in ["SingleStream"]:
        inferred = True
        # samples_per_query does not match with the one reported in the logs
        # when inferring MultiStream from SingleStream
        samples_per_query = 8
        if uses_early_stopping:
            early_stopping_latency_ms = mlperf_log["early_stopping_latency_ms"]
            if early_stopping_latency_ms == 0 and log_error:
                log.error(
                    "Not enough samples were processed for early stopping to make an estimate"
                )
                is_valid = False
            res = (early_stopping_latency_ms * samples_per_query) / MS_TO_NS
        else:
            res = (latency_99_percentile * samples_per_query) / MS_TO_NS
    if (scenario_fixed in ["Interactive"]) and scenario not in ["Server"]:
        is_valid = False
    return inferred, res, is_valid


def check_compliance_perf_dir(test_dir):
    is_valid = False
    import logging
    log = logging.getLogger("main")

    fname = os.path.join(test_dir, "verify_performance.txt")
    if not os.path.exists(fname):
        log.error("%s is missing in %s", fname, test_dir)
        is_valid = False
    else:
        with open(fname, "r") as f:
            for line in f:
                # look for: TEST PASS
                if "TEST PASS" in line:
                    is_valid = True
                    break
        if is_valid == False:
            log.error(
                "Compliance test performance check in %s failed",
                test_dir)

        # Check performance dir
        test_perf_path = os.path.join(test_dir, "performance", "run_1")
        if not os.path.exists(test_perf_path):
            log.error("%s has no performance/run_1 directory", test_dir)
            is_valid = False
        else:
            diff = files_diff(
                list_files(test_perf_path),
                REQUIRED_COMP_PER_FILES,
                ["mlperf_log_accuracy.json"],
            )
            if diff:
                log.error(
                    "%s has file list mismatch (%s)",
                    test_perf_path,
                    diff)
                is_valid = False

    return is_valid


def get_power_metric(config, scenario_fixed, log_path, is_valid, res):
    # parse the power logs
    import datetime
    import logging
    log = logging.getLogger("main")
    server_timezone = datetime.timedelta(0)
    client_timezone = datetime.timedelta(0)

    detail_log_fname = os.path.join(log_path, "mlperf_log_detail.txt")
    mlperf_log = LoadgenParser(detail_log_fname)
    datetime_format = "%m-%d-%Y %H:%M:%S.%f"
    power_begin = (
        datetime.datetime.strptime(mlperf_log["power_begin"], datetime_format)
        + client_timezone
    )
    power_end = (
        datetime.datetime.strptime(mlperf_log["power_end"], datetime_format)
        + client_timezone
    )
    # Obtain the scenario also from logs to check if power is inferred
    scenario = mlperf_log["effective_scenario"]

    spl_fname = os.path.join(log_path, "spl.txt")
    power_list = []
    with open(spl_fname) as f:
        for line in f:
            if not line.startswith("Time"):
                continue
            timestamp = (
                datetime.datetime.strptime(line.split(",")[1], datetime_format)
                + server_timezone
            )
            if timestamp > power_begin and timestamp < power_end:
                value = float(line.split(",")[3])
                if value > 0:
                    power_list.append(float(line.split(",")[3]))

    if len(power_list) == 0:
        log.error(
            "%s has no power samples falling in power range: %s - %s",
            spl_fname,
            power_begin,
            power_end,
        )
        is_valid = False
    else:
        avg_power = sum(power_list) / len(power_list)
        power_duration = (power_end - power_begin).total_seconds()
        if scenario_fixed in ["Offline", "Server", "Interactive"]:
            # In Offline and Server scenarios, the power metric is in W.
            power_metric = avg_power
            avg_power_efficiency = res / avg_power

        else:
            # In SingleStream and MultiStream scenarios, the power metric is in
            # mJ/query.
            assert scenario_fixed in [
                "MultiStream",
                "SingleStream",
            ], "Unknown scenario: {:}".format(scenario_fixed)

            num_queries = int(mlperf_log["result_query_count"])

            power_metric = avg_power * power_duration * 1000 / num_queries

            if scenario_fixed in ["SingleStream"]:
                samples_per_query = 1
            elif scenario_fixed in ["MultiStream"]:
                samples_per_query = 8

            if (scenario_fixed in ["MultiStream"]
                ) and scenario in ["SingleStream"]:
                power_metric = (
                    avg_power * power_duration * samples_per_query * 1000 / num_queries
                )

            avg_power_efficiency = (samples_per_query * 1000) / power_metric

    return is_valid, power_metric, scenario, avg_power_efficiency
