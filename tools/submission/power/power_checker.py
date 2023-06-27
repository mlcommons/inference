#!/usr/bin/env python3
# Copyright 2018 The MLPerf Authors. All Rights Reserved.
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

from collections import OrderedDict
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any, Optional, Callable
import argparse
import hashlib
import json
import os
import re
import traceback
import uuid
import logging


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")


class LineWithoutTimeStamp(Exception):
    pass


class CheckerWarning(Exception):
    pass


SUPPORTED_VERSION = ["1.10.0"]
SUPPORTED_MODEL = {
    "YokogawaWT210": 8,
    "YokogawaWT500": 35,
    "YokogawaWT500_multichannel": 48,
    "YokogawaWT310": 49,
    "YokogawaWT310E": 49,
    "YokogawaWT330": 52,
    "YokogawaWT330E": 52,
    "YokogawaWT330_multichannel": 77,
}

RANGING_MODE = "ranging"
TESTING_MODE = "run_1"

RESULT_PATHS = [
    "power/client.json",
    "power/client.log",
    "power/ptd_logs.txt",
    "power/server.json",
    "power/server.log",
    RANGING_MODE + "/mlperf_log_detail.txt",
    RANGING_MODE + "/mlperf_log_summary.txt",
    RANGING_MODE + "/spl.txt",
    TESTING_MODE + "/mlperf_log_detail.txt",
    TESTING_MODE + "/mlperf_log_summary.txt",
    TESTING_MODE + "/spl.txt",
]

COMMON_ERROR_RANGING = [
    "Can't evaluate uncertainty of this sample!",
    "Bad watts reading nan from ",
    "Bad amps reading nan from ",
    "Bad pf reading nan from ",
    "Bad volts reading nan from ",
    "Current appears to be too high for set range",
]
COMMON_ERROR_TESTING = ["USB."]
WARNING_NEEDS_TO_BE_ERROR_TESTING_RE = [
    re.compile(r"Uncertainty \d+.\d+%, which is above 1.00% limit for the last sample!")
]

TIME_DELTA_TOLERANCE = 500  # in milliseconds


def _normalize(path: str) -> str:
    allparts: List[str] = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        if parts[1] == path:  # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        path = parts[0]
        allparts.insert(0, parts[1])
    return "/".join(allparts)


def _sort_dict(x: Dict[str, Any]) -> "OrderedDict[str, Any]":
    return OrderedDict(sorted(x.items()))


def hash_dir(dirname: str) -> Dict[str, str]:
    result: Dict[str, str] = {}

    for path, dirs, files in os.walk(dirname, topdown=True):
        relpath = os.path.relpath(path, dirname)
        if relpath == ".":
            relpath = ""
        for file in files:
            fname = os.path.join(relpath, file)
            with open(os.path.join(path, file), "rb") as f:
                result[_normalize(fname)] = hashlib.sha1(f.read()).hexdigest()

    return _sort_dict(result)


def get_time_from_line(
    line: str, data_regexp: str, file: str, timezone_offset: int
) -> float:
    log_time_str = re.search(data_regexp, line)
    if log_time_str and log_time_str.group(0):
        log_datetime = datetime.strptime(log_time_str.group(0), "%m-%d-%Y %H:%M:%S.%f")
        return log_datetime.replace(tzinfo=timezone.utc).timestamp() + timezone_offset
    raise LineWithoutTimeStamp(f"{line.strip()!r} in {file}.")


class SessionDescriptor:
    def __init__(self, path: str):
        self.path = path
        with open(path, "r") as f:
            self.json_object: Dict[str, Any] = json.loads(f.read())
            self.required_fields_check()

    def required_fields_check(self) -> None:
        required_fields = [
            "version",
            "timezone",
            "modules",
            "sources",
            "messages",
            "uuid",
            "session_name",
            "results",
            "phases",
        ]
        absent_keys = set(required_fields) - self.json_object.keys()
        assert (
            len(absent_keys) == 0
        ), f"Required fields {', '.join(absent_keys)!r} does not exist in {self.path!r}"


def compare_dicts_values(d1: Dict[str, str], d2: Dict[str, str], comment: str) -> None:
    files_with_diff_check_sum = {k: d1[k] for k in d1 if k in d2 and d1[k] != d2[k]}
    assert len(files_with_diff_check_sum) == 0, f"{comment}" + "".join(
        [
            f"Expected {d1[i]}, but got {d2[i]} for {i}\n"
            for i in files_with_diff_check_sum
        ]
    )


def compare_dicts(s1: Dict[str, str], s2: Dict[str, str], comment: str) -> None:
    assert (
        not s1.keys() - s2.keys()
    ), f"{comment} Missing {', '.join(sorted(s1.keys() - s2.keys()))!r}"
    assert (
        not s2.keys() - s1.keys()
    ), f"{comment} Extra {', '.join(sorted(s2.keys() - s1.keys()))!r}"

    compare_dicts_values(s1, s2, comment)


def sources_check(sd: SessionDescriptor) -> None:
    """Compare the current checksum of the code from client.json or server.json
    against the standard checksum of the source code from sources_checksums.json.
    """
    s = sd.json_object["sources"]

    with open(os.path.join(os.path.dirname(__file__), "sources_checksums.json")) as f:
        sources_samples = json.load(f)

    assert s in sources_samples, f"{s} do not exist in 'sources_checksums.json'"


def ptd_messages_check(sd: SessionDescriptor) -> None:
    """Performs multiple checks:
    - Check the ptd version number.
    - Check the device model.
    - Compare message replies with expected values.
    - Check that initial values set after the test is completed.
    """
    msgs: List[Dict[str, str]] = sd.json_object["ptd_messages"]

    def get_ptd_answer(command: str) -> str:
        for msg in msgs:
            if msg["cmd"] == command:
                return msg["reply"]
        return ""

    identify_answer = get_ptd_answer("Identify")
    assert (
        len(identify_answer) != 0
    ), "There is no answer to the 'Identify' command for PTD."
    power_meter_model = identify_answer.split(",")[0]
    groups = re.search(r"(?<=version=)(.+?)-", identify_answer)
    version = "" if groups is None else groups.group(1)

    assert (
        version in SUPPORTED_VERSION
    ), f"PTD version {version!r} is not supported. Supported versions are 1.10.0"
    assert (
        power_meter_model in SUPPORTED_MODEL.keys()
    ), f"Power meter {power_meter_model!r} is not supported. Only {', '.join(SUPPORTED_MODEL.keys())} are supported."

    def check_reply(cmd: str, reply: str) -> None:
        stop_counter = 0
        for msg in msgs:
            if msg["cmd"].startswith(cmd):
                if msg["cmd"] == "Stop":
                    # In normal flow the third answer to stop command is `Error: no measurement to stop`
                    if stop_counter == 2:
                        reply = "Error: no measurement to stop"
                    stop_counter += 1
                assert (
                    reply == msg["reply"]
                ), f"Wrong reply for {msg['cmd']!r} command. Expected {reply!r}, but got {msg['reply']!r}"

    check_reply("SR,A", "Range A changed")
    check_reply("SR,V", "Range V changed")
    check_reply(
        "Go,1000,",
        "Starting untimed measurement, maximum 500000 samples at 1000ms with 0 rampup samples",
    )
    check_reply("Stop", "Stopping untimed measurement")

    def get_initial_range(param_num: int, reply: str) -> str:
        reply_list = reply.split(",")
        try:
            if reply_list[param_num] == "0" and float(reply_list[param_num + 1]) > 0:
                return reply_list[param_num + 1]
        except (ValueError, IndexError):
            assert False, f"Can not get power meters initial values from {reply!r}"
        return "Auto"

    def get_command_by_value_and_number(cmd: str, number: int) -> Optional[str]:
        command_counter = 0
        for msg in msgs:
            if msg["cmd"].startswith(cmd):
                command_counter += 1
                if command_counter == number:
                    return msg["cmd"]
        assert False, f"Can not find the {number} command starting with {cmd!r}."
        return None

    initial_amps = get_initial_range(1, msgs[2]["reply"])
    initial_volts = get_initial_range(3, msgs[2]["reply"])

    initial_amps_command = get_command_by_value_and_number("SR,A", 3)
    initial_volts_command = get_command_by_value_and_number("SR,V", 3)

    assert (
        initial_amps_command == f"SR,A,{initial_amps}"
    ), f"Do not set Amps range as initial. Expected 'SR,A,{initial_amps}', got {initial_amps_command!r}."
    assert (
        initial_volts_command == f"SR,V,{initial_volts}"
    ), f"Do not set Volts range as initial. Expected 'SR,V,{initial_volts}', got {initial_volts_command!r}."


def uuid_check(client_sd: SessionDescriptor, server_sd: SessionDescriptor) -> None:
    """Compare UUIDs from client.json and server.json. They should be the same."""
    uuid_c = client_sd.json_object["uuid"]
    uuid_s = server_sd.json_object["uuid"]

    assert uuid.UUID(uuid_c["client"]) == uuid.UUID(
        uuid_s["client"]
    ), "'client uuid' is not equal."
    assert uuid.UUID(uuid_c["server"]) == uuid.UUID(
        uuid_s["server"]
    ), "'server uuid' is not equal."


def _get_begin_end_time_from_mlperf_log_detail(
    path: str, client_sd: SessionDescriptor
) -> Tuple[float, float]:
    system_begin = None
    system_end = None

    file = os.path.join(path, "mlperf_log_detail.txt")

    with open(file) as f:
        for line in f:
            if re.search("power_begin", line.lower()):
                system_begin = get_time_from_line(
                    line,
                    r"(\d*-\d*-\d* \d*:\d*:\d*\.\d*)",
                    file,
                    0,
                )
            elif re.search("power_end", line.lower()):
                system_end = get_time_from_line(
                    line,
                    r"(\d*-\d*-\d* \d*:\d*:\d*\.\d*)",
                    file,
                    0,
                )
            if system_begin and system_end:
                break

    assert system_begin is not None, f"Can not get power_begin time from {file!r}"
    assert system_end is not None, f"Can not get power_end time from {file!r}"

    return system_begin, system_end


def phases_check(
    client_sd: SessionDescriptor, server_sd: SessionDescriptor, path: str
) -> None:
    """Check that the time difference between corresponding checkpoint values
    from client.json and server.json is less than or equal to TIME_DELTA_TOLERANCE ms.
    Check that the loadgen timestamps are within workload time interval.
    Check that the duration of loadgen test for the ranging mode is comparable
    with duration of loadgen test for the testing mode.
    """
    phases_ranging_c = client_sd.json_object["phases"]["ranging"]
    phases_testing_c = client_sd.json_object["phases"]["testing"]
    phases_ranging_s = server_sd.json_object["phases"]["ranging"]
    phases_testing_s = server_sd.json_object["phases"]["testing"]

    def compare_time(
        phases_client: List[List[float]], phases_server: List[List[float]], mode: str
    ) -> None:
        assert len(phases_client) == len(
            phases_server
        ), f"Phases amount is not equal for {mode} mode."
        for i in range(len(phases_client)):
            time_difference = abs(phases_client[i][0] - phases_server[i][0])
            assert time_difference <= TIME_DELTA_TOLERANCE / 1000, (
                f"The time difference for {i + 1} phase of {mode} mode is more than {TIME_DELTA_TOLERANCE}ms."
                f"Observed difference is {time_difference * 1000}ms"
            )

    compare_time(phases_ranging_c, phases_ranging_s, RANGING_MODE)
    compare_time(phases_testing_c, phases_testing_s, TESTING_MODE)

    def compare_duration(range_duration: float, test_duration: float) -> None:
        duration_diff = (range_duration - test_duration) / range_duration

        if duration_diff > 0.5:
            raise CheckerWarning(
                f"Duration of the testing mode ({round(test_duration,2)}) is lower than that of "
                f"ranging mode ({round(range_duration,2)}) by {round(duration_diff*100,2)} "
                f"percent which is more than the expected 5 percent limit."
            )

    def compare_time_boundaries(
        begin: float, end: float, phases: List[Any], mode: str
    ) -> None:
        # TODO: temporary workaround, remove when proper DST handling is implemented!
        assert (
            phases[1][0] < begin < phases[2][0]
            or phases[1][0] < begin - 3600 < phases[2][0]
        ), f"Loadgen test begin time is not within {mode} mode time interval."
        assert (
            phases[1][0] < end < phases[2][0]
            or phases[1][0] < end - 3600 < phases[2][0]
        ), f"Loadgen test end time is not within {mode} mode time interval."

    system_begin_r, system_end_r = _get_begin_end_time_from_mlperf_log_detail(
        os.path.join(path, "ranging"), client_sd
    )

    system_begin_t, system_end_t = _get_begin_end_time_from_mlperf_log_detail(
        os.path.join(path, "run_1"), client_sd
    )

    compare_time_boundaries(system_begin_r, system_end_r, phases_ranging_c, "ranging")
    compare_time_boundaries(system_begin_t, system_end_t, phases_testing_c, "testing")

    ranging_duration_d = system_end_r - system_begin_r
    testing_duration_d = system_end_t - system_begin_t

    compare_duration(ranging_duration_d, testing_duration_d)

    def get_avg_power(power_path: str, run_path: str) -> Tuple[float, float]:
        # parse the power logs

        power_begin, power_end = _get_begin_end_time_from_mlperf_log_detail(
            os.path.join(path, os.path.basename(run_path)), client_sd
        )

        detail_log_fname = os.path.join(run_path, "mlperf_log_detail.txt")
        datetime_format = "%m-%d-%Y %H:%M:%S.%f"

        spl_fname = os.path.join(run_path, "spl.txt")
        power_list = []
        pf_list = []

        with open(spl_fname) as f:
            for line in f:
                timestamp = (
                    datetime.strptime(line.split(",")[1], datetime_format)
                ).timestamp()
                if timestamp > power_begin and timestamp < power_end:
                    cpower = float(line.split(",")[3])
                    cpf = float(line.split(",")[9])
                    if cpower > 0:
                        power_list.append(cpower)
                    if cpf > 0:
                        pf_list.append(cpf)

        if len(power_list) == 0:
            power = -1.0
        else:
            power = sum(power_list) / len(power_list)
        if len(pf_list) == 0:
            pf = -1.0
        else:
            pf = sum(pf_list) / len(pf_list)
        return power, pf

    ranging_watts, ranging_pf = get_avg_power(
        os.path.join(path, "power"), os.path.join(path, "ranging")
    )
    testing_watts, testing_pf = get_avg_power(
        os.path.join(path, "power"), os.path.join(path, "run_1")
    )
    ranging_watts = round(ranging_watts, 5)
    testing_watts = round(testing_watts, 5)
    ranging_pf = round(ranging_pf, 5)
    testing_pf = round(testing_pf, 5)

    delta = round((float(testing_watts) / float(ranging_watts) - 1) * 100, 2)

    assert delta > -5, (
        f"Average power during the testing mode run is lower than that during the ranging run by more than 5%. "
        f"Observed delta is {delta}% "
        f"with avg. ranging power {ranging_watts}, avg.testing power {testing_watts}, "
        f"avg. ranging power factor {ranging_pf} and avg. testing power factor {testing_pf}"
    )
    # print(f"{path},{ranging_watts},{testing_watts},{delta}%,{ranging_pf},{testing_pf}\n")


def session_name_check(
    client_sd: SessionDescriptor, server_sd: SessionDescriptor
) -> None:
    """Check that session names from client.json and server.json are equal."""
    session_name_c = client_sd.json_object["session_name"]
    session_name_s = server_sd.json_object["session_name"]
    assert (
        session_name_c == session_name_s
    ), f"Session name is not equal. Client session name is {session_name_c!r}. Server session name is {session_name_s!r}"


def messages_check(client_sd: SessionDescriptor, server_sd: SessionDescriptor) -> None:
    """Compare client and server messages list length.
    Compare messages values and replies from client.json and server.json.
    Compare client and server version.
    """
    mc = client_sd.json_object["messages"]
    ms = server_sd.json_object["messages"]

    assert len(mc) == len(
        ms
    ), f"Client commands list length ({len(mc)}) should be equal to server commands list length ({len(ms)}). "

    # Check that server.json contains all client.json messages and replies.
    for i in range(len(mc)):
        assert (
            mc[i]["cmd"] == ms[i]["cmd"]
        ), f"Commands {i} are different. Server command is {ms[i]['cmd']!r}. Client command is {mc[i]['cmd']!r}."
        if "time" != mc[i]["cmd"]:
            assert mc[i]["reply"] == ms[i]["reply"], (
                f"Replies on command {mc[i]['cmd']!r} are different. "
                f"Server reply is {ms[i]['reply']!r}. Client command is {mc[i]['reply']!r}."
            )

    # Check client and server version from server.json.
    # Server.json contains all client.json messages and replies. Checked earlier.
    def get_version(regexp: str, line: str) -> str:
        version_o = re.search(regexp, line)
        assert version_o is not None, f"Server version is not defined in:'{line}'"
        return version_o.group(1)

    client_version = get_version(r"mlcommons\/power client v(\d+)$", ms[0]["cmd"])
    server_version = get_version(r"mlcommons\/power server v(\d+)$", ms[0]["reply"])

    assert (
        client_version == server_version
    ), f"Client.py version ({client_version}) is not equal server.py version ({server_version})."


def results_check(
    server_sd: SessionDescriptor, client_sd: SessionDescriptor, result_path: str
) -> None:
    """Calculate the checksum for result files. Compare them with the checksums
    list formed from joined results from server.json and client.json.
    Check that results from client.json and server.json have no extra and absent files.
    Compare that results files from client.json and server.json have the same checksum.
    """

    # Hashes of the files in results directory
    results = dict(hash_dir(result_path))
    # Hashes recorded in server.json
    results_s = server_sd.json_object["results"]
    # Hashes recorded in client.json
    results_c = client_sd.json_object["results"]

    # TODO: server.json checksum
    results.pop("power/server.json")
    # TODO: client.json checksum is no longer recorded
    results.pop("power/client.json")
    result_paths_copy = RESULT_PATHS.copy()
    result_paths_copy.remove("power/server.json")
    result_paths_copy.remove("power/client.json")

    def remove_optional_path(res: Dict[str, str]) -> None:
        keys = list(res.keys())
        for path in keys:
            # Ignore all the optional files.
            if path not in result_paths_copy:
                del res[path]

    # We only check the hashes of the files required for submission.
    remove_optional_path(results_s)
    remove_optional_path(results_c)
    remove_optional_path(results)

    # Make sure the hashes match between server.json and client.json
    compare_dicts_values(
        results_s,
        results_c,
        f"{server_sd.path} and {client_sd.path} results checksum comparison",
    )
    compare_dicts_values(
        results_c,
        results_s,
        f"{client_sd.path} and {server_sd.path} results checksum comparison",
    )

    # Check if the hashes of the files in results directory match the ones recorded in server.json/client.json.
    result_c_s = {**results_c, **results_s}

    compare_dicts(
        result_c_s,
        results,
        f"{server_sd.path} + {client_sd.path} results checksum values and "
        f"calculated {result_path} content checksum comparison:\n",
    )

    # Check if all the required files are present
    def result_files_compare(
        res: Dict[str, str], ref_res: List[str], path: str
    ) -> None:
        # If a file is required (in ref_res) but is not present in results directory (res),
        # then the submission is invalid.
        absent_files = set(ref_res) - set(res.keys())
        assert (
            len(absent_files) == 0
        ), f"There are absent files {', '.join(absent_files)!r} in the results of {path}"

    result_files_compare(
        result_c_s, result_paths_copy, f"{server_sd.path} + {client_sd.path}"
    )
    result_files_compare(results, result_paths_copy, result_path)


def check_ptd_logs(
    server_sd: SessionDescriptor, client_sd: SessionDescriptor, path: str
) -> None:
    """Check if ptd message starts with 'WARNING' or 'ERROR' in ptd logs.
    Check 'Uncertainty checking for Yokogawa... is activated' in PTD logs.
    """
    start_ranging_time = None
    stop_ranging_time = None
    ranging_mark = f"{server_sd.json_object['session_name']}_ranging"

    start_load_time, stop_load_time = _get_begin_end_time_from_mlperf_log_detail(
        os.path.join(path, "run_1"), client_sd
    )

    file_path = os.path.join(path, "power", "ptd_logs.txt")
    date_regexp = r"(^\d\d-\d\d-\d\d\d\d \d\d:\d\d:\d\d.\d\d\d)"
    timezone_offset = int(server_sd.json_object["timezone"])

    with open(file_path, "r") as f:
        ptd_log_lines = f.readlines()

    def find_error_or_warning(reg_exp: str, line: str, error: bool) -> None:
        problem_line = re.search(reg_exp, line)

        if problem_line and problem_line.group(0):
            log_time = get_time_from_line(line, date_regexp, file_path, timezone_offset)
            if start_ranging_time is None or stop_ranging_time is None:
                assert False, "Can not find ranging time in ptd_logs.txt."
            if error:
                if problem_line.group(0).strip() in COMMON_ERROR_TESTING:
                    raise CheckerWarning(
                        f"{line.strip()!r} in ptd_log.txt during testing stage but it is accepted. Treated as WARNING"
                    )
                assert (
                    start_ranging_time < log_time < stop_ranging_time
                ), f"{line.strip()!r} in ptd_log.txt"

                # Treat uncommon errors in ranging phase as warnings
                if all(
                    not problem_line.group(0).strip().startswith(common_ranging_error)
                    for common_ranging_error in COMMON_ERROR_RANGING
                ):
                    raise CheckerWarning(
                        f"{line.strip()!r} in ptd_log.txt during ranging stage. Treated as WARNING"
                    )
            else:
                if (
                    start_load_time + TIME_DELTA_TOLERANCE
                    < log_time
                    < stop_load_time - TIME_DELTA_TOLERANCE
                ):
                    for warning_to_be_error in WARNING_NEEDS_TO_BE_ERROR_TESTING_RE:
                        warning_line = warning_to_be_error.search(
                            problem_line.group(0).strip()
                        )
                        if warning_line and warning_line.group(0):
                            assert (
                                False
                            ), f"{line.strip()!r} during testing phase. Test start time: {start_load_time}, Log time: {log_time}, Test stop time: {stop_load_time} "

                    raise CheckerWarning(
                        f"{line.strip()!r} in ptd_log.txt during load stage"
                    )

    start_ranging_line = f": Go with mark {ranging_mark!r}"

    def get_msg_without_time(line: str) -> Optional[str]:
        try:
            get_time_from_line(line, date_regexp, file_path, timezone_offset)
        except LineWithoutTimeStamp:
            return line
        msg_o = re.search(f"(?<={date_regexp}).+", line)
        if msg_o is None:
            return None
        return msg_o.group(0).strip()

    for line in ptd_log_lines:
        msg = get_msg_without_time(line)
        if msg is None:
            continue
        if (not start_ranging_time) and (start_ranging_line == msg):
            start_ranging_time = get_time_from_line(
                line, date_regexp, file_path, timezone_offset
            )
        if (not stop_ranging_time) and bool(start_ranging_time):
            if ": Completed test" == msg:
                stop_ranging_time = get_time_from_line(
                    line, date_regexp, file_path, timezone_offset
                )
                break

    if start_ranging_time is None or stop_ranging_time is None:
        assert False, "Can not find ranging time in ptd_logs.txt."

    is_uncertainty_check_activated = False

    for line in ptd_log_lines:
        msg_o = re.search(r"Uncertainty checking for Yokogawa\S+ is activated", line)
        if msg_o is not None:
            try:
                log_time = None
                log_time = get_time_from_line(
                    line, date_regexp, file_path, timezone_offset
                )
            except LineWithoutTimeStamp:
                assert (
                    log_time is not None
                ), "ptd_logs.txt: Can not get timestamp for 'Uncertainty checking for Yokogawa... is activated' message."
            assert (
                start_ranging_time is not None and log_time < start_ranging_time
            ), "ptd_logs.txt: Uncertainty checking Yokogawa... was activated after ranging mode was started."
            is_uncertainty_check_activated = True
            break

    assert (
        is_uncertainty_check_activated
    ), "ptd_logs.txt: Line 'Uncertainty checking for Yokogawa... is activated' is not found."

    for line in ptd_log_lines:
        find_error_or_warning("(?<=WARNING:).+", line, error=False)
        find_error_or_warning("(?<=ERROR:).+", line, error=True)


def check_ptd_config(server_sd: SessionDescriptor) -> None:
    """Check the device number is supported.
    If the device is multichannel, check that two numbers are using for channel configuration.
    """
    ptd_config = server_sd.json_object["ptd_config"]

    dev_num = ptd_config["device_type"]
    assert (
        dev_num in SUPPORTED_MODEL.values()
    ), f"Device number {dev_num} is not supported. Supported numbers are " + ", ".join(
        [str(i) for i in set(SUPPORTED_MODEL.values())]
    )

    if dev_num == 77:
        channels = ""
        command = ptd_config["command"]

        for i in range(len(command)):
            if command[i] == "-c":
                channels = command[i + 1]
                break

        dev_name = ""
        for name, num in SUPPORTED_MODEL.items():
            if num == dev_num:
                dev_name = name
                break

        assert (
            len(channels.split(",")) == 2
            and ptd_config["channel"]
            and len(ptd_config["channel"]) == 2
        ), f"Expected multichannel mode for {dev_name}, but got 1-channel."


def debug_check(server_sd: SessionDescriptor) -> None:
    """Check debug is disabled on server-side"""
    assert (
        server_sd.json_object.get("debug", False) is False
    ), "Server was running in debug mode"


def check_with_logging(check_name: str, check: Callable[[], None]) -> Tuple[bool, bool]:
    try:
        check()
    except AssertionError as e:
        log.error(f"[ ] {check_name}")
        log.error(f"\t{e}\n")
        return False, False
    except CheckerWarning as e:
        log.warning(f"[x] {check_name}")
        log.warning(f"\t{e}\n")
        return True, True
    except Exception:
        log.exception(f"[ ] {check_name}")
        log.exception("Unhandled exeception:")
        traceback.print_exc()
        return False, False
    else:
        log.info(f"[x] {check_name}")
    return True, False


def check(path: str) -> int:
    client = SessionDescriptor(os.path.join(path, "power/client.json"))
    server = SessionDescriptor(os.path.join(path, "power/server.json"))

    check_with_description = {
        "Check client sources checksum": lambda: sources_check(client),
        "Check server sources checksum": lambda: sources_check(server),
        "Check PTD commands and replies": lambda: ptd_messages_check(server),
        "Check UUID": lambda: uuid_check(client, server),
        "Check session name": lambda: session_name_check(client, server),
        "Check time difference": lambda: phases_check(client, server, path),
        "Check client server messages": lambda: messages_check(client, server),
        "Check results checksum": lambda: results_check(server, client, path),
        "Check errors and warnings from PTD logs": lambda: check_ptd_logs(
            server, client, path
        ),
        "Check PTD configuration": lambda: check_ptd_config(server),
        "Check debug is disabled on server-side": lambda: debug_check(server),
    }

    result = True
    warnings = False

    for description in check_with_description.keys():
        check_result, check_warnings = check_with_logging(
            description, check_with_description[description]
        )
        result &= check_result
        warnings |= check_warnings

    if result:
        log.info(
            "\nAll checks passed"
            f"{'. Warnings encountered, check for audit!' if warnings else ''}"
        )
    else:
        log.error(
            f"\nERROR: Not all checks passed"
            f"{'. Warnings encountered, check for audit!' if warnings else ''}"
        )

    return 0 if result else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check PTD client-server session results"
    )
    parser.add_argument("session_directory", help="directory with session results data")

    args = parser.parse_args()

    return_code = check(args.session_directory)

    exit(return_code)
