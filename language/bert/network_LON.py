# Copyright 2023 MLCommons. All Rights Reserved.
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


import sys
import os
sys.path.insert(0, os.getcwd())
from absl import app
import squad_QSL
import mlperf_loadgen as lg
import bert_QDL

def set_args(argv, g_settings, g_log_settings, g_audit_conf, g_sut_server, g_backend, g_total_count_override=None, g_perf_count_override=None):

    global settings, log_settings, audit_conf, sut_server, total_count_override, perf_count_override, backend
    sys.argv = sys.argv[0:1]
    settings = g_settings
    log_settings = g_log_settings
    audit_conf = g_audit_conf
    sut_server = g_sut_server
    total_count_override = g_total_count_override
    perf_count_override = g_perf_count_override
    backend = g_backend

def main(argv):
        qsl = squad_QSL.get_squad_QSL(total_count_override, perf_count_override)
        qdl = bert_QDL.bert_QDL(qsl, sut_server_addr=sut_server)

        lg.StartTestWithLogSettings(qdl.qdl, qsl.qsl, settings, log_settings, audit_conf)


if __name__ == "__main__":
    app.run(main)
