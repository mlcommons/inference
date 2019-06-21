/* Copyright 2019 The MLPerf Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "version.h"

#include "logging.h"

namespace mlperf {

void LogLoadgenVersion() {
  LogDetail([](AsyncLog& log) {
    log.LogDetail("LoadgenVersionInfo:");
    log.LogDetail("version : " + LoadgenVersion() + " @ " +
                  LoadgenGitRevision());
    log.LogDetail("build_date_local : " + LoadgenBuildDateLocal());
    log.LogDetail("build_date_utc   : " + LoadgenBuildDateUtc());
    log.LogDetail("git_commit_date  : " + LoadgenGitCommitDate());
    log.LogDetail("git_log :\n\n" + LoadgenGitLog() + "\n");
    log.LogDetail("git_status :\n\n" + LoadgenGitStatus() + "\n");
    if (!LoadgenGitStatus().empty() && LoadgenGitStatus() != "NA") {
      log.FlagError();
      log.LogDetail("Loadgen built with uncommitted changes!");
    }
    log.LogDetail("SHA1 of files :\n\n" + LoadgenSha1OfFiles() + "\n");
  });
}

}  // namespace mlperf
