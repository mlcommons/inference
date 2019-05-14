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
    log.LogDetail("git_status :\n" + LoadgenGitStatus() + "\n");
    log.LogDetail("git_log :\n" + LoadgenGitLog() + "\n");
  });

  if (LoadgenGitStatus() != "") {
    LogError([](AsyncLog& log) {
      log.LogDetail("Loadgen built with uncommitted changes:");
      log.LogDetail("git_status :\n" + LoadgenGitStatus() + "\n");
    });
  }
}

}  // namespace mlperf
