#include "logging.h"
#include "version.h"

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
    if (LoadgenGitStatus() != "") {
      log.FlagError();
      log.LogDetail("Loadgen built with uncommitted changes!");
    }
  });
}

}  // namespace mlperf
