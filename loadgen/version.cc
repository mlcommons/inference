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
