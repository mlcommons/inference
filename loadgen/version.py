# Creates version files used by the loadgen at compile time.

import datetime
import errno
import os
import sys

from absl import app

def generate_loadgen_version_header(header_filename):
    gitRev = os.popen("git rev-parse --short=10 HEAD").read()
    gitCommitDate = os.popen("git log --format=\"%cI\" -n 1").read()
    gitStatus = os.popen("git status -s -uno").read()
    gitLog = os.popen("git log --pretty=oneline -n 16 --no-decorate").read()
    dateTimeNowLocal = datetime.datetime.now().isoformat()
    dateTimeNowUtc = datetime.datetime.utcnow().isoformat()

    try:
        os.makedirs(os.path.dirname(header_filename))
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise

    file = open(header_filename, "w")
    file.write("namespace mlperf {\n\n")
    file.write("constexpr char kLoadgenVersion[] = \".5a1\";\n")
    file.write("constexpr char kLoadgenGitRevision[] = \"" + gitRev[0:-1] + "\";\n")
    file.write("constexpr char kLoadgenBuildDateLocal[] = \"" + dateTimeNowLocal + "\";\n")
    file.write("constexpr char kLoadgenBuildDateUtc[]   = \"" + dateTimeNowUtc + "\";\n")
    file.write("constexpr char kLoadgenGitCommitDate[]  = \"" + gitCommitDate[0:-1] + "\";\n")
    file.write("constexpr char kLoadgenGitStatus[] =\nR\"(" + gitStatus[0:-1] + ")\";\n")
    file.write("constexpr char kLoadgenGitLog[] =\nR\"(" + gitLog[0:-1] + ")\";\n")
    file.write("\n}  // namespace mlperf\n");
    file.close()


def main(argv):
    if len(argv) > 2:
        raise app.UsageError('Too many command-line arguments.')
    generate_loadgen_version_header(argv[1])


if __name__ == '__main__':
  app.run(main)
