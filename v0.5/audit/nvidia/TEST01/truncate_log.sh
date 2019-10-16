# Usage:
# 1) bash ./truncate_log.sh <accuracy_log_file>

#!/bin/bash
log=$1
samples=$2

head -n $((samples + 1)) ${log} > ${log}.new
sed -i '$ s/,$/]/g' ${log}.new
rm ${log}
mv ${log}.new ${log}
