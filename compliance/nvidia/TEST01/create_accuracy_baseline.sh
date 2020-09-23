# Usage:
# 1) bash ./create_accuracy_baseline.sh <accuracy_accuracy_log_file> <perf_accuracy_log_file>
# 2) python inference/v0.5/translation/gnmt/tensorflow/process_accuracy.py <perf_accuracy_log_file>
# 3) python inference/v0.5/translation/gnmt/tensorflow/process_accuracy.py on generated baseline
# 4) Compare BLEU scores

#!/bin/bash
accuracy_log=$1
perf_log=$2
patterns="unique_patterns.txt"
accuracy_baseline=$(basename -- "$accuracy_log")
accuracy_baseline="${accuracy_baseline%.*}"_baseline.json

cut -d ':' -f 2,3 ${perf_log} | cut -d ',' -f 2- | sort | uniq | grep qsl > ${patterns}
echo '[' > ${accuracy_baseline}
grep -f ${patterns} ${accuracy_log} >> ${accuracy_baseline}
sed -i '$ s/,$//g' ${accuracy_baseline}
echo ']' >> ${accuracy_baseline}
rm ${patterns}
echo "Created a baseline accuracy file: ${accuracy_baseline}"
