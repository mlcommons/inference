#!/bin/bash

echo "Clearing caches."
sync && echo 3 | tee /host_proc/sys/vm/drop_caches


cd /root

common_opt=""
dataset="--dataset-path /data"

start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING RUN AT $start_fmt"

cd /mlperf
python python/main.py $opts $dataset --output /output/results.json

end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING RUN AT $end_fmt"
