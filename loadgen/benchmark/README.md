Note: please install jemalloc first. See: http://jemalloc.net/
Command: bash run.sh <target_qps> <0=Basic,1=Queue> <numCompleteThreads> <maxSizeInComplete>

Experiments:
- On Intel(R) Xeon(R) CPU E5-1650 v4 @ 3.60GHz
- Basic SUT : 500-600k i/s
- Basic SUT + jemalloc: 800-900k i/s (`bash run.sh 800000 0`)
- Queued SUT (2 complete threads) + jemalloc: 1.2-1.3M i/s (`bash run.sh 1200000 1 2 2048`)
