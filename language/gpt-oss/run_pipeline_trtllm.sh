#!/bin/bash
#SBATCH --output slurm_logs/run_%j/stdout.txt

output_dir=slurm_logs/run_$SLURM_JOBID

srun_header="srun \
        --container-mounts=$(pwd)/../:/work,/lustre/share/coreai_mlperf_inference/mlperf_inference_storage_clone:/home/mlperf_inference_storage/ \
        --container-name=trtllm_gptoss_2 \
        --container-mount-home --container-remap-root --container-workdir /work/gpt-oss"

set -x

$srun_header --container-image ./sqsh_files/trtllm_with_nettools.sqsh --output slurm_logs/run_$SLURM_JOBID/server_output.log ./run_server_trtllm.sh --model_path /home/mlperf_inference_storage/models/gpt-oss/gpt-oss-120b --output_dir $output_dir &

sleep 20

$srun_header --overlap /bin/bash -c '
  while ! netstat -tuln | grep -q ":30000 .*LISTEN"; do
    sleep 5
  done
'

$srun_header --overlap /bin/bash -c '
  while ! netstat -tuln | grep -q ":30007 .*LISTEN"; do
    sleep 5
  done
'

sleep 20

$srun_header --overlap python3 run_infer_trtllm.py \
	--input-tokens data/accuracy_eval_tokenized.pkl \
	--output data/accuracy_eval_inferred_trtllm_job-$SLURM_JOBID-nongreedy_temp1_top-p1.pkl \
	--max-tokens 32768 \
	--server-url localhost:30000,localhost:30001,localhost:30002,localhost:30003,localhost:30004,localhost:30005,localhost:30006,localhost:30007 \
	--max-concurrency 2048 \
	--pass-k 5 \
	--temperature 1.0 \
	--top-p 1.0 --top-k 0 --timeout 2400
