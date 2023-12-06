docker build . -t llm/gpubringup
nvidia-docker run -it --rm --net=host --runtime=nvidia --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --cap-add=SYS_PTRACE --cap-add=SYS_ADMIN --cap-add=DAC_READ_SEARCH \
  --security-opt seccomp=unconfined \
  -e HISTFILE=/home/scratch.alicheng_sw/skritch/tmp/bash_histories/.tekit \
  --env-file /home/scratch.alicheng_sw/mlpinf/experiments/llm/docker_env \
  -w $PWD \
  --mount type=bind,source=/raid/data,target=/raid/data \
  --mount type=bind,source=/home/scratch.alicheng_sw,target=/home/scratch.alicheng_sw \
  --mount type=bind,source=/home/mlperf_inference_data,target=/home/mlperf_inference_data \
  llm/gpubringup \
  bash /home/scratch.alicheng_sw/mlpinf/inference/language/llama2-70b/with_the_same_user
