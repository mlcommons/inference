# Mixtral reference standalone inference script

The reference output and accuracy can be checked using the standalone hugginface inference script following the instructions below:

```
cd language/mixtral-8x7b
docker build -t mlc-ngc .
nvidia-docker run -it --rm --net=host --runtime=nvidia --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --cap-add=SYS_PTRACE --cap-add=SYS_ADMIN --cap-add=DAC_READ_SEARCH --security-opt seccomp=unconfined -w $PWD -v $PWD:$PWD -t mlc-ngc

pip install -r requirements.txt
cd standalone_infer
# Make sure the checkpoint and reference pickle file is already downloaded
python3 hf_eval_all.py --input_pkl=09292024_mixtral_15k_mintoken2_v1.pkl --checkpoint_path=/raid/data/mlperf-llm/Mixtral-8x7B-Instruct-v0.1 --output_pkl=mixtral_8x7b_15000_greedy_reference_fp16_mintoken2.pkl --batch_size=64

# Exit the container and enter the evaluation container
exit
docker build . -f Dockerfile.eval -t evaluation
docker run -it --rm --net=host --runtime=nvidia --ipc=host -v $PWD:$PWD -w $PWD evaluation
cd standalone_infer
python3 run_accuracy.py --results_path=mixtral_8x7b_15000_greedy_reference_fp16_mintoken2.pkl
```

Expected output:
```
EM: 0.7366, correct: 3683 / 5000, gen_token_per_sample: 129.9604
Evaluating OpenOrca score...
OpenOrca score: {'rouge1': np.float64(45.5989), 'rouge2': np.float64(23.3526), 'rougeL': np.float64(30.4608), 'rougeLsum': np.float64(42.5396)}, gen_token_per_sample: 205.8656
Evaluating MBXP score...
100%|| 5000/5000 [02:33<00:00, 32.50it/s]
Processed 5000 in 153.89411109898356s
 60.16% pass@1
{'cpp': 381, 'typescript': 438, 'ruby': 419, 'python': 492, 'php': 809, 'javascript': 469}  out of  {'cpp': 743, 'typescript': 868, 'ruby': 846, 'python': 863, 'php': 846, 'javascript': 834}
gen_tokens_per_sample: 98.7026
```
