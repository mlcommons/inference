#!/usr/bin/env bash
# Script to test and time DeepSpeech 2 implementation

RANDOM_SEED=1
TARGET_ACC=23

# ARG		CHOICE						    HELP
# ${1} = {batch_1_latency,					Get latency results for batch of 1
#          batching_throughput}				Get slowdowns vs throughput results for select batch sizes

DATASET="libri"
MODELS_DIR="."
SLICE_FLAG=""
FORCE_DURATION=-1
DEVICE="cpu"		# Use "gpu" for gpu
WARMUPS=5
if [ "${1}" = "batch_1_latency" ]
then
	BATCHSIZE=1
	python inference.py \
	    --device ${DEVICE} \
	    --batch_size_val ${BATCHSIZE} \
	    --checkpoint \
	    --continue_from ${MODELS_DIR}/trained_model_deepspeech2.pth \
	    --use_set ${DATASET} \
	    --warmups ${WARMUPS} \
	    --seed $RANDOM_SEED \
	    --force_duration ${FORCE_DURATION} \
	    ${SLICE_FLAG}
fi
if [ "${1}" = "batching_throughput" ]
then    
	python inference.py \
	    --device ${DEVICE} \
	    --batch_size_val 1 \
	    --checkpoint \
	    --continue_from ${MODELS_DIR}/trained_model_deepspeech2.pth \
	    --use_set ${DATASET} \
	    --warmups ${WARMUPS} \
	    --seed $RANDOM_SEED \
	    --force_duration ${FORCE_DURATION} \
	    --batch_1_file none \
	    ${SLICE_FLAG}
	bs=("12" "4" "32" "6" "64" "8" "24" "10" "128" "2" "16")
	for i in "${!bs[@]}"
	do
		bsi="${bs[$i]}"
		fdj="${fd[$j]}"
		echo $bsi
		echo $fdj
		python inference.py \
		    --device ${DEVICE} \
		    --batch_size_val ${bsi} \
		    --checkpoint \
		    --continue_from ${MODELS_DIR}/trained_model_deepspeech2.pth \
		    --use_set ${DATASET} \
		    --warmups ${WARMUPS} \
		    --seed $RANDOM_SEED \
		    --force_duration ${FORCE_DURATION} \
		    --batch_1_file inference_bs1_${DEVICE}.csv \
		    ${SLICE_FLAG}
	done
fi
