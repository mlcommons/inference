#!/usr/bin/env bash
# Script to test and time DeepSpeech 2 implementation

RANDOM_SEED=1
TARGET_ACC=23

# ARG		CHOICE						    HELP
# ${1} = {Integer >= 0}						Looks for a model at new_training/deepspeech_${1}.pth.tar
# ${2} = {libri, ov (coming soon)}			Chooses with dataset to use	for testing			
# ${3} = {Integer >= 0}						To use a homogenous dataset consisting of chiefly inputs of the same audio duration, set >=0. Else use -1 to ignore. 
# ${4} = float							    The duration of the held input's audio clip in seconds used to normalize the timing stats. Set to 1 for no effect.
# ${5} = {Integer >= -1}					Max number of trials(batches) to infer. Causes the test script to infer the first n batches of the selected dataset. Set to -1 for no limit.

EPOCH=20	 	    #${1}
DATASET="libri" 	#${2}
HOLD_IDX=-1		    #${3}
HOLD_SEC=1.0		#${4}
N_TRIALS=-1		    #${5}
BATCHSIZE=1
MODELS_DIR="."
# By default we use GPU and batch size of 40
python inference.py \
    --cpu \
    --batch_size_val ${BATCHSIZE} \
    --checkpoint \
    --continue_from ${MODELS_DIR}/trained_model_deepspeech2.pth \
    --use_set ${DATASET} \
    --seed $RANDOM_SEED \
    --hold_idx ${HOLD_IDX} \
    --hold_sec ${HOLD_SEC} \
    --n_trials ${N_TRIALS}
