#!/bin/bash

#######################################################################
# Batch Conversion Script
#
# This script executes a series of conversion scripts with specified
# data types (fp8 and int8) for different models.
#
# Usage:
#   bash ./ci_scripts/backward_compatibility_test/run_all_conversions.sh
#
# Author: [Your Name]
# Date: [Current Date]
#######################################################################

echo "Starting batch conversion..."

##############################
# Convert QLV4 Generate Scripts
##############################

# # Convert for llama3-8b
# echo -e "\nRunning convert_qlv4_generate_llama3-8b.sh..."
# bash ./ci_scripts/backward_compatibility_test/convert_qlv4_generate_llama3-8b.sh

# # Convert for bert with fp8 and int8
# for dtype in fp8 int8; do
#     echo -e "\nRunning convert_qlv4_generate_bert.sh with dtype=$dtype..."
#     bash ./ci_scripts/backward_compatibility_test/convert_qlv4_generate_bert.sh $dtype
# done

# # Convert for gptj with fp8 and int8
# for dtype in fp8 int8; do
#     echo -e "\nRunning convert_qlv4_generate_gptj.sh with dtype=$dtype..."
#     bash ./ci_scripts/backward_compatibility_test/convert_qlv4_generate_gptj.sh $dtype
# done


##############################
# Convert QLV4 Scripts for Other Models
##############################

# Models and data types
models=("llama3-8b" "llama3-70b" "llama2-70b")
dtypes=("fp8" "int8")

# Loop through models and data types
for model in "${models[@]}"; do
    for dtype in "${dtypes[@]}"; do
        echo -e "\nRunning convert_qlv4_${model}.sh with dtype=$dtype..."
        bash ./ci_scripts/backward_compatibility_test/convert_qlv4_${model}.sh $dtype
    done
done

echo -e "\nBatch conversion completed successfully."
