#!/bin/bash

data_dir=${DATA_DIR:-./nmt/data}
model_dir=${MODEL_DIR:-./ende_gnmt_model_4_layer}
output_root_dir=${OUTPUT_DIR:-./nmt/data}
output_dir=$output_root_dir/closed/mlperf-org/
scenario=${scenario:-SingleStream}
mode=${mode:-performance}
debug_settings=${debug_settings:-}
store_translation=${store_translation:-}
system_id="tf-gpu"

if [ ! -d "./nmt/data" ]; then
    ln -s $data_dir ./nmt
fi

if [ ! -d "./ende_gnmt_model_4_layer" ]; then
    ln -s $model_dir ./ende_gnmt_model_4_layer
fi

if [ "$mode" = "performance" ]; then
  save_path=${output_dir}/results/$system_id/gmnt/${scenario,,}/$mode/run_1
else
  save_path=${output_dir}/results/$system_id/gmnt/${scenario,,}/$mode
fi

mkdir -p $save_path

time python loadgen_gnmt.py --verbose \
    --scenario=$scenario \
    $store_translation \
    $debug_settings \
    --mode=$mode \
    --batch_size=$batch_size \
    --output_path=$save_path

cp -t $save_path ./mlperf_log_*

# Create system specs file
bash ./create_system_file.sh $system_id

# setup the measurements directory
mdir=$output_dir/measurements/$system_id/gnmt/$scenario
mkdir -p $mdir

# reference app uses command line instead of user.conf
echo "# empty" > $mdir/user.conf
touch $mdir/README.md
impid="reference"
cat > $mdir/$system_id"_reference_"$scenario".json" <<EOF
{
    "input_data_types": "fp32",
    "retraining": "none",
    "starting_weights_filename": "https://zenodo.org/record/2530924/files/gnmt_model.zip",
    "weight_data_types": "fp32",
    "weight_transformations": "none"
}
EOF

# Create code reference folder and run submission checker
code_dir=${output_dir}/code/gnmt/reference
mkdir -p $code_dir
echo "Finishing generating files..."
{
    git clone https://github.com/mlcommons/inference.git
    pushd ./inference
    cp ./mlperf.conf $mdir
    echo "git clone https://github.com/mlcommons/inference.git" > $code_dir/VERSION.txt
    git rev-parse HEAD >> $code_dir/VERSION.txt
    python tools/submission/submission-checker.py --input ${output_root_dir} > ${output_dir}/submission-checker.log 2>&1 
    cat ${output_dir}/submission-checker.log
    popd
} &> /dev/null
echo "DONE!"