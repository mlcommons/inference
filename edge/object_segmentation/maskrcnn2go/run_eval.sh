DATASET_IM_DIR="path_to_/coco_val2014"
DATASET_ANN="path_to_/coco/instances_minival2014.json"

python code/eval_seg_cpu.py -- \
    --net "model/fp32/model.pb" \
    --init_net "model/fp32/model_init.pb" \
    --dataset "coco_2014_minival" \
    --dataset_dir "$DATASET_IM_DIR" \
    --dataset_ann "$DATASET_ANN" \
    --output_dir output \
    --min_size 320 \
    --max_size 640 \
