# DLRM TF1 Implementation

DLRM TF1 Implementation

## Usage

To run inference on an evaluation set:

```sh
python3 [DLRM code directory]/dlrm_main.py \
--batch_size=65536 \
--nobfloat16_grads_all_reduce \
--data_dir=gs://dmchen-data/dlrm_tf1_data/criteo \
--decay_start_step=38000 \
--decay_steps=40000 \
--dim_embed=128 \
--noenable_profiling \
--noenable_summary \
--eval_batch_size=65536 \
--eval_steps=1362 \
--gcp_project=[ID of your GCP project] \
--learning_rate=0.7 \
--lr_warmup_steps=2000 \
--master=[Name of your cloud TPU] \
--mlp_bottom=512,256,128 \
--mlp_top=1024,1024,512,256,1 \
--model_dir=gs://dmchen-data/dlrm_tf1_data/model_dir_0 \
--num_dense_features=13 \
--num_tables_in_ec=13 \
--num_tpu_shards=128 \
--optimizer=sgd \
--nopipeline_execution \
--replicas_per_host=8 \
--restore_checkpoint \
--nosave_checkpoint \
--sleep_after_init=60 \
--steps_between_evals=0 \
--tpu_zone=europe-west4-a \
--train_steps=0 \
--use_batched_tfrecords \
--nouse_cached_data \
--nouse_synthetic_data \
--vocab_sizes_embed=39884406,39043,17289,7420,20263,3,7120,1543,63,38532951,2953546,403346,10,2208,11938,155,4,976,14,39979771,25641295,39664984,585935,12972,108,36
```

A model checkpoint is available in this public GCS bucket:
gs://dmchen-data/dlrm_tf1_data/model_dir_0.

The Criteo dataset is stored in a format compatible with the model in this
public GCS bucket:
gs://dmchen-data/dlrm_tf1_data/criteo.

The model checkpoint and Criteo dataset files can be accessed with the
[gsutil tool](https://cloud.google.com/storage/docs/gsutil).

```sh
gsutil cp -r gs://dmchen-data/dlrm_tf1_data/model_dir_0 [your folder]
```
