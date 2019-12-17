freeze_graph \
  --input_graph="ssd_resnet34_large.pbtxt" \
  --input_checkpoint="logs_mine_sec.ssd_resnet34_pretrain.no-bn_in_ssd_block_3*3_map.21.1/model.ckpt-111089" \
  --input_binary=false \
  --output_graph="resnet34_tf_21.1.pb" \
  --output_node_names="detection_bboxes,detection_scores,detection_classes"
  
