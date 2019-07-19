freeze_graph \
  --input_graph="resnet34_ssd.pbtxt" \
  --input_checkpoint="shuffle_ckpt/model.ckpt-shuffleweights" \
  --input_binary=false \
  --output_graph="resnet34_tf.21.1.pb" \
  --output_node_names="detection_bboxes,detection_scores,detection_classes"
  
