freeze_graph \
  --input_graph="resnet34_ssd.pbtxt" \
  --input_checkpoint="logs_mine_sec.pytorch/model.ckpt-87187" \
  --input_binary=false \
  --output_graph="resnet34_tf.22.1.pb" \
  --output_node_names="detection_bboxes,detection_scores,detection_classes,ssd1200/py_cls_pred,ssd1200/py_location_pred"
  
