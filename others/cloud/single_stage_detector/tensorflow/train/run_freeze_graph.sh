python export_graph.py

freeze_graph \
  --input_graph="resnet34_ssd.pbtxt" \
  --input_checkpoint="logs/model.ckpt-109774" \
  --input_binary=false \
  --output_graph="resnet34_tf.pb" \
  --output_node_names="detection_bboxes,detection_scores,detection_classes,ssd1200/py_cls_pred,ssd1200/py_location_pred"
  
