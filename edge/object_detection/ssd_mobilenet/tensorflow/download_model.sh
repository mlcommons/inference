dir=$(pwd)
mkdir /ssd_model
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz -P /ssd_model
tar zxvf /ssd_model/ssd_mobilenet_v1_coco_2017_11_17.tar.gz -C /ssd_model
rm /ssd_model/ssd_mobilenet_v1_coco_2017_11_17.tar.gz
cd $dir
