name=ssd_resnet34_cloud_native_tf.pb
wget --no-check-certificate https://zenodo.org/record/3246481/files/ssd_resnet34_mAP_20.2.pb?download=1 -O $name

mkdir pretrained
mv $name ./pretrained/$name
