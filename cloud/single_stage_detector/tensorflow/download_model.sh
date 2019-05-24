wget --no-check-certificate https://zenodo.org/record/3060467/files/ssd_resnet-34_from_onnx.zip?download=1  -O ssd_resnet34_from_onnx.zip

mkdir pretrained
mv ssd_resnet34_from_onnx.zip ./pretrained
cd pretrained
unzip ssd_resnet34_from_onnx.zip
cd ..

