wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1by8rNAebute_uqI6-t0OSs6HAqWNGOrH' -O  'params_numpy.npy'
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1X-SARKWBQUjh_4BqSYTjkWBMlzjZcHqT'  -O 'seq2cnn_model-symbol.json'
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1GOoAINGDhM5NZYRMwNHin68a1OGD0OmH'  -O 'seq2cnn_model-0000.params'

mkdir pretrained_mxnet
mv 'params_numpy.npy' ./pretrained_mxnet
mv 'seq2cnn_model-symbol.json' ./pretrained_mxnet
mv 'seq2cnn_model-0000.params' ./pretrained_mxnet
