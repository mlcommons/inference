wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1by8rNAebute_uqI6-t0OSs6HAqWNGOrH' -O  'params_numpy.npy'
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1EHhfizSCc10uillXW-mAXk7lulum56zr'  -O 'seq2cnn_imported_mxnet.pth'

mkdir pretrained_pyt
mv 'params_numpy.npy' ./pretrained_pyt
mv 'seq2cnn_imported_mxnet.pth' ./pretrained_pyt

