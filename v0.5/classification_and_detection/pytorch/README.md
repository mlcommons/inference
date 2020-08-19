# PyTorch SSD-ResNet34

## From: https://github.com/BowenBao/inference/tree/master/cloud/single_stage_detector/pytorch

### Build the project with custom op support
```
cd v0.5/classification_and_detection/pytorch
python setup.py develop --install-dir=.
```
Which should generate `v0.5/classification_and_detection/pytorch/lib/custom_ops.cpython-*-linux-gnu.so`

If you get the error:
```
vision.cpp:7:4: error: 'struct torch::jit::RegisterOperators' has no member named 'op'
```
you are using a newer version of PyTorch and just need to remove the `jit` namespace per https://github.com/pytorch/pytorch/issues/29642#issuecomment-553144431
