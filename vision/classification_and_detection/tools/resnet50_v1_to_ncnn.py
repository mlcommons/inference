import os
import torch
import torchvision.models as models
from PIL import Image


net = models.resnet50(pretrained=True)
net.eval()

torch.manual_seed(0)
x = torch.rand(1, 3, 224, 224)

a = net(x)

# export torchscript
mod = torch.jit.trace(net, x)
mod.save("resnet50_v1.pt")

# torchscript to pnnx
# install ncnn and pnnx to have this working the official docs are well documented
# to help with the installation of ncnn and pnnx
os.system("pnnx resnet50_v1.pt inputshape=[1,3,224,224] fp16=0")

# pnnx inference
import resnet50_v1_pnnx
b = resnet50_v1_pnnx.test_inference()

# ncnn inference
import numpy as np
import ncnn
with ncnn.Net() as net:
    net.opt.use_fp16_packed = False
    net.opt.use_fp16_storage = False
    net.opt.use_fp16_arithmetic = False
    net.load_param("resnet50_v1.ncnn.param")
    net.load_model("resnet50_v1.ncnn.bin")

    with net.create_extractor() as ex:
        ex.input("in0", ncnn.Mat(x.squeeze(0).numpy()).clone())

        _, out0 = ex.extract("out0")
        c = torch.from_numpy(np.array(out0)).unsqueeze(0)

print(a)
print(b)
print(c)
