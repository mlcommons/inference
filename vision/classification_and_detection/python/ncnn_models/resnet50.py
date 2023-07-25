import numpy as np
import ncnn


class Resnet50:
    def __init__(self, model_param, model_bin, target_size=224, num_threads=1, use_gpu=False):
        self.target_size = target_size
        self.num_threads = num_threads

        self.net = ncnn.Net()

        self.net.load_param(model_param)
        self.net.load_model(model_bin)

    def __del__(self):
        self.net = None
    
    @property
    def input_name(self):
        return "in0"
    
    @property
    def output_name(self):
        return "out0"

    def __call__(self, img):
        mat_in = ncnn.Mat(img)

        ex = self.net.create_extractor()
        # ex.set_num_threads(self.num_threads)

        ex.input(self.input_name, mat_in)

        ret, mat_out = ex.extract(self.output_name)

        # manually call softmax on the fc output
        # convert result into probability
        # skip if your model already has softmax operation
        softmax = ncnn.create_layer("Softmax")

        pd = ncnn.ParamDict()
        softmax.load_param(pd)

        softmax.forward_inplace(mat_out, self.net.opt)

        mat_out = mat_out.reshape(mat_out.w * mat_out.h * mat_out.c)

        cls_scores = np.array(mat_out)

        return [np.array([cls_scores])]
