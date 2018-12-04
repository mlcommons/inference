import time
import sys
import argparse

import numpy as np
import torch
import torch.onnx
import onnx
import platform
import caffe2.python.onnx.backend as backend

from dataset.data_loader import AudioDataLoader, SpectrogramDataset
from utils import *
import params

print(platform.python_version())
print(torch.__version__)

# Import Data Utils
sys.path.append('../')
print("FORCE CPU...")

params.cuda = False


def convert(args):

    make_folder(args.save_folder)

    labels = get_labels(params)
    audio_conf = get_audio_conf(params)

    val_batch_size = min(8, params.batch_size_val)
    print("Using bs={} for validation. Parameter found was {}".format(val_batch_size, params.batch_size_val))

    train_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=params.train_manifest, labels=labels,
                                       normalize=True, augment=params.augment)
    train_loader = AudioDataLoader(train_dataset, batch_size=params.batch_size,
                                   num_workers=(1 if params.cuda else 1))

    model = get_model(params)

    if args.continue_from:
        print("Loading checkpoint model %s" % args.continue_from)
        package = torch.load(args.continue_from)
        model.load_state_dict(package['state_dict'])
        if params.cuda:
            model = model.cuda()

    if params.cuda:
        model = torch.nn.DataParallel(model).cuda()

    print(model)
    print("Number of parameters: %d" % DeepSpeech.get_param_size(model))

    # Begin ONNX conversion
    model.train(False)
    # Input to the model
    data = next(iter(train_loader))
    inputs, targets, input_percentages, target_sizes = data
    inputs = torch.Tensor(inputs, requires_grad=False)

    if params.cuda:
        inputs = inputs.cuda()

    x = inputs
    print("input has size:{}".format(x.size()))

    # Export the model
    onnx_file_path = osp.join(osp.dirname(args.continue_from), osp.basename(args.continue_from).split('.')[0] + ".onnx")
    print("Saving new ONNX model to: {}".format(onnx_file_path))
    torch.onnx.export(model,  # model being run
                      inputs,  # model input (or a tuple for multiple inputs)
                      onnx_file_path,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      verbose=False)


def onnx_inference(args):
    # Load the ONNX model
    model = onnx.load("models/deepspeech_{}.onnx".format(args.continue_from))

    # Check that the IR is well formed
    onnx.checker.check_model(model)

    onnx.helper.printable_graph(model.graph)

    print("model checked, preparing backend!")
    rep = backend.prepare(model, device="CPU")  # or "CPU"

    print("running inference!")

    # Hard coded input dim
    inputs = np.random.randn(16, 1, 161, 129).astype(np.float32)

    start = time.time()
    outputs = rep.run(inputs)
    print("time used: {}".format(time.time() - start))
    # To run networks with more than one input, pass a tuple
    # rather than a single numpy ndarray.
    print(outputs[0])


if __name__ == "__main__":
    # Comand line arguments, handled by params except seed
    parser = argparse.ArgumentParser(description='DeepSpeech training')
    parser.add_argument('--checkpoint', dest='checkpoint',
                        action='store_true', help='Enables checkpoint saving of model')
    parser.add_argument('--save_folder', default='models/',
                        help='Location to save epoch models')
    parser.add_argument('--model_path', default='models/deepspeech_final.pth.tar',
                        help='Location to save best validation model')
    parser.add_argument('--continue_from', default='',
                        help='Continue from checkpoint model')
    args = parser.parse_args()

    convert(args)
    print("=======finished converting models!========")
    onnx_inference(args)
    print("=======finished checking onnx models!==========")
