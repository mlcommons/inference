# coding: utf-8
# Copyright 2018 The MLPerf Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""
Synthesis waveform from trained WaveNet.

usage: synthesis.py [options] <checkpoint>

options:
    --length=<T>                      Steps to generate. If not specified will default to local conditioning length.
    --conditional=<p>                 Conditional features path.
    --symmetric-mels                  Symmetric mel.
    --max-abs-value=<N>               Max abs value [default: -1].
    -h, --help               Show help message.
"""

import time

from docopt import docopt

import torch

import numpy as np
import librosa

from wavenet_vocoder.wavenet_caffe2 import local_conditioning_net,wavenet_net

from caffe2.python import workspace

import math 
import time

torch.set_num_threads(4)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def _to_numpy(x):
    # this is ugly
    if x is None:
        return None
    if isinstance(x, np.ndarray) or np.isscalar(x):
        return x
    # remove batch axis
    if x.dim() == 3:
        x = x.squeeze(0)
    return x.numpy()



if __name__ == "__main__":
    args = docopt(__doc__)
    print("Command line args:\n", args)
    checkpoint_path = args["<checkpoint>"]
    length=args["--length"]
    if (length):
        length = int(length)
    conditional_path = args["--conditional"]
    # From https://github.com/Rayhane-mamah/Tacotron-2
    symmetric_mels = args["--symmetric-mels"]
    max_abs_value = float(args["--max-abs-value"])

    # Networks hyper parameters
    num_mels = 80
    upsample_scales = [4,4,4,4]
    num_layers = 24
    upsample_conditional_features = True
    freq_axis_kernel_size = 3
    out_channels = 30 
    residual_channels = 512
    gate_channels = 512
    skip_out_channels = 256
    cin_channels = 80
    num_stacks = 4
    kernel_size = 3
    sample_rate = 22050


    # Load conditional features
    if conditional_path is not None:
        c = np.load(conditional_path)
        if c.shape[1] != num_mels:
            np.swapaxes(c, 0, 1)
        if max_abs_value > 0:
            min_, max_ = 0, max_abs_value
            if symmetric_mels:
                min_ = -max_
            print("Normalize features to desired range [0, 1] from [{}, {}]".format(min_, max_))
            c = np.interp(c, (min_, max_), (0, 1))
    else:
        c = None
    c=c.astype(np.float32)
    #from train import build_model

    # Model
    #model = build_model().to(device)

    # Load checkpoint
    print("Load checkpoint from {}".format(checkpoint_path))
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

    #load input to workspace
    workspace.ResetWorkspace()
    batch_size=1
    c_transposed=c.swapaxes(0,1)
    workspace.FeedBlob("c", c_transposed)
    scalar_input=True
    if scalar_input:
        initial_input=np.full((batch_size,1,1),0).astype(np.float32) # Shape is (batch_size,1,input_channels)
    else:
        raise RuntimeError("Non-scalar(one-hot) input is currently not supported")
    workspace.FeedBlob("initial_input",initial_input)
    # helper blobs for local conditioning slicing at current timestamp
    starts_tensor=np.full((3),(0,0,0)).astype(np.int32)
    ends_tensor=np.full((3),(-1,1,-1)).astype(np.int32)
    add_tensor=np.full((3),(0,1,0)).astype(np.int32)
    workspace.FeedBlob("starts_tensor",starts_tensor)
    workspace.FeedBlob("ends_tensor",ends_tensor)
    workspace.FeedBlob("add_tensor",add_tensor)
    one_const=np.full((1),1).astype(np.float32)
    workspace.FeedBlob("one_const",one_const)
    sqrt2_const=np.full((1),(math.sqrt(0.5))).astype(np.float32)
    workspace.FeedBlob("sqrt2_const",sqrt2_const)
    local_conditioning_size=c.shape[0]
    for i in upsample_scales:
        local_conditioning_size=local_conditioning_size*i
    local_conditioning_size=local_conditioning_size-1
    if (length):
        local_conditioning_size=length
    max_trip_count = np.full(1,local_conditioning_size).astype(np.int64)
    condition = np.full(1,True).astype(np.bool)
    workspace.FeedBlob("max_trip_count", max_trip_count)
    workspace.FeedBlob("condition", condition)
    
    #load pytorch weights and biases to workspace:
    # 1) Pre-processing parameters
    pytorch_params_to_load=[['upsample_conv.0.weight_v','upsample_conv.0.weights'],
                            ['upsample_conv.0.bias','upsample_conv.0.biases'],
                            ['upsample_conv.2.weight_v','upsample_conv.1.weights'],
                            ['upsample_conv.2.bias','upsample_conv.1.biases'],
                            ['upsample_conv.4.weight_v','upsample_conv.2.weights'],
                            ['upsample_conv.4.bias','upsample_conv.2.biases'],
                            ['upsample_conv.6.weight_v','upsample_conv.3.weights'],
                            ['upsample_conv.6.bias','upsample_conv.3.biases'],
                            ['first_conv.weight_v','first_conv.weights'],
                            ['first_conv.bias','first_conv.biases'],
                            ['last_conv_layers.1.weight_v','postconv1.weights'],
                            ['last_conv_layers.1.bias','postconv1.biases'],
                            ['last_conv_layers.3.weight_v','postconv2.weights'],
                            ['last_conv_layers.3.bias','postconv2.biases'],
                            ]
    # 2) Dilation layers parameters
    for i in range(num_layers):
        pytorch_dconv_weights_name="conv_layers."+str(i)+".conv.weight_v"
        c2_dconv_weights_name="conv_layers."+str(i)+".dconv.weights"
        pytorch_dconv_bias_name="conv_layers."+str(i)+".conv.bias"
        c2_dconv_biases_name="conv_layers."+str(i)+".dconv.biases"
        pytorch_params_to_load=pytorch_params_to_load+[[pytorch_dconv_weights_name,c2_dconv_weights_name]]
        pytorch_params_to_load=pytorch_params_to_load+[[pytorch_dconv_bias_name,c2_dconv_biases_name]]
        pytorch_lcconv_weights_name="conv_layers."+str(i)+".conv1x1c.weight_v"
        c2_lcconv_weights_name="conv_layers."+str(i)+".lcconv.weights"
        pytorch_lcconv_bias_name="conv_layers."+str(i)+".conv1x1c.bias"
        c2_lcconv_biases_name="conv_layers."+str(i)+".lcconv.biases"
        pytorch_params_to_load=pytorch_params_to_load+[[pytorch_lcconv_weights_name,c2_lcconv_weights_name]]
        pytorch_params_to_load=pytorch_params_to_load+[[pytorch_lcconv_bias_name,c2_lcconv_biases_name]]
        pytorch_outconv_weights_name="conv_layers."+str(i)+".conv1x1_out.weight_v"
        c2_outconv_weights_name="conv_layers."+str(i)+".outconv.weights"
        pytorch_outconv_bias_name="conv_layers."+str(i)+".conv1x1_out.bias"
        c2_outconv_biases_name="conv_layers."+str(i)+".outconv.biases"
        pytorch_params_to_load=pytorch_params_to_load+[[pytorch_outconv_weights_name,c2_outconv_weights_name]]
        pytorch_params_to_load=pytorch_params_to_load+[[pytorch_outconv_bias_name,c2_outconv_biases_name]]
        pytorch_skipconv_weights_name="conv_layers."+str(i)+".conv1x1_skip.weight_v"
        c2_skipconv_weights_name="conv_layers."+str(i)+".skipconv.weights"
        pytorch_skipconv_bias_name="conv_layers."+str(i)+".conv1x1_skip.bias"
        c2_skipconv_biases_name="conv_layers."+str(i)+".skipconv.biases"
        pytorch_params_to_load=pytorch_params_to_load+[[pytorch_skipconv_weights_name,c2_skipconv_weights_name]]
        pytorch_params_to_load=pytorch_params_to_load+[[pytorch_skipconv_bias_name,c2_skipconv_biases_name]]
   
    const_inputs=[]
    for param in pytorch_params_to_load:
        if "_v" in param[0]:
                v=checkpoint['state_dict'][param[0]]
                g=checkpoint['state_dict'][param[0][:-2]+"_g"]
                output_size = (v.size(0),) + (1,) * (v.dim() - 1)
                norm=v.contiguous().view(v.size(0), -1).norm(dim=1).view(*output_size)
                a=v*g/norm
        else:
                a=checkpoint['state_dict'][param[0]]
        workspace.FeedBlob(param[1], a.numpy())
        #const_inputs=const_inputs+[param[1]]
        
    local_conditioning_net_out_name='Relu_3_t'
    local_conditioning_at_timestamp_name="current_local_conditioning"
    const_inputs=const_inputs+[local_conditioning_net_out_name]
    const_inputs=const_inputs+["add_tensor","one_const","sqrt2_const"]
#    const_inputs=const_inputs+["add_tensor","one_const","sqrt2_const","skip_add_out"]


    # Create network
    local_conditioning_net=local_conditioning_net(c,
                                                  upsample_conditional_features,
                                                  upsample_scales,
                                                  freq_axis_kernel_size)
    wavenet_net,body_net=wavenet_net(local_conditioning_net_out_name,
                         'initial_input',
                         const_inputs,
                         scalar_input,
                         batch_size, #batch size
                         out_channels,
                         residual_channels,
                         gate_channels,
                         skip_out_channels,
                         cin_channels,
                         num_layers,
                         num_stacks,
                         kernel_size)
        
    # Run the network
    workspace.RunNetOnce(local_conditioning_net)
    out=workspace.FetchBlob('Relu_3_t')
    print("Shape of conditioning vector: "+str(out.shape))
    
    print("Starting network to generate "+str(length)+ " Samples. This can take some time (~70 samples per sec)....")
    start = time.time()
    workspace.RunNetOnce(wavenet_net)
    end = time.time()
    print("Total run time:",end - start)
    testit=workspace.FetchBlob('testit')
    
    
    #save
    output_filename="out/test.wav"
    librosa.output.write_wav(output_filename, testit, sr=sample_rate)
    print("Resulting wave sample at: "+output_filename)
