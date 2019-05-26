# -*- coding: utf-8 -*-
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
# =============================================================================from caffe2.python import workspace, model_helper,core
import numpy as np
from caffe2.proto import caffe2_pb2

def add_inc_conv_c2(net,layer_name,
                        input_name,
                        output_name,
                        weights_name,
                        bias_name,
                        batch_size,
                        dilation,
                        kernel_width,
                        in_channels,
                        out_channels):
    net.external_input.extend([weights_name,bias_name])
    queue_size=kernel_width+(kernel_width-1)*(dilation-1)
    # Create queue for incremental dilated conv stack
    queue = np.full((batch_size,queue_size,in_channels),0).astype(np.float32)
    queue_name=layer_name+"_queue"
    workspace.FeedBlob(queue_name, queue)
    net.external_input.extend([queue_name])
    # Create unique tensor for gather of the weights per layer
    gather_indexes=np.zeros(kernel_width).astype(np.int)
    for i in range(kernel_width):
        gather_indexes[i]=dilation*i
    print(gather_indexes)    
    workspace.FeedBlob(queue_name+"_gather_indexes", gather_indexes)
    net.external_input.extend([queue_name+"_gather_indexes"])
    # re-arrange the conv layer weights to fit the GEMM (this is same as _get_linearized_weight in r9y9 conv.py)
    weights=workspace.FetchBlob(weights_name)
    if weights.shape==(out_channels,in_channels,kernel_width):
        weights=weights.transpose(0,2,1).reshape((out_channels,-1))
        workspace.FeedBlob(weights_name,weights)
    else:
        raise RuntimeError("convolution supplied wheights are not in the expected dims. Layer name"+layer_name)

    # Concat the input to the end of the queue
    ConcatQ = core.CreateOperator(
             'Concat',
             [queue_name,input_name],
             [queue_name+"_concat","info"],
             axis=1,
             )
    # Shift the queue
    SplitQ = core.CreateOperator(
             'Split',
             [queue_name+"_concat"],
             ["scrap",queue_name+"_shifted"],
             split=[1,queue.shape[1]],
             axis=1,
             )
    # Tranpose the queue befor gather operation since gather can only work on axis=0 and we need axis=1
    TransposeQ = core.CreateOperator(
             'Transpose',
             [queue_name+"_shifted"],
             [queue_name+"_shifted_t"],
             axes=[1,0,2],
             )
    # Gather needed weights from queue
    GatherWeights = core.CreateOperator(
             'Gather',
             [queue_name+"_shifted_t",queue_name+"_gather_indexes"],
             [layer_name+"_gemm_input_t"],
             )
    # Re-Tranpose the gathered queue after gather operation to get (batch_size,kernel_size,input_channels)
    ReTransposeQ = core.CreateOperator(
             'Transpose',
             [layer_name+"_gemm_input_t"],
             [layer_name+"_gemm_input"],
             axes=[1,0,2],
             )
    # Re-shape the input for GEMM input (batch_size,-1)
    ReshapeQ = core.CreateOperator(
             'Reshape',
             [layer_name+"_gemm_input"],
             [layer_name+"_gemm_input_reshaped","old_shape"],
             shape=[batch_size,-1],
             )
    # Perform convolution using GEMM
    GEMM = core.CreateOperator(
             'FC',
             [layer_name+"_gemm_input_reshaped",weights_name,bias_name],
             [output_name],
             )
    print_tensor = core.CreateOperator(
                   'Print',
                   [layer_name+"_gemm_input_t"],
                   [],
                   )
    net.op.extend([ConcatQ,SplitQ,TransposeQ,GatherWeights,print_tensor,ReTransposeQ,ReshapeQ,GEMM])
    #net.op.extend([SplitQ,ConcatQ,TransposeQ,GatherWeights,print_tensor])
    return net