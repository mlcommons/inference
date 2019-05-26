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
# =============================================================================
from caffe2.python import workspace,core
import numpy as np

def add_inc_conv_c2(net,
                    layer_name,
                    input_name,
                    output_name,
                    weights_name,
                    bias_name,
                    batch_size,
                    dilation,
                    kernel_width,
                    in_channels,
                    out_channels,
                    created_inputs):
    net.external_input.extend([weights_name,bias_name])
    net.external_output.extend([weights_name,bias_name])
    created_inputs=created_inputs+[weights_name,bias_name]
    if kernel_width>1:
        queue_size=kernel_width+(kernel_width-1)*(dilation-1)
        # Create queue for incremental dilated conv stack
        queue = np.full((batch_size,queue_size,in_channels),0).astype(np.float32)
        queue_name=layer_name+"_queue"
        workspace.FeedBlob(queue_name, queue)
        net.external_input.extend([queue_name])
        net.external_output.extend([queue_name])
        created_inputs=created_inputs+[queue_name]
        # Create unique tensor for gather of the weights per layer
        gather_indexes=np.zeros(kernel_width).astype(np.int32)
        for i in range(kernel_width):
            gather_indexes[i]=dilation*i
        workspace.FeedBlob(queue_name+"_gather_indexes", gather_indexes)
        net.external_input.extend([queue_name+"_gather_indexes"])
        net.external_output.extend([queue_name+"_gather_indexes"])
        created_inputs=created_inputs+[queue_name+"_gather_indexes"]
    # re-arrange the conv layer weights to fit the GEMM (this is same as _get_linearized_weight in r9y9 conv.py)
    weights=workspace.FetchBlob(weights_name)
    if weights.shape==(out_channels,in_channels,kernel_width):
        weights=weights.transpose(0,2,1).reshape((out_channels,-1))
        workspace.FeedBlob(weights_name,weights)
    else:
        raise RuntimeError("convolution supplied wheights are not in the expected dims. Layer name"+layer_name)

    if kernel_width>1:
#        CreateQueueIndexesTensor = core.CreateOperator(
#                'GivenTensorIntFill',
#                [],
#                [queue_name+"_gather_indexes"],
#                values=gather_indexes,
#                shape=[kernel_width]
#                )
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
#                 ["scrap",queue_name+"_shifted"],
                 ["scrap",queue_name],
                 split=[1,queue.shape[1]],
                 axis=1,
                 )
#        CopyQueue = core.CreateOperator(
#                'Copy',
#                 [queue_name+"_shifted"],
#                 [queue_name],
#                )
        # Tranpose the queue befor gather operation since gather can only work on axis=0 and we need axis=1
        TransposeQ = core.CreateOperator(
                 'Transpose',
#                 [queue_name+"_shifted"],
                 [queue_name],
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
                 [layer_name+"_gemm_out"],
                 )
        #net.external_output.extend([queue_name+"_shifted_t"])
    else:
        GEMM = core.CreateOperator(
                 'FC',
                 [input_name,weights_name,bias_name],
                 [layer_name+"_gemm_out"],
                 )
            
    # Perform convolution using GEMM
    GEMMOutReshape = core.CreateOperator(
             'Reshape',
             [layer_name+"_gemm_out"],
             [output_name,"old_shape"],
             shape=[batch_size,1,-1],
             )
    #print_tensor = core.CreateOperator(
    #               'Print',
    #               ['i'],
    #               [],
    #               )
    if kernel_width>1:
#        net.op.extend([ConcatQ,SplitQ,CopyQueue,TransposeQ,GatherWeights,ReTransposeQ,ReshapeQ,GEMM,GEMMOutReshape])
        net.op.extend([ConcatQ,SplitQ,TransposeQ,GatherWeights,ReTransposeQ,ReshapeQ,GEMM,GEMMOutReshape])
    else:
        net.op.extend([GEMM,GEMMOutReshape])
        
    #net.op.extend([SplitQ,ConcatQ,TransposeQ,GatherWeights,print_tensor])

    return net,created_inputs

def add_wavenet_stack_layer(net,
                            layer_index, # index of the layer to add this will be usedd for naming
                            batch_size,
                            residual_channels,
                            gate_channels,
                            kernel_size,
                            skip_channels,
                            cin_channels,
                            dilation,
                            ct_name, # local conditioning vector at time=t
                            created_inputs,
                            ):
    if layer_index==0:
        input_name="wavenet_pre_out"
    else:
        input_name="conv_"+str(layer_index-1)+"_out"
    dconv_layer_name="dilationconv_"+str(layer_index)
    net,created_inputs=add_inc_conv_c2(net,
                   dconv_layer_name,
                   input_name,
                   dconv_layer_name+"_out",
                   "conv_layers."+str(layer_index)+".dconv.weights", # weights tensor name
                   "conv_layers."+str(layer_index)+".dconv.biases", # biases tensor name
                   batch_size,
                   dilation, # conv dilation
                   kernel_size,
                   residual_channels,
                   gate_channels,
                   created_inputs,
                   )

    lc_layer_name="localcondconv_"+str(layer_index) # local conditioning convolution
    net,created_inputs=add_inc_conv_c2(net,
                        lc_layer_name,
                        ct_name,
                        lc_layer_name+"_out",
                        "conv_layers."+str(layer_index)+".lcconv.weights", # weights tensor name
                        "conv_layers."+str(layer_index)+".lcconv.biases", # biases tensor name
                        batch_size,
                        1, # conv dilation
                        1, # conv kernel size
                        cin_channels, # input channels
                        gate_channels, # output channels
                        created_inputs,
                        )
    AddLocalCond=core.CreateOperator(
            'Add',
            [dconv_layer_name+"_out",lc_layer_name+"_out"],
            ["dconv_lc_add_out"+str(layer_index)],
            )
    net.op.extend([AddLocalCond])
    SplitExpertGate = core.CreateOperator(
            'Split',
            ["dconv_lc_add_out"+str(layer_index)],
            ["experts"+str(layer_index),"gates"+str(layer_index)],
            split=[gate_channels//2,gate_channels//2],
            axis=-1
            )
    net.op.extend([SplitExpertGate])
    TanhExperts = core.CreateOperator(
            'Tanh',
            ["experts"+str(layer_index)],
            ["tanhexperts"+str(layer_index)])
    SigmoidGates = core.CreateOperator(
            'Sigmoid',
            ["gates"+str(layer_index)],
            ["sigmoidgates"+str(layer_index)])
    expertsmix_name="ExpertsMix"+str(layer_index)
    ElMul = core.CreateOperator(
            'Mul',
            ["tanhexperts"+str(layer_index),"sigmoidgates"+str(layer_index)],
            [expertsmix_name])
    net.op.extend([TanhExperts,SigmoidGates,ElMul])
    
    skipconv_layer_name="skipconv_"+str(layer_index) # local conditioning convolution
    net,created_inputs=add_inc_conv_c2(net,
                        skipconv_layer_name,
                        expertsmix_name,
                        skipconv_layer_name+"_out",
                        "conv_layers."+str(layer_index)+".skipconv.weights", # weights tensor name
                        "conv_layers."+str(layer_index)+".skipconv.biases", # biases tensor name
                        batch_size,
                        1, # conv dilation
                        1, # conv kernel size
                        gate_channels // 2, # input channels
                        skip_channels, # output channels
                        created_inputs,
                        )
    outconv_layer_name="outconv_"+str(layer_index) # local conditioning convolution
    net,created_inputs=add_inc_conv_c2(net,
                        outconv_layer_name,
                        expertsmix_name,
                        outconv_layer_name+"_out",
                        "conv_layers."+str(layer_index)+".outconv.weights", # weights tensor name
                        "conv_layers."+str(layer_index)+".outconv.biases", # biases tensor name
                        batch_size,
                        1, # conv dilation
                        1, # conv kernel size
                        gate_channels // 2, # input channels
                        residual_channels, # output channels
                        created_inputs,
                        )
    AddResidual = core.CreateOperator(
            'Add',
            [outconv_layer_name+"_out",input_name],
            ["conv_"+str(layer_index)+"_addout"],
            )

    MulSqrt = core.CreateOperator(
            'Mul',
            ["conv_"+str(layer_index)+"_addout","sqrt2_const"],
            ["conv_"+str(layer_index)+"_out"],
            broadcast=1,
            )
#    if layer_index==0:
#        CreateSkipSum = core.CreateOperator(
#                'ConstantFill',
#                [skipconv_layer_name+"_out"],
#                ["skip_add_out"],
#                value=0.0,
#                )
#    SkipAdd = core.CreateOperator(
#            'Add',
#            [skipconv_layer_name+"_out","skip_add_out"],
#            ["skip_add_out"],
#            )

    if layer_index==0:
        SkipAdd = core.CreateOperator(
                'Copy',
                [skipconv_layer_name+"_out"],
                ["skip_add_out"],
                )
    else:            
        SkipAdd = core.CreateOperator(
                'Add',
                [skipconv_layer_name+"_out","skip_add_out"],
                ["skip_add_out"],
                )
        
#    if layer_index==0:
#        net.op.extend([AddResidual,MulSqrt,CreateSkipSum,SkipAdd])
#    else:
    net.op.extend([AddResidual,MulSqrt,SkipAdd])
    if layer_index!=0:
        SkipMulSqrt = core.CreateOperator(
                'Mul',
                ["skip_add_out","sqrt2_const"],
                ["skip_add_out"],
                broadcast=1,
                )
        net.op.extend([SkipMulSqrt])

    return net,created_inputs
    
def add_sample_from_discretized_mix_logistic(
        net,
        input_name,
        batch_size,
        out_channels,
        log_scale_min=-7.0,
        ):
    assert out_channels % 3 == 0
    logits_num=out_channels//3
    SelectLogitUniformVector=core.CreateOperator(
            'UniformFill',
            [],
            ["logit_selection_vector"],
            min=1e-5,
            max=1.0 - 1e-5,
            shape=[batch_size,1,logits_num]
            )
    LogSLV=core.CreateOperator(
            'Log',
            ["logit_selection_vector"],
            ["logit_selection_vector"],
            )
#    CreateNegMat=core.CreateOperator(
#            'ConstantFill',
#            ["logit_selection_vector"],
#            ["minus_1"],
#            value=-1.0,
#            )
#    NegSLV=core.CreateOperator(
#            'Mul',
#            ["logit_selection_vector","minus_1"],
#            ["logit_selection_vector"],
#            )
    
    NegSLV=core.CreateOperator(
            'Negative',
            ["logit_selection_vector"],
            ["logit_selection_vector"],
            )
    LogSLV2=core.CreateOperator(
            'Log',
            ["logit_selection_vector"],
            ["logit_selection_vector"],
            )
    ReshapeProbs=core.CreateOperator(
            'Reshape',
            [input_name],
            ["logit_probs_reshaped","old_shape"],
            shape=[batch_size,1,3,logits_num],
            )
    GetProbs=core.CreateOperator(
            'Slice',
            ["logit_probs_reshaped"],
            ["logit_probs"],
            starts=[0,0,0,0],
            ends=[-1,-1,1,-1],
            )
    SqueezeProbs=core.CreateOperator(
            'Squeeze',
            ["logit_probs"],
            ["logit_probs"],
            dims=[1]
            )
    SubUniform = core.CreateOperator(
            'Sub',
            ["logit_probs","logit_selection_vector"],
            ["logit_selection_vector"],
            )
    GetStats=core.CreateOperator(
            'Slice',
            ["logit_probs_reshaped"],
            ["logit_stats"],
            starts=[0,0,1,0],
            ends=[-1,-1,3,-1],
            )
    SqueezeStats=core.CreateOperator( # This will give us a tensor of shape 2xlogits_num !!!!!! removing axis 0 assumes batchsize=1!!!!Dans
            'Squeeze',
            ["logit_stats"],
            ["logit_stats"],
            dims=[0,1]
            )
    ArgMax=core.CreateOperator( # This will be of size batch_size x 3
            'ArgMax',
            ["logit_selection_vector"],
            ["argmax"],
            axis=2,
            )
    squeezeArgMax=core.CreateOperator( # This will give us a tensor of shape Batch_size
            'Squeeze',
            ["argmax"],
            ["argmax"],
            dims=[0]
            )
    transposeForGather=core.CreateOperator( # This will give us a tensor of logits_numx2 in order to gather on outer dim
            'Transpose',
            ["logit_stats"],
            ["logit_stats_t"],
            axes=[1,0]
            )
    gatherMaxprob=core.CreateOperator( # This will give us a tensor of 1x1x2 in order to gather on outer dim
                 'Gather',
                 ["logit_stats_t","argmax"],
                 ["selected_stats"],
                 )
    squeezeGather=core.CreateOperator( # This will give us a tensor of 1x2
            'Squeeze',
            ["selected_stats"],
            ["selected_stats"],
            dims=[0]
            )
    print_tensor = core.CreateOperator(
            'Print',
            ['logit_selection_vector'],
            [],
            )
    print_tensor2 = core.CreateOperator(
            'Print',
            ['logit_stats_t'],
            [],
            )
    print_tensor3 = core.CreateOperator(
            'Print',
            ['selected_stats'],
            [],
            )
    GetMeans=core.CreateOperator(
            'Slice',
            ["selected_stats"],
            ["logit_means"],# logit_sampling_output,logit_means
            starts=[0,0],
            ends=[-1,1],
            )
    GetLogScales=core.CreateOperator(
            'Slice',
            ["selected_stats"],
            ["logit_logscales"],
            starts=[0,1],
            ends=[-1,2],
            )
    ClipLogScales=core.CreateOperator(
            'Clip',
            ["logit_logscales"],
            ["logit_logscales"],
            min=log_scale_min
            )
    Logits_u=core.CreateOperator(
            'UniformFill',
            [],
            ["logits_u"],
            min=1e-5,
            max=1.0 - 1e-5,
            shape=[batch_size,1]
            )
    Logit=core.CreateOperator(
            'Logit',
            ["logits_u"],
            ["logits_u"],
            )
# The below ops will be in place of caffe2 logit which is out=log(x/(1-x))    
    LogitNeg=core.CreateOperator(
            'Negative',
            ["logits_u"],
            ["logits_u_neg"],
            )
    LogitNegAdd=core.CreateOperator(
            'Add',
            ["logits_u_neg","one_const"],
            ["logits_u_neg"],
            broadcast=1,
            )
    LogitDiv=core.CreateOperator(
            'Div',
            ["logits_u","logits_u_neg"],
            ["logits_u"],
            )
    LogitLog=core.CreateOperator(
            'Log',
            ["logits_u"],
            ["logits_u"],
            )
# End - logit ops    
    ExpLogScale=core.CreateOperator(
            'Exp',
            ["logit_logscales"],
            ["exp_logscale"],
            )
    MulExpLogit=core.CreateOperator(
            'Mul',
            ["logits_u","exp_logscale"],
            ["mul_exp_logit"],
            )
    CalcOut=core.CreateOperator(
            'Add',
            ["mul_exp_logit","logit_means"],
            ["logit_sampling_output"]
            )
    
#    net.op.extend([SelectLogitUniformVector,LogSLV,CreateNegMat,NegSLV,LogSLV2,ReshapeProbs,GetProbs,SqueezeProbs,SubUniform])
    net.op.extend([SelectLogitUniformVector,LogSLV,NegSLV,LogSLV2,ReshapeProbs,GetProbs,SqueezeProbs,SubUniform])
    net.op.extend([GetStats,SqueezeStats,ArgMax,squeezeArgMax,transposeForGather,gatherMaxprob,squeezeGather,GetMeans,GetLogScales,ClipLogScales])
    net.op.extend([Logits_u,LogitNeg,LogitNegAdd,LogitDiv,LogitLog,ExpLogScale,MulExpLogit,CalcOut])
#    net.op.extend([Logits_u,Logit,ExpLogScale,MulExpLogit,CalcOut])
    return net
    
