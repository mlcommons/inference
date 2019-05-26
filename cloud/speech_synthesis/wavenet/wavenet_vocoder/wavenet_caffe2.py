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
# =============================================================================from __future__ import with_statement, print_function, absolute_import

from wavenet_vocoder.modules_c2 import add_inc_conv_c2,add_wavenet_stack_layer,add_sample_from_discretized_mix_logistic

from caffe2.python import core
from caffe2.proto import caffe2_pb2

def local_conditioning_net(c,upsample_conditional_features,upsample_scales,freq_axis_kernel_size):
    if upsample_conditional_features == False:
        raise RuntimeError("Cant create local conditioning net since upsample_conditional_features == False")
    freq_axis_padding = (freq_axis_kernel_size - 1) // 2
    body_net = caffe2_pb2.NetDef()
    body_net.external_input.extend(['c'])
    Reshape = core.CreateOperator(
            'Reshape',
            ['c'],
            ['unsqueezed_c','old_shape'],
            shape=[1,1,c.shape[1],c.shape[0]],
            )
    print_tensor = core.CreateOperator(
            'Print',
            ['ConvTransposed_0'],
            [],
            )
    body_net.op.extend([Reshape])
    input_name='unsqueezed_c'
    for i,s in enumerate(upsample_scales):
        output_name='ConvTransposed_'+str(i)
        filter_name='upsample_conv.'+str(i)+'.weights'
        bias_name='upsample_conv.'+str(i)+'.biases'
        body_net.external_input.extend([filter_name,bias_name])
        ConvTranspose2d=core.CreateOperator(
                'ConvTranspose',
                [input_name,filter_name,bias_name],
                [output_name],
                kernel_h=freq_axis_kernel_size,
                kernel_w=s,
                stride_h=1,
                stride_w=s,
                #pad=np.array([1,0],dtype=np.int32).transpose(),
                pads=[1,0,1,0],
                )
        input_name=output_name
        output_name='Relu_'+str(i)
        Relu=core.CreateOperator(
                'Relu',
                [input_name],
                [output_name])
        body_net.op.extend([ConvTranspose2d,Relu])
        input_name=output_name
    Squeeze = core.CreateOperator(
            'Squeeze',
            [output_name],
            [output_name],
            dims=[1]
            )
    Transpose = core.CreateOperator(
            'Transpose',
            [output_name],
            [output_name+'_t'],
            axes=[0,2,1]
            )
    body_net.op.extend([Squeeze,Transpose])
    body_net.external_output.extend([output_name+"_t"])
    #body_net.external_output.extend('unsqueezed_c')

    return body_net


def wavenet_net(local_conditioning_net_out_name, 
                initial_input_name,
                const_inputs,
                scalar_input,
                batch_size,
                out_channels,
                residual_channels,
                gate_channels,
                skip_channels,
                cin_channels,
                layers,
                stacks,
                kernel_size,
                ):
    if scalar_input:
        layers_per_stack = layers // stacks
        wavenet_body = caffe2_pb2.NetDef(name="Wavenet_body") # This is the internal loop network which is repeated per each output sample
        wavenet_body.external_input.extend(['i','cond','float_input',"starts_tensor","ends_tensor"])
        wavenet_body.external_input.extend(const_inputs)
        #wavenet_body.external_input.extend([local_conditioning_net_out_name,"starts_tensor","ends_tensor","add_tensor","sqrt2_const","skip_add_out","first_conv.biases"])
        wavenet_body.external_output.extend(["cond","logit_sampling_output","starts_tensor","ends_tensor"])
        wavenet_body.external_output.extend(const_inputs)
        # add the pre-processing part (conv layer)
        created_inputs=[]
        wavenet_body,created_inputs = add_inc_conv_c2(wavenet_body, # Network to construct
                                  "first_conv", # Layer name
                                  'float_input', # input name
                                  "wavenet_pre_out", # output name
                                  "first_conv.weights",
                                  "first_conv.biases",
                                  batch_size,
                                  1, # Dilation
                                  1, # kernel_width
                                  1, # input channels in scalar input
                                  residual_channels,
                                  created_inputs,
                                  )
    else:
        raise RuntimeError("Non-scalar(one-hot) input is currently not supported")
    # crop the current time step from local conditioning vector
    ct_name='local_conditioning_at_timestamp'
    CropLocalConditioning= core.CreateOperator(
            'Slice',
            [local_conditioning_net_out_name,"starts_tensor","ends_tensor"],
            [ct_name],
            )
#    CropLocalConditioning= core.CreateOperator(
#            'Slice',
#            [local_conditioning_net_out_name],
#            [ct_name],
#            starts=[0,0,0],
#            ends=[-1,1,-1],
#            )
    IncrementSliceTime1=core.CreateOperator(
            'Add',
            ["starts_tensor","add_tensor"],
            ["starts_tensor"]
            )
    IncrementSliceTime2=core.CreateOperator(
            'Add',
            ["ends_tensor","add_tensor"],
            ["ends_tensor"]
            )
    wavenet_body.op.extend([CropLocalConditioning,IncrementSliceTime1,IncrementSliceTime2])
    #wavenet_body.op.extend([CropLocalConditioning])
#    for i in range(1):
    for i in range(layers):
        wavenet_body,created_inputs = add_wavenet_stack_layer(wavenet_body,
                                          i, # Layer's index
                                          batch_size,
                                          residual_channels,
                                          gate_channels,
                                          kernel_size,
                                          skip_channels,
                                          cin_channels,
                                          2**(i % layers_per_stack), # conv dilation
                                          ct_name,
                                          created_inputs,
                                          )

    # Add the post processing layers
    PostRelu1= core.CreateOperator(
            'Relu',
            ["skip_add_out"],
            ["skip_add_out"],
            )
    wavenet_body.op.extend([PostRelu1])
    wavenet_body,created_inputs = add_inc_conv_c2(wavenet_body,
                                  "postconv1",
                                  "skip_add_out",
                                  "postconv1_out",
                                  "postconv1.weights",
                                  "postconv1.biases",
                                  batch_size,
                                  1, # Dilation
                                  1, # kernel_width
                                  skip_channels, # input channels
                                  skip_channels, # output channels
                                  created_inputs,
                                  ) 
    PostRelu2= core.CreateOperator(
            'Relu',
            ["postconv1_out"],
            ["postconv1_out"],
            )
    wavenet_body.op.extend([PostRelu2])
    wavenet_body,created_inputs = add_inc_conv_c2(wavenet_body,
                                  "postconv2",
                                  "postconv1_out",
                                  "postconv2_out",
                                  "postconv2.weights",
                                  "postconv2.biases",
                                  batch_size,
                                  1, # Dilation
                                  1, # kernel_width
                                  skip_channels, # input channels
                                  out_channels,  # output channels
                                  created_inputs,
                                  )
    testing=False
    if testing:
        calcOut = core.CreateOperator(
                'Slice',
                ['postconv2_out'],
                ['logit_sampling_output'],
                starts=[0,0,0],
                ends=[1,1,-1],
                )
        reshapeOut = core.CreateOperator(
                'Reshape',
                ['logit_sampling_output'],
                ['logit_sampling_output',"old_shape"],
                shape=[10,3]
                )
        wavenet_body.op.extend([calcOut,reshapeOut])
    else:
        wavenet_body = add_sample_from_discretized_mix_logistic(
            wavenet_body,
            "postconv2_out",
            batch_size,
            out_channels,
            log_scale_min=-20.0,
            )
    wavenet_body.external_output.extend(["logit_sampling_output"])
    while_op = core.CreateOperator(
        'ONNXWhile',
        ['max_trip_count', 'condition', initial_input_name,"starts_tensor","ends_tensor"]+const_inputs+created_inputs,
        ['new_input',"starts_tensor","ends_tensor"]+const_inputs+created_inputs+["testit"],
        body=wavenet_body,
        has_cond=False,
        has_trip_count=True,
        save_scopes=0,
    )

    wavenet = caffe2_pb2.NetDef(name="Wavenet")
    wavenet.external_input.extend(['max_trip_count','condition',initial_input_name,"starts_tensor","ends_tensor"])
    wavenet.external_input.extend(const_inputs+created_inputs)
    wavenet.op.extend([while_op])
    wavenet.external_output.extend(['new_input',"starts_tensor","ends_tensor"])
    wavenet.external_output.extend(const_inputs+created_inputs)
    wavenet.external_output.extend(["testit"])
    

    return wavenet,wavenet_body
