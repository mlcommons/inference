# coding=utf-8
# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors.
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

# Modified by SCC23 UCSD Zixian Wang, Nov 11, 2023

import array
import json
import os
import sys
sys.path.insert(0, os.getcwd())
import psutil

import mlperf_loadgen as lg
import numpy as np
import onnxruntime
from transformers import BertConfig, BertForQuestionAnswering
from squad_QSL import get_squad_QSL


# For multi-gpu runs
import multiprocessing # not sure about this? 
import threading
import random

# time
import time 



global_batch_size = 1




class MyThreadOrtValue(threading.Thread):

    def __init__(self, sess_i, btch):
        threading.Thread.__init__(self)
        self.sess_i = sess_i
        self.btch = btch
        self.q = []
        # self.ort_device = get_ort_device_from_session(self.sess)
        # self.ort_device = onnxcustom.utils.onnxruntime_helper.get_ort_device ('gpu')

    def run(self):
        # ort_device = self.ort_device
        sess_to_run = self.sess_i        # Could be bug
        q = self.q

        # Run this session with session's output and sub-batch 
        # not too sure about get_outputs()

        # print ('batch content: ')
        # print ('batch content: ')
        # print ('batch content: ')
        # print ('batch content: ')
        # print ('batch content: ')
        # print (self.btch)

        
        out = sess_to_run.run([o.name for o in sess_to_run.get_outputs()], self.btch)

        # print ("out: ")
        # print (out)
        q.append (out)

        # print (onnxruntime.InferenceSession.get_provider_options())

        # for img in self.imgs:
        #     ov = C_OrtValue.ortvalue_from_numpy(img, ort_device)
        #     out = sess.run_with_ort_values(
        #         {input_name: ov}, [output_name], None)[0]
        #     q.append(out.numpy())



def parallel_ort_value (sess_s, big_batch, batch_size): 
    # ------------------------------
    
    # Create sub_batch for each GPU
    
    # ------------------------------
    lst_sub_fd = [] 
    sub_batch_size = batch_size // len(sess_s)
    for i in range (len (sess_s)): 
        # batch starting and ending indices 
        sub_batch_start = i * sub_batch_size
        sub_batch_end = sub_batch_start + sub_batch_size
    
        # Extract sub-batch
        sub_fd = {k: v[sub_batch_start:sub_batch_end] for k, v in big_batch.items()}
        # Append each sub-batch 
        lst_sub_fd.append (sub_fd)

    # ------------------------------
    
    # Create thread for each GPU and sub_batch 
    
    # ------------------------------
    n_threads = len (sess_s)
    # Create thread for each GPU
    threads = [] 
    for i, sess_i in enumerate (sess_s): 
        # sess_i.set_providers(providers=['ROCMExecutionProvider', "CPUExecutionProvider"], provider_options=[{"device_id":i}, {}])
        threads.append (MyThreadOrtValue (sess_i, lst_sub_fd[i]))  # [MyThreadOrtValue (sess_i, lst_sub_fd[i]) for i, sess_i in enumerate (sess_s)]
        # print (sess_i)
        # print (sess_i.get_provider_options())

    # ------------------------------
    
    # run each thread and wait for all the finish 
    
    # ------------------------------

    # print ("threads len: ", len (threads))
    for t in threads: 
        # print (t)
        t.start()
    res = []
    for t in threads: 
        t.join ()
        res.extend (t.q)

    return res 


class BERT_ONNXRuntime_SUT():
    def __init__(self, args):

        # Keep track of starting time 
        # init_start_time = time.time()

        
        self.profile = args.profile
        self.options = onnxruntime.SessionOptions()
        self.options.enable_profiling = args.profile
        self.options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Please change the value according to best setting in Performance Test Tool result.
        self.options.intra_op_num_threads=psutil.cpu_count(logical=True)

        print("Loading ONNX model...")
        self.quantized = args.quantized


        # Handle different scenarios
        self.scenario = args.scenario

        #### Script for setting up batch_size based on user input
        global global_batch_size 
        global_batch_size = args.batch_size

        model_path = os.environ.get("ML_MODEL_FILE_WITH_PATH")
        if not model_path:
            if self.quantized:
                # model_path = "build/data/bert_tf_v1_1_large_fp32_384_v2/bert_large_v1_1_fake_quant.onnx"

                # model_path = "build/data/bert_tf_v1_1_large_fp32_384_v2/bert-base-cased-squad_opt_gpu_fp16.onnx"
                # model_path = "/home/ziw081/onnxruntime/onnxruntime/python/tools/transformers/notebooks/onnx/bert-base-cased-squad_opt_gpu_fp16_v2.onnx"
                model_path = "/home/ziw081/onnxruntime/onnxruntime/python/tools/transformers/notebooks/onnx/bert-base-cased-squad_opt_gpu_fp16_v3.onnx"

            else:
                # model_path = "build/data/bert_tf_v1_1_large_fp32_384_v2/model.onnx"
                model_path = "/home/ziw081/onnxruntime/onnxruntime/python/tools/transformers/notebooks/onnx/bert-base-cased-squad_opt_gpu_fp32_v3.onnx"
                # model_path = "build/data/bert_tf_v1_1_large_fp32_384_v2/bertsquad-10.onnx"
        if len(onnxruntime.get_all_providers()) > 1 and os.environ.get("USE_GPU", "yes").lower() not in [ "0", "false", "off", "no" ]:
            #Currently considering only CUDAExecutionProvider
            # self.sess = onnxruntime.InferenceSession(model_path, self.options, providers=['CUDAExecutionProvider'])

            print ("")
            print ("Inside rocmExecution provider")
            print ("Detected providers: ")
            print (onnxruntime.get_all_providers())
            
            
            # self.sess = onnxruntime.InferenceSession(model_path, self.options, providers=['ROCMExecutionProvider'])


            # Trying to do multi-gpu InferenceSession
            self.gpu_num = args.gpu_num
            starting_device = args.gpu_device
            ending_device = starting_device + self.gpu_num
            self.sess = []
            for i in range (starting_device, ending_device, 1): 
                self.sess.append (
                    onnxruntime.InferenceSession (model_path, self.options, providers=['ROCMExecutionProvider', "CPUExecutionProvider"], provider_options=[{"device_id": i}, {}])
                )
                print(f"Initialize device {i}")
                print (self.sess)

                print ("printing output: ")
                print (self.sess[0].get_outputs())
                
            

            print ("After rocmExecution provider")
            print (self.sess)
        
        else:
            print ('------------------------------')
            print ('------------------------------')
            print ('cpu only runs: ')
            print ('------------------------------')
            print ('------------------------------')
            self.sess = onnxruntime.InferenceSession(model_path, self.options, providers=["CPUExecutionProvider"])

        

        # Sleep for 1 minute
        # time.sleep (60)

        # init_elapsed_time = time.time() - init_start_time
        # print ("Time to initialize all variables are: ")
        # print (init_elapsed_time)




        
        # Keep track of starting time 
        # sut_start_time = time.time()
        
        print("Constructing SUT...")
        print("Constructing SUT...")
        print("Constructing SUT...")
        print("Constructing SUT...")
        print("Constructing SUT...")
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        print("Finished constructing SUT.")

        # sut_elapsed_time = time.time() - sut_start_time
        # print ("Time to run loagen Constrct sut: ")
        # print (init_elapsed_time)


        

        self.qsl = get_squad_QSL(args.max_examples)




    


    
    def issue_queries(self, query_samples):

        
        # Sleep for 1 minute
        # time.sleep (60)

        print ('query_samples len is')
        print (len (query_samples))
        print (query_samples[0])
        
        # Batch size
        batch_size = global_batch_size  # Choose your desired batch size
        print ('batchsize is ', batch_size)

        print (self.sess)
        
        responses = [] 
        max_seq_length = 384
        
        # Loop over batches
        for start_idx in range(0, len(query_samples), batch_size):

            
            # print (start_idx)
            
            # Creating BIG batch
            
            end_idx = start_idx + batch_size
            # if (end_idx )
            batch_samples = query_samples[start_idx:end_idx]
            
            # 1. Extract features for the current batch of query_samples.
            
            input_ids, input_masks, segment_ids = [], [], []
            # input_ids, input_masks, segment_ids = np.array ([],dtype=np.int32), np.array ([],dtype=np.int32), np.array ([],dtype=np.int32)
            # input_ids = np.zeros((batch_size, max_seq_length), dtype=np.int32)
            # input_masks = np.zeros((batch_size, max_seq_length), dtype=np.int32)
            # segment_ids = np.zeros((batch_size, max_seq_length), dtype=np.int32)
            
            for index, sample in enumerate (batch_samples):

                # Some way to convert all features all at once instead of one at a time in each 
                eval_features = self.qsl.get_features(sample.index)

                
                input_ids.append(eval_features.input_ids)
                input_masks.append(eval_features.input_mask)
                segment_ids.append(eval_features.segment_ids)

                # input_ids [index,:] = (eval_features.input_ids)
                # input_masks [index,:] = (eval_features.input_mask)
                # segment_ids [index,:] = (eval_features.segment_ids)

            
             
    
             # 2. Convert lists to numpy arrays and adjust shape for batch processing.
            # print ("input_ids.shape")
            # print (len (input_ids))
            # print (len (input_ids[0]))
            # print (len (input_ids[0][0]))
            
            
            input_ids = np.array(input_ids).astype(np.int64)
            input_masks = np.array(input_masks).astype(np.int64)
            segment_ids = np.array(segment_ids).astype(np.int64)  # Outputs error if not int64

            # input_ids = np.array(input_ids).astype(np.float16)
            # input_masks = np.array(input_masks).astype(np.float16)
            # segment_ids = np.array(segment_ids).astype(np.float16) 

            # input_ids = np.array(input_ids).astype(np.int32)
            # input_masks = np.array(input_masks).astype(np.int32)
            # segment_ids = np.array(segment_ids).astype(np.int32)

    
            # 3. Prepare the feature dictionary.
            if self.quantized:
                
            
                fd = {
                    "input_ids": input_ids,
                    # "attention_mask": input_masks,
                    # "token_type_ids": segment_ids
                    "input_mask": input_masks,
                    "segment_ids": segment_ids
                    
                    
                    # Debug for unquantized model with quantized flag
                    # "input_ids": input_ids,
                    # "input_mask": input_masks,
                    # "segment_ids": segment_ids
                }
            else:
                fd = {
                    "input_ids": input_ids,
                    "input_mask": input_masks,
                    "segment_ids": segment_ids
                }

            # if self.quantized:
            #     fd = {
            #         "input_ids": input_ids,
            #         "attention_mask": input_masks,
            #         "token_type_ids": segment_ids
            #     }
            # else:
            #     fd = {
            #         "unique_ids_raw_output___9:0": np.array ([0]),
            #         "input_ids:0": input_ids,
            #         "input_mask:0": input_masks,
            #         "segment_ids:0": segment_ids
            #     }

            # # ------------------------------
            
            # # Create sub_batch for each GPU
            
            # # ------------------------------
            # lst_sub_fd = [] 
            # sub_batch_size = batch_size // len(self.sess)
            # for i in range (len (self.sess)): 
            #     # batch starting and ending indices 
            #     sub_batch_start = i * sub_batch_size
            #     sub_batch_end = sub_batch_start + sub_batch_size
            
            #     # Extract sub-batch
            #     sub_fd = {k: v[sub_batch_start:sub_batch_end] for k, v in fd.items()}
            #     # Append each sub-batch 
            #     lst_sub_fd.append (sub_fd)

            # # ------------------------------
            
            # # Create thread for each GPU and sub_batch 
            
            # # ------------------------------
            # n_threads = len (self.sess)
            # # Create thread for each GPU
            # threads = [MyThreadOrtValue (sess_i, lst_sub_fd[i]) for i, sess_i in enumerate (self.sess)]




            
            # # ------------------------------
            
            # # run each thread and wait for all the finish 
            
            # # ------------------------------

            # print ("threads len: ", len (threads))
            # for t in threads: 
            #     t.start()
            # res = []
            # for t in threads: 
            #     t.join ()
            #     res.extend (t.q)

            # self.InferenceSession.get_provider_options()


            # if (start_idx == 0): 
            
            if ((self.scenario == 'Server') or (self.scenario == 'SingleStream') or (self.scenario == 'MultiStream')): 
                ind = random.randint(0, self.gpu_num-1)
                scores = parallel_ort_value ([self.sess[ind]], fd, batch_size)
            else: # self.scenario = 'Offline'
                scores = parallel_ort_value (self.sess, fd, batch_size)
            
            # print ('receiving results from each gpu')
            # print ('res.shape: ', len (scores))
            # print ('res[0].shape: ', len (scores[0]))
            # print ('res[0][0].shape: ', len (scores[0][0]))
    
            # # 4. Run the inference for the current batch.
            # scores = self.sess.run([o.name for o in self.sess.get_outputs()], fd)
            # # outputs = scores[0]
            # outputs = np.stack(scores, axis=-1)

            # outputs = np.stack(scores, axis=-1)
            # print ("scores")
            # print (scores)
            # a
            outputs = [np.stack (output, axis=-1) for output in scores]
            
            # print (outputs)
            outputs = np.vstack (outputs)
            
            # print ("output's shape", len (outputs))
            # print ("output's[0] shape", len (outputs[0]))
            # print ("output's[0][0] shape", len (outputs[0][0]))

            # 5. Convert batched outputs to responses.
            for i, output in enumerate(outputs):

                # print (output.shape)
                
                response_array = array.array("B", output.tobytes())
                bi = response_array.buffer_info()
                response = lg.QuerySampleResponse(batch_samples[i].id, bi[0], bi[1])
                # responses.append(response)
                lg.QuerySamplesComplete([response])

        
        

    def flush_queries(self):
        pass

    def __del__(self):
        if self.profile:
            print("ONNX runtime profile dumped to: '{}'".format(self.sess.end_profiling()))
        print("Finished destroying SUT.")

def get_onnxruntime_sut(args):
    return BERT_ONNXRuntime_SUT(args)
