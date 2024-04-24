# coding=utf-8
# Copyright 2021 Arm Limited and affiliates.
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

# Modified by SCC23 UCSD Zixian Wang, Nov 12, 2023

import array
import json
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "DeepLearningExamples", "PyTorch", "LanguageModeling", "BERT"))
sys.path.insert(0, os.getcwd())

import mlperf_loadgen as lg
import numpy as np
import torch
import transformers
from transformers import BertConfig, BertForQuestionAnswering
from squad_QSL import get_squad_QSL
from torch.utils.data import Dataset, DataLoader


global_batch_size = 1



class LG_Dataset(Dataset):
    def __init__(self, query_samples, qsl):
        self.query_samples = query_samples
        self.qsl = qsl

    def __len__(self):
        return len(self.query_samples)

    def __getitem__(self, idx):
        # sample = self.query_samples[idx]

        # # print ("Sample size is ", sample.index)
        
        # features = self.qsl.get_features(sample.index)

        # print ("features.input_ids: ", len (features.input_ids))

        sample = self.query_samples[idx]
        features = self.qsl.get_features(sample.index)

        # Ensure that the returned data points are in the correct shape
        # input_ids = torch.tensor(features.input_ids)
        # input_mask = torch.tensor(features.input_mask)
        # segment_ids = torch.tensor(features.segment_ids)
        
        return features.input_ids, features.input_mask, features.segment_ids, sample.id

def custom_collate_fn(batch):
    # Each entry in the batch is a tuple of (input_ids, attention_masks, segment_ids, sample_id)
    # We'll separate these out, and batch them along the 0-th dimension.
    input_ids = [item[0] for item in batch]
    attention_masks = [item[1] for item in batch]
    segment_ids = [item[2] for item in batch]
    sample_ids = [item[3] for item in batch]

    return input_ids, attention_masks, segment_ids, sample_ids



class BERT_PyTorch_SUT():
    
    
    def __init__(self, args):
        print("Loading BERT configs...")
        with open("bert_config.json") as f:
            config_json = json.load(f)
        
        config = BertConfig(
            attention_probs_dropout_prob=config_json["attention_probs_dropout_prob"],
            hidden_act=config_json["hidden_act"],
            hidden_dropout_prob=config_json["hidden_dropout_prob"],
            hidden_size=config_json["hidden_size"],
            initializer_range=config_json["initializer_range"],
            intermediate_size=config_json["intermediate_size"],
            max_position_embeddings=config_json["max_position_embeddings"],
            num_attention_heads=config_json["num_attention_heads"],
            num_hidden_layers=config_json["num_hidden_layers"],
            type_vocab_size=config_json["type_vocab_size"],
            vocab_size=config_json["vocab_size"])
       
        # Testing for multi gpu runs  
        available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
        # available_gpus = torch.cuda.device_count()
        print ("Devices found below: ")
        print (available_gpus)

        #### Script for setting up batch_size based on user input
        global global_batch_size 
        global_batch_size = args.batch_size
        

        #### Script for setting devices based on user input
        
        print ("")
        print ("Requested ", args.gpu_num, " gpus")

        gpu_id_list = [] 
        if (args.gpu_num == 0): 
            self.dev = torch.device("cpu")
            print ('Device is on ', self.dev)
        elif (not torch.cuda.is_available()): 
            raise CustomError("No gpu found in device, but requesting gpu usages")
        else: 
            gpu_num = args.gpu_num 
            # gpu_device = 8 - gpu_num 
            gpu_device = args.gpu_device
            gpu_id_list = [gpu_device + i for i in range (gpu_num)]

            self.dev = torch.device ("cuda:"+str(gpu_device))
            print ("gpu devices requested: ", gpu_id_list)
            print ("device starting with ", self.dev)
            
        
        # self.dev = torch.device("cuda:4") if torch.cuda.is_available() else torch.device("cpu")
        
        print (self.dev)
        # ------------------------------------------


        self.version = transformers.__version__

        print("Loading PyTorch model...")
        self.model = BertForQuestionAnswering(config)


        
        # # Quantizing the model
        # self.model.half()  # ---------------- This has dramatic impact on the model accuracy ---------------

        
        
        # Testing for multi gpu runs
        if (args.gpu_num != 0): 
            self.model = torch.nn.DataParallel(self.model,device_ids = gpu_id_list)
        
        
        self.model.to(self.dev)
        self.model.eval()
        model_file = os.environ.get("ML_MODEL_FILE_WITH_PATH", "build/data/bert_tf_v1_1_large_fp32_384_v2/model.pytorch")
        self.model.load_state_dict(torch.load(model_file), strict=False)

        print("Constructing SUT...")
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        print("Finished constructing SUT.")

        self.qsl = get_squad_QSL(args.max_examples)







    

    def issue_queries(self, query_samples):
        
        
        # Batch size
        batch_size = global_batch_size  # Choose your desired batch size
        
        # print ('Requested batch_size is: ', batch_size)


        dataset = LG_Dataset(query_samples, self.qsl)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn, num_workers=2)
        
        
        with torch.no_grad():
            print ("The size of query_samples is: ")
            print (len (query_samples))

            # for i in range(0,len(query_samples), batch_size):

            for input_ids, attention_masks, segment_ids, sample_ids in dataloader:

# #                print (i)
#                 # Trying to create a batch
                # if (i+batch_size >= len (query_samples)):
                #     batched_samples = query_samples[i:len (query_samples)-1]
                # else:
                #     batched_samples = query_samples[i:i+batch_size]

#                 # Trying to create a batch
#                 batched_samples = query_samples[i:i+batch_size]


#                 # Extract features for each sample in the batch
                # input_ids = [self.qsl.get_features(sample.index).input_ids for sample in batched_samples]
                # attention_masks = [self.qsl.get_features(sample.index).input_mask for sample in batched_samples]
                # segment_ids = [self.qsl.get_features(sample.index).segment_ids for sample in batched_samples]

                # input_ids = [self.qsl.get_features(sample.index).input_ids for sample in batched_samples]
                # attention_masks = [self.qsl.get_features(sample.index).input_mask for sample in batched_samples]
                # segment_ids = [self.qsl.get_features(sample.index).segment_ids for sample in batched_samples]

                
                # input_ids = np.stack (input_ids, axis=-1)
                # attention_masks = np.stack (attention_masks, axis=-1)
                # segment_ids = np.stack (segment_ids, axis=-1)
                
                # print ("input_ids: ", len (input_ids))
                # print ("input_ids: ", len (input_ids[0]))
                # print ("input_ids: ", type (input_ids[0]))
                
                # Convert lists to tensors
                # input_ids = input_ids.tolist()
                input_ids_tensor = torch.LongTensor(input_ids).to(self.dev)
                attention_mask_tensor = torch.LongTensor(attention_masks).to(self.dev)
                segment_ids_tensor = torch.LongTensor(segment_ids).to(self.dev)


                # input_ids_tensor = input_ids.to(self.dev)
                # attention_mask_tensor = attention_masks.to(self.dev)
                # segment_ids_tensor = segment_ids.to(self.dev)
                
                # # Convert lists to tensors
                # input_ids_tensor = torch.LongTensor(input_ids).to(self.dev)
                # attention_mask_tensor = torch.LongTensor(attention_masks).to(self.dev)
                # segment_ids_tensor = torch.LongTensor(segment_ids).to(self.dev)


                # No longer need the feature for only 1 feature: 
                # eval_features = self.qsl.get_features(query_samples[i].index)
                
                
                # No longer need this model output for 1 input
                #model_output = self.model.forward(input_ids=torch.LongTensor(eval_features.input_ids).unsqueeze(0).to(self.dev),
                #    attention_mask=torch.LongTensor(eval_features.input_mask).unsqueeze(0).to(self.dev),
                #    token_type_ids=torch.LongTensor(eval_features.segment_ids).unsqueeze(0).to(self.dev))
                


                # print ("Before forward")
                
                # print (input_ids_tensor)
                # print (type (input_ids_tensor))


                try: 
                    # Get model output for the entire batch
                    # Feed forward
                    model_output = self.model.forward(
                        input_ids=input_ids_tensor,
                        attention_mask=attention_mask_tensor,
                        token_type_ids=segment_ids_tensor
                    )
                except Exception as e:
                    print (f"An error occurred: {e}")

 #               if (i==0): 
 #                   print ("After forward")
 #                   print ("start_logits.shape: "+str(model_output.start_logits.shape))
                
                
                if self.version >= '4.0.0':
                    start_scores = model_output.start_logits
                    end_scores = model_output.end_logits
                else:
                    start_scores, end_scores = model_output
                
                # Old code: 
                # output = torch.stack([start_scores, end_scores], axis=-1).squeeze(0).cpu().numpy()

                # Removed the squeeze operation since it may interfere when batch size is 1.
                output = torch.stack([start_scores, end_scores], axis=-1).cpu().numpy()

                # print ("sample_ids: ", len (sample_ids))

                for index, item in enumerate (output): 
                    # print (item.shape)

#                    if (i==0): 
#                        print ("Output shape is: " + str(output.shape))


                    response_array = array.array("B", item.tobytes())
#                    if (i==0):
#                        print ("response_array is: ")
#                        print (len (response_array))
                    bi = response_array.buffer_info()
                    # response = lg.QuerySampleResponse(query_samples[i+index].id, bi[0], bi[1])
                    response = lg.QuerySampleResponse(sample_ids[index], bi[0], bi[1])
                    lg.QuerySamplesComplete([response])


    def flush_queries(self):
        pass

    def __del__(self):
        print("Finished destroying SUT.")

def get_pytorch_sut(args):
    return BERT_PyTorch_SUT(args)
