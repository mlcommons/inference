import os
import time
import numpy as np
import array
import torch
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import mlperf_loadgen as lg
from dataset import Dataset
from tqdm import tqdm
from accelerate import disk_offload

# For QDL
import threading
import requests
from time import sleep

gen_kwargs = {
    "early_stopping": True,
    "max_new_tokens": 128,
    "min_new_tokens": 30,
    "num_beams": int(os.environ.get("GPTJ_BEAM_SIZE", "4")), # only beam_size 4 is allowed for official submission
}


class SUT_base():
    def __init__(self, model_path, dtype, dataset_path, max_examples, use_gpu=False, network=None):
        # TODO : Pass model file name to init instead of args
        self.dataset_path = dataset_path
        self.network = network
        self.model_name = "EleutherAI/gpt-j-6B"
        self.model_path = model_path
        self.use_gpu = use_gpu
        if not self.network == "lon":
            print("Loading PyTorch model...")
            
            # dtype
            if dtype == 'bfloat16':
                self.amp_enabled = True
                self.amp_dtype = torch.bfloat16
                print("BF16 autocast")
            elif dtype == 'float16':
                self.amp_enabled = True
                self.amp_dtype = torch.float16
            else:
                self.amp_enabled = False
                self.amp_dtype = torch.float32
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map="auto" if not self.use_gpu else None,
                    low_cpu_mem_usage=True if not self.use_gpu else False,
                    torch_dtype=self.amp_dtype,
                    offload_folder="offload" if not self.use_gpu else None,    # specify offload folder when using devices with less RAM
                    offload_state_dict = True if not self.use_gpu else False   # to have some shards of the model to be on the disk
                )
            except ValueError as e:
                if "disk_offload" in str(e):
                    print("Offloading the whole model to disk...")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        low_cpu_mem_usage=True if not self.use_gpu else False,
                        torch_dtype=self.amp_dtype,
                        offload_state_dict = True if not self.use_gpu else False   # to have some shards of the model to be on the disk
                    ).cpu()
                    disk_offload(model=self.model, offload_dir="offload")

            # Cast the model to GPU if the flag is set.
            if self.use_gpu:
                print(f"Casting models to GPU...")
                assert torch.cuda.is_available(), "torch gpu is not available, exiting..."
                self.device = torch.device("cuda:0")
                self.model.to(self.device)

            self.model.eval()
            try: # for systems with low ram, the below command gives error as some part is offloaded to disk
                self.model = self.model.to(memory_format=torch.channels_last)
            except:
                pass

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                model_max_length=1919,
                padding_side="left",
                use_fast=False,)
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.data_object = Dataset(
                self.dataset_path, total_count_override=max_examples)
        self.qsl = lg.ConstructQSL(self.data_object.count, self.data_object.perf_count,
                                   self.data_object.LoadSamplesToRam, self.data_object.UnloadSamplesFromRam)

        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)

    def issue_queries(self, query_samples):
        print("Number of Samples in query_samples : ", len(query_samples))

        total_samples_done = 0
        list_prompts_tokens = []
        list_prompts_attn_masks = []

        for i in tqdm(range(len(query_samples))):
            index = query_samples[i].index
            input_ids_tensor = self.data_object.source_encoded_input_ids[index]
            input_masks_tensor = self.data_object.source_encoded_attn_masks[index]

            # Cast to GPU
            if self.use_gpu:
                input_ids_tensor = input_ids_tensor.to(self.device)
                input_masks_tensor = input_masks_tensor.to(self.device)

            self.inference_call(input_ids_tensor, input_masks_tensor, query_samples[i].id)

    def inference_call(self, input_ids_tensor, input_masks_tensor, query_id=None):
        ''' Common for all scenarios '''
        torch_device_type = 'cuda' if self.use_gpu else 'cpu'
        if isinstance(input_ids_tensor, list):
            input_ids_tensor = torch.tensor(input_ids_tensor, dtype=torch.long)
            input_masks_tensor = torch.tensor(input_masks_tensor, dtype=torch.long)
            
        with torch.inference_mode(), torch.autocast(device_type=torch_device_type, enabled=self.amp_enabled, dtype=self.amp_dtype if self.amp_enabled else None):
            input_batch = dict()
            input_batch['input_ids'] = input_ids_tensor
            input_batch['attention_mask'] = input_masks_tensor

            output_batch = self.model.generate(# takes Long and Int tensor datatype only
                **input_batch, **gen_kwargs, pad_token_id=self.tokenizer.eos_token_id)

            input_batch_lengths = [x.shape[0]
                                   for x in input_batch["input_ids"]]

            output_batch_lengths = [x.shape[0] for x in output_batch]

            output_batch_truncated = []
            for data, source_len in zip(output_batch, input_batch_lengths):
                output_batch_truncated.append(data[source_len:])

            output_batch_truncated = torch.stack(output_batch_truncated)

            pred_output_batch = output_batch_truncated.cpu().numpy()

            if self.network == "sut":
                return output_batch_truncated.tolist()

            response_array = array.array("B", pred_output_batch[0].tobytes())
            bi = response_array.buffer_info()
            response = lg.QuerySampleResponse(query_id, bi[0], bi[1])
            lg.QuerySamplesComplete([response])

    def flush_queries(self):
        pass

    def __del__(self):
        print("Finished destroying SUT.")


class SUT_Offline(SUT_base):
    def __init__(self, model_path, dtype, dataset_path, max_examples, use_gpu, network):
        SUT_base.__init__(self, model_path, dtype, dataset_path, max_examples, use_gpu, network)
    '''IssueQuery and inference methods implemented in Base class'''


class SUT_Server(SUT_base):
    def __init__(self, model_path, dtype, dataset_path, max_examples, use_gpu, network):

        SUT_base.__init__(self, model_path, dtype, dataset_path, max_examples, use_gpu, network)
        self.total_samples_done = 0
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        print("SUT Server")

    def issue_queries(self, query_samples):

        index = query_samples[0].index
        input_ids_tensor = self.data_object.source_encoded_input_ids[index]
        input_masks_tensor = self.data_object.source_encoded_attn_masks[index]

        if self.use_gpu:
            input_ids_tensor = input_ids_tensor.to(self.device)
            input_masks_tensor = input_masks_tensor.to(self.device)

        pred_output_batch = self.inference_call(
            input_ids_tensor, input_masks_tensor).cpu().numpy()

        response_array = array.array("B", pred_output_batch.tobytes())
        bi = response_array.buffer_info()
        responses = [lg.QuerySampleResponse(query_samples[0].id, bi[0], bi[1])]
        lg.QuerySamplesComplete(responses)
        self.total_samples_done += 1
        if self.total_samples_done % 5 == 0:
            print("Completed : ", self.total_samples_done)


class SUT_SingleStream(SUT_base):
    def __init__(self, model_path, dtype, dataset_path, max_examples, use_gpu, network):
        SUT_base.__init__(self, model_path, dtype, dataset_path, max_examples, use_gpu, network)
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        self.total_samples_done = 0

    def issue_queries(self, query_samples):

        index = query_samples[0].index
        input_ids_tensor = self.data_object.source_encoded_input_ids[index]
        input_masks_tensor = self.data_object.source_encoded_attn_masks[index]

        if self.use_gpu:
            input_ids_tensor = input_ids_tensor.to(self.device)
            input_masks_tensor = input_masks_tensor.to(self.device)

        pred_output_batch = self.inference_call(
            input_ids_tensor, input_masks_tensor).cpu().numpy()

        response_array = array.array("B", pred_output_batch.tobytes())
        bi = response_array.buffer_info()
        responses = [lg.QuerySampleResponse(query_samples[0].id, bi[0], bi[1])]
        lg.QuerySamplesComplete(responses)
        self.total_samples_done += 1
        if self.total_samples_done % 5 == 0:
            print("Completed : ", self.total_samples_done)

class GPTJ_QDL:
    """QDL acting as a proxy to the SUT.
    This QDL communicates with the SUT via HTTP.
    It uses two endpoints to communicate with the SUT:
    - /predict/ : Send a query to the SUT and get a response.
    - /getname/ : Get the name of the SUT. Send a getname to the SUT and get a response.
    """
    def __init__(self, sut, sut_server_addr: list):
        self.qsl = sut.qsl
        self.sut = sut
        # Construct QDL from the python binding
        self.qdl = lg.ConstructQDL(
            self.issue_query, self.flush_queries, self.client_get_name)
        self.sut_server_addr = sut_server_addr
        self.num_nodes = len(sut_server_addr)

        # For round robin between the SUTs:
        self.next_sut_id = 0
        self.lock = threading.Lock()

    def issue_query(self, query_samples):
        """Process the query to send to the SUT"""
        threading.Thread(target=self.process_query_async,
                         args=[query_samples]).start()

    def flush_queries(self):
        """Flush the queries. Dummy implementation."""
        pass

    def process_query_async(self, query_samples):
        """
        This function is called by the Loadgen in a separate thread.
        It is responsible for
            1. Creating a query for the SUT, by reading the features from the QSL.
            2. Sending the query to the SUT.
            3. Waiting for the response from the SUT.
            4. Deserializing the response.
            5. Calling mlperf_loadgen.QuerySamplesComplete(query_samples, response)
        Args:
            query_samples: A list of QuerySample objects.
        """

        max_num_threads = int(os.environ.get('CM_MAX_NUM_THREADS', os.cpu_count()))

        for i in range(len(query_samples)):
            index = query_samples[i].index
            input_ids_tensor = self.sut.data_object.source_encoded_input_ids[index]
            input_masks_tensor = self.sut.data_object.source_encoded_attn_masks[index]

            # for serialising the tensor to json, it should be converted to list
            input_ids_list = input_ids_tensor.cpu().numpy().tolist()
            input_masks_list = input_masks_tensor.cpu().numpy().tolist()

            # # Cast to GPU
            # if self.use_gpu:
            #     input_ids_tensor = input_ids_tensor.to(self.device)
            #     input_masks_tensor = input_masks_tensor.to(self.device)
            encoded_eval_features = {
                    "input_ids_tensor": input_ids_list,
                    "input_masks_tensor": input_masks_list
                    }
            n = threading.active_count()
            while n >= max_num_threads:
                sleep(0.0001)
                n = threading.active_count()
            threading.Thread(target=self.client_predict_worker,
                         args=[encoded_eval_features, query_samples[i].id]).start()
    
    def get_sut_id_round_robin(self):
        """Get the SUT id in round robin."""
        with self.lock:
            res = self.next_sut_id
            self.next_sut_id = (self.next_sut_id + 1) % self.num_nodes
        return res
    
    def client_predict_worker(self, query, query_id):
        """Serialize the query, send it to the SUT in round robin, and return the deserialized response."""
        url = '{}/predict/'.format(self.sut_server_addr[self.get_sut_id_round_robin()])
        responses = []
        print(f"The url:{url} and query:{query}")
        response = requests.post(url, json={'query': query})
        output = response.json()['result']
        output = np.array(output).astype(np.float32)
        response_array = array.array("B", output.tobytes())
        bi = response_array.buffer_info()

        responses.append(lg.QuerySampleResponse(query_id, bi[0], bi[1]))
        lg.QuerySamplesComplete(responses)
    
    def client_get_name(self):
        """Get the name of the SUT from ALL the SUTS."""
        if len(self.sut_server_addr) == 1:
            return requests.post(f'{self.sut_server_addr[0]}/getname/').json()['name']
    
        sut_names = [requests.post(f'{addr}/getname/').json()['name'] for addr in self.sut_server_addr]
        return "Multi-node SUT: " + ', '.join(sut_names)

    def __del__(self):
        lg.DestroyQDL(self.qdl)   




def get_SUT(model_path, scenario, dtype, dataset_path, max_examples, use_gpu=False, network=None):
    if scenario == "Offline":
        return SUT_Offline(model_path, dtype, dataset_path, max_examples, use_gpu, network)
    elif scenario == "Server":
        return SUT_Server(model_path, dtype, dataset_path, max_examples, use_gpu, network)
    elif scenario == "SingleStream":
        return SUT_SingleStream(model_path, dtype, dataset_path, max_examples, use_gpu, network)
