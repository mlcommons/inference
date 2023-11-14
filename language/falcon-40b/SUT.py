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


gen_kwargs = {
    "early_stopping": True,
    "max_new_tokens": 1024,
    "min_new_tokens": 1,
    "num_beams": 4
}
#int(os.environ.get("GPTJ_BEAM_SIZE", "4")), # only beam_size 4 is allowed for official submission
#}


class SUT():
    def __init__(self, model_path=None, dtype="bfloat16", device="cpu", total_sample_count=24576, dataset_path=None):
        # TODO : dataset_path should be used when dataset is already available on disk

        print("Loading PyTorch model...")
        self.model_path = model_path or "tiiuae/falcon-40b-instruct"
        self.device = device

        # dtype
        if dtype == 'bfloat16':
            self.amp_enabled = True
            self.amp_dtype = torch.bfloat16
        elif dtype == 'float16':
            self.amp_enabled = True
            self.amp_dtype = torch.float16
        else:
            self.amp_enabled = False
            self.amp_dtype = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map= "auto" if self.device=="cpu" else None,
            low_cpu_mem_usage=True if self.device=="cpu" else False,
            torch_dtype=self.amp_dtype
        )

        # Cast the model to GPU if the flag is set.
        if 'cuda' in self.device:
            print(f"Casting models to GPU...")
            assert torch.cuda.is_available(), "torch gpu is not available, exiting..."
            self.device = torch.device(self.device)
            self.model.to(self.device)

        self.model.eval()
        self.model = self.model.to(memory_format=torch.channels_last)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            model_max_length=1024,
            padding_side="left",
            use_fast=False,)

        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.dataset_path = dataset_path
        self.data_object = Dataset(dataset_path=self.dataset_path, total_sample_count=total_sample_count)
        self.qsl = lg.ConstructQSL(self.data_object.total_sample_count, self.data_object.perf_count,
                                   self.data_object.LoadSamplesToRam, self.data_object.UnloadSamplesFromRam)

        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)

    def predict(self,**kwargs):
        raise NotImplementedError

    def issue_queries(self, query_samples):

        total_samples_done = 0
        list_prompts_tokens = []
        list_prompts_attn_masks = []

        for i in range(len(query_samples)):
            index = query_samples[i].index
            input_ids_tensor = self.data_object.input_ids[index]
            input_masks_tensor = self.data_object.attention_masks[index]

            pred_output_tokens = self.model.generate(
                                                input_ids=input_ids_tensor,
                                                attention_mask=input_masks_tensor,
                                                pad_token_id=self.tokenizer.pad_token_id,
                                                **gen_kwargs
                                                )

            processed_output = self.data_object.postProcess(pred_output_tokens)

            response_array = array.array("B", processed_output[0].tobytes())
            bi = response_array.buffer_info()
            response = [lg.QuerySampleResponse(
                query_samples[i].id, bi[0], bi[1])]
            lg.QuerySamplesComplete(response)


    def flush_queries(self):
        pass

    def __del__(self):
        print("Finished destroying SUT.")


class SUT_Server(SUT):
    def __init__(self, model_path, dtype, dataset_path, max_examples, use_gpu):

        SUT_base.__init__(self, model_path, dtype, dataset_path, max_examples, use_gpu)
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



