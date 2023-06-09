import array
import torch
import os
import sys
sys.path.append(os.environ["MEGATRON_PATH"])
from megatron import print_rank_0

from megatron.initialize import initialize_megatron, set_jit_fusion_options
from megatron.training import setup_model_and_optimizer
from megatron.model import GPTModel, ModelType
import mlperf_loadgen as lg
from dataset import Dataset

from megatron.model.text_generation import generate_tokens_probs_and_return_on_first_stage




class SUT_base():
    def __init__(self, model_path, dtype, dataset_path, max_examples, args, use_gpu=False, gen_kwargs = {}):
        # TODO : Pass model file name to init instead of args
        print("Loading PyTorch model...")
        self.model_name = "Megatron-LM"
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.gen_kwargs = gen_kwargs
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

        initialize_megatron(args_defaults = args)
        set_jit_fusion_options()
        self.data_object = Dataset(
            self.dataset_path, total_count_override=max_examples, args = args)
        
        def model_provider(pre_process=True, post_process=True):
            """Build the model."""
            print_rank_0('building GPT model ...')
            model = GPTModel(
                num_tokentypes=0,
                parallel_output=True,
                pre_process=pre_process,
                post_process=post_process
            )
            return model
        
        self.model, opt_param_scheduler = setup_model_and_optimizer(model_provider, ModelType.encoder_or_decoder)


        self.qsl = lg.ConstructQSL(self.data_object.count, self.data_object.perf_count,
                                   self.data_object.LoadSamplesToRam, self.data_object.UnloadSamplesFromRam)

        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)

    def issue_queries(self, query_samples):
        print("Number of Samples in query_samples : ", len(query_samples))

        for i in range(len(query_samples)):
            index = query_samples[i].index
            input_ids_tensor = self.data_object.source_encoded_input_ids[index]
            input_masks_tensor = self.data_object.source_encoded_attn_masks[index]
            input_length_tensor = self.data_object.source_encoded_input_id_leghts[index]

            # Cast to GPU
            if self.use_gpu:
                input_ids_tensor = input_ids_tensor.to(self.device)
                input_masks_tensor = input_masks_tensor.to(self.device)

            pred_output_batch = self.inference_call(
                input_ids_tensor, input_masks_tensor, input_length_tensor).cpu().numpy()

            response_array = array.array("B", pred_output_batch[0].tobytes())
            bi = response_array.buffer_info()
            response = [lg.QuerySampleResponse(
                query_samples[i].id, bi[0], bi[1])]
            lg.QuerySamplesComplete(response)
            if i % 5 == 0:
                print("Completed : ", i)

    def inference_call(self, input_ids_tensor, input_masks_tensor, input_length_tensor):
        ''' Common for all scenarios '''
        torch_device_type = 'cuda' if self.use_gpu else 'cpu'

        with torch.inference_mode(), torch.autocast(device_type=torch_device_type, enabled=self.amp_enabled, dtype=self.amp_dtype if self.amp_enabled else None):
            input_batch = dict()
            input_batch['input_ids'] = input_ids_tensor
            input_batch['attention_mask'] = input_masks_tensor

            output_tokens, _, _ = generate_tokens_probs_and_return_on_first_stage(self.model, input_ids_tensor, input_length_tensor, top_k = self.gen_kwargs.get())

            input_batch_lengths = [x.shape[0]
                                   for x in input_batch["input_ids"]]

            output_batch_lengths = [x.shape[0] for x in output_tokens]

            output_batch_truncated = []
            for data, source_len in zip(output_tokens, input_batch_lengths):
                output_batch_truncated.append(data[source_len:])

            output_batch_truncated = torch.stack(output_batch_truncated)

        return output_batch_truncated

    def flush_queries(self):
        pass

    def __del__(self):
        print("Finished destroying SUT.")


class SUT_Offline(SUT_base):
    def __init__(self, model_path, dtype, dataset_path, max_examples,args, use_gpu, gen_kwargs):
        SUT_base.__init__(self, model_path, dtype, dataset_path, max_examples,args, use_gpu, gen_kwargs)
    '''IssueQuery and inference methods implemented in Base class'''


class SUT_Server(SUT_base):
    def __init__(self, model_path, dtype, dataset_path, max_examples, args, use_gpu, gen_kwargs):

        SUT_base.__init__(self, model_path, dtype, dataset_path, max_examples,args, use_gpu, gen_kwargs)
        self.total_samples_done = 0
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        print("SUT Server")

    def issue_queries(self, query_samples):

        index = query_samples[0].index
        input_ids_tensor = self.data_object.source_encoded_input_ids[index]
        input_masks_tensor = self.data_object.source_encoded_attn_masks[index]
        input_length_tensor = self.data_object.source_encoded_input_id_leghts[index]

        if self.use_gpu:
            input_ids_tensor = input_ids_tensor.to(self.device)
            input_masks_tensor = input_masks_tensor.to(self.device)

        pred_output_batch = self.inference_call(
            input_ids_tensor, input_masks_tensor, input_length_tensor).cpu().numpy()

        response_array = array.array("B", pred_output_batch.tobytes())
        bi = response_array.buffer_info()
        responses = [lg.QuerySampleResponse(query_samples[0].id, bi[0], bi[1])]
        lg.QuerySamplesComplete(responses)
        self.total_samples_done += 1
        if self.total_samples_done % 5 == 0:
            print("Completed : ", self.total_samples_done)


class SUT_SingleStream(SUT_base):
    def __init__(self, model_path, dtype, dataset_path, max_examples,args, use_gpu, gen_kwargs):
        SUT_base.__init__(self, model_path, dtype, dataset_path, max_examples,args, use_gpu, gen_kwargs)
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        self.total_samples_done = 0

    def issue_queries(self, query_samples):

        index = query_samples[0].index
        input_ids_tensor = self.data_object.source_encoded_input_ids[index]
        input_masks_tensor = self.data_object.source_encoded_attn_masks[index]
        input_length_tensor = self.data_object.source_encoded_input_id_leghts[index]

        if self.use_gpu:
            input_ids_tensor = input_ids_tensor.to(self.device)
            input_masks_tensor = input_masks_tensor.to(self.device)

        pred_output_batch = self.inference_call(
            input_ids_tensor, input_masks_tensor, input_length_tensor).cpu().numpy()

        response_array = array.array("B", pred_output_batch.tobytes())
        bi = response_array.buffer_info()
        responses = [lg.QuerySampleResponse(query_samples[0].id, bi[0], bi[1])]
        lg.QuerySamplesComplete(responses)
        self.total_samples_done += 1
        if self.total_samples_done % 5 == 0:
            print("Completed : ", self.total_samples_done)


def get_SUT(model_path, scenario, dtype, dataset_path, max_examples, args, use_gpu=False, gen_kwargs = {}):
    if scenario == "Offline":
        return SUT_Offline(model_path, dtype, dataset_path, max_examples,args, use_gpu, gen_kwargs = gen_kwargs)
    elif scenario == "Server":
        return SUT_Server(model_path, dtype, dataset_path, max_examples,args, use_gpu, gen_kwargs = gen_kwargs)
    elif scenario == "SingleStream":
        return SUT_SingleStream(model_path, dtype, dataset_path, max_examples,args, use_gpu, gen_kwargs = gen_kwargs)
