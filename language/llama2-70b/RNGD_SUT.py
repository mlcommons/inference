import array
import logging
import os
import pickle
import queue
import threading
import time
from pathlib import Path

import mlperf_loadgen as lg
import numpy as np
import torch
from furiosa_llm_models.llama.symbolic.mlperf_submission import \
    LlamaForCausalLM  # isort:skip
from RNGD_generator import MLPerfSubmissionGreedySearch
from SUT import SUT as PyTorchSUT
from SUT import FirstTokenStreamer
from torch.nn.functional import pad
from transformers import AutoTokenizer
from transformers.generation.logits_process import \
    MinNewTokensLengthLogitsProcessor
from transformers.generation.stopping_criteria import MaxLengthCriteria

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Llama-70B-SUT")

gen_kwargs = {
    "early_stopping": True,
    "max_new_tokens": 1024,
    "min_new_tokens": 1,
    "num_beams": 1,
    "do_sample": False,
}

EARLY_STOPPING = True
PAD_TOKEN_ID = EOS_TOKEN_ID = 2
MAX_LENGTH = 2048
MAX_NEW_TOKENS = 1024
MIN_NEW_TOKENS = 1
NUM_BEAMS = 1
DO_SAMPLE = False
RETURN_DICT_IN_GENERATE = False
LOGITS_PROCESSOR = MinNewTokensLengthLogitsProcessor
STOPPING_CRITERIA = MaxLengthCriteria
KV_DTYPE = torch.float32
QUANT_KV_DTYPE = torch.int8
BUCKET_SIZE = 2048

LLAMA_TRANSFORMER_LAYER = "model.layers"


class SUT(PyTorchSUT):
    def __init__(
        self,
        model_path=None,
        dtype="bfloat16",
        device="cpu",
        batch_size=None,
        total_sample_count=24576,
        dataset_path=None,
        use_cached_outputs=False,  # Set this to True *only for test accuracy runs* in case your prior session was killed partway through
        workers=1,
        args=None,
    ):
        self.quantize = args.quantize
        self.torch_numeric_optim = args.torch_numeric_optim
        self.quant_param_path = args.quant_param_path
        self.quant_format_path = args.quant_format_path
        super().__init__(
            model_path,
            dtype,
            device,
            batch_size,
            total_sample_count,
            dataset_path,
            use_cached_outputs,
            workers,
        )

    def process_queries(self):
        """Processor of the queued queries. User may choose to add batching logic"""

        while True:
            qitem = self.query_queue.get()
            if qitem is None:
                break

            query_ids = [q.index for q in qitem]

            fname = "q" + "_".join([str(i) for i in query_ids])
            fname = f"run_outputs/{fname}.pkl"
            _p = Path(fname)
            if self.use_cached_outputs and _p.exists():
                # Read cache
                with _p.open(mode="rb") as f:
                    d = pickle.load(f)
                processed_output = d["outputs"]
                tik1 = None
                tik2 = None
                tik3 = None
                tok = None
            else:
                # Construct / collate batch
                max_seq_len = 1024

                tik1 = time.time()

                input_ids_tensor = []
                input_masks_tensor = []
                input_len = []
                max_input_len = max
                for q in qitem:
                    # max padding
                    input_ids_tensor.append(
                        pad(
                            self.data_object.input_ids[q.index],
                            (
                                max_seq_len - self.data_object.input_lens[q.index],
                                0,
                                0,
                                0,
                            ),
                            value=self.tokenizer.pad_token_id,
                        )
                    )
                    input_masks_tensor.append(
                        pad(
                            self.data_object.attention_masks[q.index],
                            (
                                max_seq_len - self.data_object.input_lens[q.index],
                                0,
                                0,
                                0,
                            ),
                            value=0,
                        )
                    )
                    input_len.append(self.data_object.input_lens[q.index])
                input_ids_tensor = torch.cat(input_ids_tensor)
                input_masks_tensor = torch.cat(input_masks_tensor)

                assert input_ids_tensor.shape == input_masks_tensor.shape
                assert input_ids_tensor.shape[0] <= self.batch_size

                tik2 = time.time()

                logits_processor = LOGITS_PROCESSOR(
                    input_ids_tensor.shape[-1], MIN_NEW_TOKENS, EOS_TOKEN_ID
                )
                stopping_criteria = STOPPING_CRITERIA(
                    MAX_LENGTH,
                    getattr(self.model, "max_position_embeddings", None),
                )

                pred_output_tokens = self.generator.generate(
                    input_ids=input_ids_tensor,
                    attention_mask=input_masks_tensor,
                    logits_processor=logits_processor,
                    stopping_criteria=stopping_criteria,
                    max_length=MAX_LENGTH,
                    pad_token_id=PAD_TOKEN_ID,
                    eos_token_id=EOS_TOKEN_ID,
                    return_dict_in_generate=RETURN_DICT_IN_GENERATE,
                    kv_dtype=self.kv_dtype,
                    bucket_size=BUCKET_SIZE,
                )
                tik3 = time.time()

                processed_output = self.data_object.postProcess(
                    pred_output_tokens,
                    input_seq_lens=input_len,
                    query_id_list=query_ids,
                )

            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                response = [lg.QuerySampleResponse(qitem[i].id, bi[0], bi[1], n_tokens)]
                lg.QuerySamplesComplete(response)

            tok = time.time()

            with self.sample_counter_lock:
                self.sample_counter += len(qitem)
                print(f"Samples run: {self.sample_counter}")
                if tik1:
                    print(f"\tBatchMaker time: {tik2 - tik1}")
                    print(f"\tInference time: {tik3 - tik2}")
                    print(f"\tPostprocess time: {tok - tik3}")
                    print(f"\t==== Total time: {tok - tik1}")
                else:
                    print(f"\tLoaded from cache: {_p}")

    def load_model(self):
        self.model = LlamaForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            low_cpu_mem_usage=True,
            torch_dtype=self.amp_dtype,
        )
        print("Loaded model")

        self.device = torch.device(self.device)
        if self.device == "cpu":
            self.model = self.model.to(
                self.device
            )  # Force CPU if your system has GPU and you specifically want CPU-only run

        self.model.eval()
        self.model = self.model.to(memory_format=torch.channels_last)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            model_max_length=1024,
            padding_side="left",
            use_fast=False,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        device_map = None
        # Needs to place paged attention key value blocks on the same device as the transformer layers
        if hasattr(self.model, "hf_device_map"):
            device_map = {
                k.split(LLAMA_TRANSFORMER_LAYER + ".")[1]: v
                for k, v in self.model.hf_device_map.items()
                if LLAMA_TRANSFORMER_LAYER in k
            }

        if self.quantize:
            from quantization import quantize_model
            from quantization.utils import random_seed, set_optimization

            random_seed()
            set_optimization(self.torch_numeric_optim)

            # if self.device != "cuda:0":
            #     raise ValueError(
            #         "Inference on a device other than GPU is not supported yet."
            #     )
            traced_model = self.model.trace_all()
            model = quantize_model(
                traced_model,
                qparam_path=self.quant_param_path,
                qformat_path=self.quant_format_path,
            )
            self.kv_dtype = QUANT_KV_DTYPE
        else:
            model = self.model.trace_all()
            self.kv_dtype = KV_DTYPE

        self.generator = MLPerfSubmissionGreedySearch(**model, device_map=device_map)
        print("Loaded tokenizer")


# TODO: Implement the server scenario for RNGD
class SUTServer(SUT):
    def __init__(
        self,
        model_path=None,
        dtype="bfloat16",
        device="cpu",
        total_sample_count=24576,
        dataset_path=None,
        batch_size=None,
        workers=1,
        args=None,
    ):
        raise NotImplementedError("Server scenario for RNGD is not implemented yet.")
        super().__init__(
            model_path=model_path,
            dtype=dtype,
            device=device,
            total_sample_count=total_sample_count,
            dataset_path=dataset_path,
            workers=workers,
            args=args,
        )

        self.first_token_queue = queue.Queue()

    def start(self):

        # Create worker threads
        for j in range(self.num_workers):
            worker = threading.Thread(target=self.process_queries)
            worker.start()
            self.worker_threads[j] = worker

        # Create first token response thread
        self.ft_response_thread = threading.Thread(target=self.process_first_tokens)
        self.ft_response_thread.start()

    def process_first_tokens(self):

        while True:
            first_token_item = self.first_token_queue.get()

            if first_token_item is None:
                log.info("Exiting First token response thread")
                break

            first_tokens, response_id = first_token_item

            response_data = array.array(
                "B", np.array(first_tokens, np.float32).tobytes()
            )
            bi = response_data.buffer_info()
            response = [lg.QuerySampleResponse(response_id, bi[0], bi[1])]
            lg.FirstTokenComplete(response)

    def process_queries(self):
        """Processor of the queued queries. User may choose to add batching logic"""
        while True:

            qitem = self.query_queue.get()
            if qitem is None:
                break

            input_ids_tensor = self.data_object.input_ids[qitem.index]
            input_masks_tensor = self.data_object.attention_masks[qitem.index]

            # TODO: This PoC is super slow with significant overhead. Best to create a patch to `generate`
            tokens_cache = []
            tokens_streamer = FirstTokenStreamer(
                self.first_token_queue,
                tokens_cache=tokens_cache,
                is_first_token=True,
                response_ids=[qitem.id],
            )

            _ = self.model.generate(
                input_ids=input_ids_tensor,
                attention_mask=input_masks_tensor,
                pad_token_id=self.tokenizer.pad_token_id,
                streamer=tokens_streamer,
                **gen_kwargs,
            )

            output_tokens = tokens_streamer.get_out_tokens()
            n_tokens = len(output_tokens)
            response_array = array.array(
                "B", np.array(output_tokens, np.int32).tobytes()
            )
            bi = response_array.buffer_info()
            response = [lg.QuerySampleResponse(qitem.id, bi[0], bi[1], n_tokens)]
            lg.QuerySamplesComplete(response)

    def issue_queries(self, query_samples):

        self.query_queue.put(query_samples[0])

    def stop(self):
        for _ in range(self.num_workers):
            self.query_queue.put(None)

        for worker in self.worker_threads:
            worker.join()

        self.first_token_queue.put(None)
        self.ft_response_thread.join()
