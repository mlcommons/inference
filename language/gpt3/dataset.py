import os
import sys

sys.path.append(os.environ["MEGATRON_PATH"])
from megatron.tokenizer import build_tokenizer
from megatron.utils import get_ltor_masks_and_position_ids
import torch
import argparse
import utils


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


class Dataset:
    def __init__(
        self,
        dataset_path,
        batch_size=1,
        args=None,
        gen_kwards={},
        total_count_override=None,
        perf_count_override=None,
        debug=False,
    ):
        print("Constructing QSL")

        self.dataset = "cnn_dailymail"
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.debug = debug
        self.gen_kwards = gen_kwards

        args.rank = 0
        args.tokenizer_type = "GPT2BPETokenizer"
        args.vocab_file = "data/vocab.json"
        args.merge_file = "data/merges.txt"
        args.make_vocab_size_divisible_by = 128
        args.tensor_model_parallel_size = 1
        self.tokenizer = build_tokenizer(args)

        self.list_data_dict = utils.jload(self.dataset_path)

        prompt_input, prompt_no_input = (
            PROMPT_DICT["prompt_input"],
            PROMPT_DICT["prompt_no_input"],
        )
        self.sources = [
            prompt_input.format_map(example) for example in self.list_data_dict
        ]
        self.targets = [f"{example['output']}" for example in self.list_data_dict]

        (
            self.source_encoded_input_ids,
            # self.source_encoded_attn_masks,
            self.source_encoded_input_id_lengths,
        ) = self.encode_samples()
        self.count = total_count_override or len(self.sources)
        self.perf_count = perf_count_override or self.count

    def _build_attention_mask(self, tokens):
        """Build the attention mask and postition ids for the input tokens."""

        # Since we are not interested in loss-mask and reset attention/position
        # is also False, eod_token is not used so it is safe to set it to None.
        attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
            data=tokens,
            eod_token=None,
            reset_position_ids=False,
            reset_attention_mask=False,
            eod_mask_loss=False,
        )
        return attention_mask

    def tokenize_prompts(self, prompts, tokens_to_generate, add_BOS):
        """Given a set of prompts and number of tokens to generate:
        - tokenize prompts
        - set the sequence length to be the max of length of prompts
        plus the number of tokens we would like to generate
        - pad all the sequences to this length so we can convert them
        into a 2D tensor.
        """

        # Tokenize all the prompts.
        if add_BOS:
            prompts_tokens = [
                [self.tokenizer.eod] + self.tokenizer.tokenize(prompt)
                for prompt in prompts
            ]
        else:
            prompts_tokens = [self.tokenizer.tokenize(prompt) for prompt in prompts]

        # Now we have a list of list of tokens which each list has a different
        # size. We want to extend this list to:
        #   - incorporate the tokens that need to be generated
        #   - make all the sequences equal length.
        # Get the prompts length.
        prompts_length = [len(prompt_tokens) for prompt_tokens in prompts_tokens]
        # Get the max prompts length.
        max_prompt_len = max(prompts_length)
        # Number of tokens in the each sample of the batch.
        samples_length = max_prompt_len + tokens_to_generate
        # Now update the list of list to be of the same size: samples_length.
        for prompt_tokens, prompt_length in zip(prompts_tokens, prompts_length):
            padding_size = samples_length - prompt_length
            prompt_tokens.extend([self.tokenizer.eod] * padding_size)

        # Now we are in a structured format, we can convert to tensors.
        # prompts_tokens_tensor = torch.LongTensor(prompts_tokens)
        # prompts_length_tensor = torch.LongTensor(prompts_length)

        return prompts_tokens, prompts_length

    def encode_samples(self):
        print("Encoding Samples")

        total_samples = len(self.sources)

        source_encoded_input_ids = []
        source_encoded_input_id_lengths = []
        # source_encoded_attn_masks = []

        for i in range(total_samples):
            if i % 100 == 0 and self.debug:
                print("Sentence:")
                print(self.sources[i])
                print(
                    "--------------------------------------------------------------------------------"
                )
            tokens, length = self.tokenize_prompts(
                [self.sources[i]], self.gen_kwards.get("max_new_tokens", 128), None
            )
            # attn_mask = self._build_attention_mask(tokens)
            source_encoded_input_ids.append(tokens)
            # source_encoded_attn_masks.append(attn_mask)
            source_encoded_input_id_lengths.append(length)
            if i % 100 == 0 and self.debug:
                print(f"Tokens: {tokens}")
                print(f"Original length: {length}")
                # input("...")

        return (
            source_encoded_input_ids,
            # source_encoded_attn_masks,
            source_encoded_input_id_lengths,
        )

    def LoadSamplesToRam(self, sample_list):
        pass

    def UnloadSamplesFromRam(self, sample_list):
        pass

    def __del__(self):
        print("Finished destroying QSL.")

