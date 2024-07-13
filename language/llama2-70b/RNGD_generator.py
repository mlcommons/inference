import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch._dynamo.eval_frame import OptimizedModule
from torch.fx import GraphModule
from transformers import (GPTJConfig, LlamaConfig, PretrainedConfig,
                          PreTrainedModel)
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.utils import GreedySearchDecoderOnlyOutput
from transformers.modeling_outputs import (CausalLMOutputWithCrossAttentions,
                                           CausalLMOutputWithPast)

SUPPORTED_GENERATION_RETURN_DICT_TYPES = (
    CausalLMOutputWithPast,
    CausalLMOutputWithCrossAttentions,
)
MAX_PACKING_PER_ROW: int = 254
MAX_BATCH_SIZE: int = 4
BLOCK_SIZE: int = 1
logger = logging.getLogger(__name__)


class MLPerfSubmissionGreedySearch:
    def __init__(
        self,
        model: Optional[PreTrainedModel] = None,
        prefill: Optional[GraphModule] = None,
        decode: Optional[GraphModule] = None,
        device_map: Optional[Dict[str, int]] = None,
    ) -> None:
        if isinstance(model, PreTrainedModel):
            self.model = model
        # OptimizedModule cannot be checked by isinstance
        # TypeError: Subscripted generics cannot be used with class and instance checks
        if type(model) == OptimizedModule:
            self.model = model
        if (
            isinstance(model, Dict)
            and isinstance(model["prefill"], GraphModule)
            and isinstance(model["decode"], GraphModule)
        ):
            self.model = self.prefill = model["prefill"]
            self.decode = model["decode"]
        if prefill is not None and decode is not None:
            self.model = self.prefill = prefill
            self.decode = decode

        if self.model is None:
            raise ValueError("model is not provided or valid.")
        self.model_config = self.model.config
        self.device_map = device_map

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        return_dict_in_generate: Optional[bool] = None,
        **model_kwargs,
    ):
        kv_dtype = model_kwargs.get("kv_dtype")
        if kv_dtype is None:
            raise ValueError("`kv_dtype` is required for Paged Attention.")

        device = input_ids.device
        batch_size = input_ids.shape[0]
        bucket_size = model_kwargs.get("bucket_size") or max_length

        if bucket_size is None:
            raise ValueError("`bucket_size` is required for Paged Attention.")

        key_value_blocks = model_kwargs.get(
            "key_value_blocks"
        ) or self.create_key_value_blocks(
            batch_size, bucket_size, kv_dtype, device, self.device_map
        )
        self.initialize_key_value_block_indices(key_value_blocks)
        # ----------- initial_settings -----------------
        starting_input_ids = input_ids
        starting_attention_mask = attention_mask
        batch_size, prompt_len = starting_input_ids.shape

        # TODO(MAX BATCH CHECK ONLY EXISTS FOR THIS PYTHON GENERATOR)
        # In vllm, generate is async and inner scheduler decides which batch to use based on
        # memory allocation
        assert batch_size <= MAX_BATCH_SIZE

        starting_position_ids = starting_attention_mask.long().cumsum(-1) - 1
        starting_position_ids.masked_fill_(starting_attention_mask == 0, 1)

        # ----------- adjust to bucket settings --------
        device = input_ids.device
        attention_mask = torch.zeros((batch_size, bucket_size), dtype=torch.int).to(
            device
        )
        attention_mask[:, :prompt_len] = starting_attention_mask

        input_ids = torch.full(
            (batch_size, bucket_size), fill_value=pad_token_id, dtype=torch.int
        ).to(device)
        input_ids[:, :prompt_len] = starting_input_ids

        position_ids = torch.zeros((batch_size, bucket_size), dtype=torch.long).to(
            device
        )
        position_ids[:, :prompt_len] = starting_position_ids

        sequence_idx = prompt_len - 1
        is_prefill = True

        scores = None

        unfinished_sequences = torch.ones(
            batch_size, dtype=torch.long, device=input_ids.device
        ).to(device)
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device)
        next_tokens = None

        # start generating new tokens
        for i in range(max_length - prompt_len):
            if is_prefill:
                (new_key_location, new_value_location) = (
                    self.prepare_prefill_input_metadata(attention_mask)
                )

                (
                    packed_input_ids,
                    _,
                    causal_mask,
                    packed_position_ids,
                    logit_target_locations,
                    new_key_location,
                    new_value_location,
                ) = self.prepare_prefill_inputs(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    new_key_location=new_key_location,
                    new_value_location=new_value_location,
                    pad_token_id=pad_token_id,
                )  # original attention mask, original position ids

                forward_kwargs = {
                    "input_ids": packed_input_ids.to(device),
                    "attention_mask": None,
                    "causal_mask": causal_mask.to(device),
                    "position_ids": packed_position_ids.to(device),
                    "past_key_values": key_value_blocks,
                    "new_key_location": new_key_location.to(device),
                    "new_value_location": new_value_location.to(device),
                    "past_valid_key_indices": None,
                    "past_valid_value_indices": None,
                    "is_prefill": is_prefill,
                    "bucket_size": bucket_size,
                    "use_cache": False,
                }

                # If the model is a GraphModule, we need to switch the model to prefill.
                if isinstance(self.model, GraphModule) and self.model != self.prefill:
                    self.model = self.prefill
            else:
                (input_ids, attention_mask, position_ids) = self.prepare_decode_inputs(
                    next_input_ids=next_tokens,
                    prev_attention_mask=attention_mask,
                    is_first_decode=(True if i == 1 else False),  # FIXME: hacky
                    seq_idx=sequence_idx,
                )

                logit_target_locations = None  # for decode, not needed

                (
                    new_key_location,
                    new_value_location,
                    valid_key_indices,
                    valid_value_indices,
                ) = self.prepare_decode_input_metadata()
                forward_kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "causal_mask": None,
                    "position_ids": position_ids,
                    "past_key_values": key_value_blocks,
                    "new_key_location": new_key_location,
                    "new_value_location": new_value_location,
                    "past_valid_key_indices": valid_key_indices,
                    "past_valid_value_indices": valid_value_indices,
                    "is_prefill": is_prefill,
                    "bucket_size": bucket_size,
                    "use_cache": False,
                }

                # If the model is a GraphModule, we need to switch the model to decode.
                if isinstance(self.model, GraphModule) and self.model != self.decode:
                    self.model = self.decode

            # remove all concrete args from forward_kwargs for they will not be used in the
            # forward pass.
            if isinstance(self.model, GraphModule):
                for arg in self.model.concrete_args:
                    if arg in forward_kwargs:
                        del forward_kwargs[arg]

            outputs = self.model(**forward_kwargs)

            if not is_prefill:
                # now copy the new_key location back to original place for decode_phase
                self.move_kv_cache_block_in_place(
                    seq_idx=sequence_idx,
                    new_location=new_key_location,
                    existing_block_indices=self.active_key_block_indices,
                )
                self.move_kv_cache_block_in_place(
                    seq_idx=sequence_idx,
                    new_location=new_value_location,
                    existing_block_indices=self.active_value_block_indices,
                )

            # done
            outputs = handle_outputs(outputs).to(input_ids.device)

            # done
            next_tokens = self.find_next_tokens(
                outputs,
                logit_target_locations,
                starting_input_ids,
                logits_processor,
                unfinished_sequences,
                pad_token_id,
                is_prefill,
            )

            starting_input_ids = torch.cat(
                [starting_input_ids, next_tokens[:, None]], dim=-1
            )

            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1)
                    .ne(eos_token_id_tensor.unsqueeze(1))
                    .prod(dim=0)
                )
                if unfinished_sequences.max() == 0:
                    break
            if sequence_idx == 10:
                break
            if stopping_criteria(starting_input_ids, scores):
                break

            sequence_idx += 1

            # prepare for next phase
            is_prefill = False

        # reset must be called
        self.reset()
        if return_dict_in_generate:
            return GreedySearchDecoderOnlyOutput(
                sequences=starting_input_ids, scores=scores
            )
        return starting_input_ids

    def prepare_prefill_input_metadata(
        self, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        for prefill, valid_key_indices and valid_value_indices are none
        return (new_key_location, new_value_location)
        """
        new_key_location = []  # shape = (batch, bucket_size)
        new_value_location = []  # shape = (batch, bucket_size)
        for single_attention_mask in attention_mask:
            # for each attention_mask add zero block for padding
            block_indices = []
            for val in single_attention_mask:
                if val == 0:
                    # padding
                    block_indices.append(self.zero_block_index)
                else:
                    block_indices.append(self.available_block_indices.pop())

            self.active_key_block_indices.append(block_indices[:])
            self.active_value_block_indices.append(block_indices)

        new_key_location = torch.IntTensor(self.active_key_block_indices)
        new_value_location = torch.IntTensor(self.active_value_block_indices)

        return (
            new_key_location,
            new_value_location,
        )

    def prepare_decode_input_metadata(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        return (new_key_location, new_value_location, valid_key_indices, valid_valie_indices)
        """
        new_key_location = []  # shape = (batch, 1)
        new_value_location = []  # shape = (batch, 1)
        valid_key_indices = []  # shape = (batch*(bucket_size -1))
        valid_value_indices = []  #
        for key_batch, value_batch in zip(
            self.active_key_block_indices, self.active_value_block_indices
        ):
            valid_key_indices.extend(key_batch[:-1])
            valid_value_indices.extend(value_batch[:-1])

            # we use same block idx for key and value here
            new_block_idx = self.available_block_indices.pop()

            key_batch[-1] = new_block_idx
            value_batch[-1] = new_block_idx

            new_key_location.append([new_block_idx])
            new_value_location.append([new_block_idx])

        new_key_location = torch.IntTensor(new_key_location)
        new_value_location = torch.IntTensor(new_value_location)
        valid_key_indices = torch.IntTensor(valid_key_indices)
        valid_value_indices = torch.IntTensor(valid_value_indices)

        return (
            new_key_location,
            new_value_location,
            valid_key_indices,
            valid_value_indices,
        )

    def prepare_prefill_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        new_key_location: torch.Tensor,
        new_value_location: torch.Tensor,
        pad_token_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[int]]]:
        """
        return (packed_input_ids, causal_mask, packed_position_ids, logit_target_locations, packed_new_key_locatoin, packed_new_value_location)
        """  # noqa: E501
        (
            packed_attention_mask,
            packed_input_ids,
            causal_mask,
            logit_target_locations,
            packed_position_ids,
            packed_new_key_location,
            packed_new_value_location,
        ) = greedy_attention_packing(
            input_ids,
            attention_mask,
            new_key_location,
            new_value_location,
            pad_token_id=pad_token_id,
        )
        return (
            packed_input_ids,
            packed_attention_mask,
            causal_mask,
            packed_position_ids,
            logit_target_locations,
            packed_new_key_location,
            packed_new_value_location,
        )

    def prepare_decode_inputs(
        self,
        next_input_ids: torch.Tensor,
        prev_attention_mask: torch.Tensor,
        is_first_decode: bool,
        seq_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        return (input_ids, attention_mask, position_ids)
        """
        next_attention_mask = prev_attention_mask.clone()

        if is_first_decode:
            # Before: [[1, 1, 1, 0, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 0, 0]]
            # After : [[1, 1, 1, 0, 0, 0, 0, 1], [0, 1, 1, 0, 0, 0, 0, 1]]
            next_attention_mask[:, -1] = 1
        else:
            # Before: [[1, 1, 1, 0, 0, 0, 0, 1], [0, 1, 1, 0, 0, 0, 0, 1]]
            # After : [[1, 1, 1, 1, 0, 0, 0, 1], [0, 1, 1, 1, 0, 0, 0, 1]]
            next_attention_mask[:, seq_idx - 1] = 1

        next_position_ids = next_attention_mask.long().cumsum(-1) - 1
        next_position_ids = next_position_ids[:, -1:]

        return (next_input_ids[:, None], next_attention_mask, next_position_ids)

    def find_next_tokens(
        self,
        logits: torch.Tensor,
        logit_target_locations: Optional[List[List[int]]],
        input_ids: torch.Tensor,
        logits_processor: Optional[LogitsProcessorList],
        unfinished_sequences: torch.Tensor,
        pad_token_id: int,
        is_prefill: bool,
    ) -> torch.Tensor:
        next_tokens_scores: torch.Tensor
        if is_prefill:
            # outputs should be logits which would have shape of [batch, seq_len, embedding_dimension] # noqa
            # loop through each batch and find the logit location due to attention_packing
            next_token_logits = []
            for single_batch_logit, single_batch_logit_target_location in zip(
                logits, logit_target_locations
            ):
                assert single_batch_logit.dim() == 2
                for logit_target in single_batch_logit_target_location:
                    # logit target will just be index
                    next_token_logits.append(
                        single_batch_logit[logit_target]
                    )  # will be [embedding_dimension]

            # stack this back to [batch, vocab_size]
            next_token_logits = torch.stack(next_token_logits)

        else:
            next_token_logits = logits[:, 0, :]  # for decode seq_len would just be 1
        next_tokens_scores = logits_processor(input_ids, next_token_logits)
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)
        next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
            1 - unfinished_sequences
        )
        return next_tokens

    def create_key_value_blocks(
        self,
        batch_size: int,
        bucket_size: int,
        kv_dtype: torch.dtype,
        device: torch.device,
        device_map: Optional[Dict[str, int]] = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        batch_size = min(batch_size, MAX_BATCH_SIZE)

        key_value_blocks = create_key_value_blocks(
            model_config=self.model_config,
            batch_size=batch_size,
            block_size=BLOCK_SIZE,
            device=device,
            bucket_size=bucket_size,
            kv_dtype=kv_dtype,
            device_map=device_map,
        )

        _, block_size, _, _ = key_value_blocks[0][0].shape

        if bucket_size % block_size != 0:
            raise ValueError(
                f"Bucket size ({bucket_size}) should be divisible by block size ({block_size})"
            )

        if BLOCK_SIZE != 1:
            raise ValueError(
                "Block size is fixed for RNGD architecture. Got block_size: {block_size} != 1"
            )

        return key_value_blocks

    def initialize_key_value_block_indices(
        self, key_value_blocks: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        block_indices, block_size, _, _ = key_value_blocks[0][0].shape
        self.block_size = block_size

        self.active_key_block_indices: List[List[int]] = []
        self.active_value_block_indices: List[List[int]] = []

        # Below fields keep track of prompt block indices which are shared across beam candidates
        self.prompt_key_block_indices: List[List[int]] = []
        self.prompt_value_block_indices: List[List[int]] = []

        self.available_block_indices = list(range(1, block_indices))
        self.zero_block_index = 0  # this is a special zero block
        self.total_block_count = block_indices

    def move_kv_cache_block_in_place(
        self,
        seq_idx: int,
        new_location: torch.Tensor,
        existing_block_indices: List[List[int]],
    ) -> None:
        # new key location should always be shape [batch, 1]
        for single_batch_block_indices, new_index in zip(
            existing_block_indices, new_location
        ):
            single_batch_block_indices[seq_idx] = new_index.item()

    def reset(self):
        self.active_key_block_indices: List[List[int]] = []
        self.active_value_block_indices: List[List[int]] = []
        self.available_block_indices = list(range(1, self.total_block_count))


def greedy_attention_packing(
    input_ids: torch.Tensor,
    bucketized_attention_mask: torch.Tensor,
    new_key_location: torch.Tensor,
    new_value_location: torch.Tensor,
    pad_token_id: int,
    compact_mask: bool = False,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    List[List[int]],
    torch.Tensor,
    torch.Tensor,
]:
    """
    return (packed_attention_mask, packed_input_ids, causal_mask, packed_position_ids, logit_target_locations, packed_new_key_location, packed_new_value_location)
    """  # noqa: E501
    assert input_ids.shape == bucketized_attention_mask.shape
    assert bucketized_attention_mask.shape == new_key_location.shape
    assert bucketized_attention_mask.shape == new_value_location.shape

    logit_target_locations = []
    (original_batch, bucket_size) = bucketized_attention_mask.shape

    # split attention mask by batch
    batch_real_len = []
    for single_batch in bucketized_attention_mask:
        num_real_token = single_batch.sum().item()
        batch_real_len.append(num_real_token)

    # find real tokens
    # first convert all padding tensors to 0
    # This ensures that non_zero_indices contains all input_ids that are not padding tokens,
    # regardless of the padding token value.
    non_zero_indices = (bucketized_attention_mask != 0).nonzero().tolist()
    real_locations = []
    for i, real_len in enumerate(batch_real_len):
        locations = [non_zero_indices.pop(0)[1] for _ in range(real_len)]
        start = locations[0]
        end = locations[-1] + 1
        real_locations.append((i, start, end))

    marker = bucket_size
    target_locations: List[List[Tuple[int, int]]] = []  # List of List
    temp_indices = []
    for i in range(original_batch):
        cur_len = batch_real_len[i]
        if marker - cur_len < 0 or len(temp_indices) >= MAX_PACKING_PER_ROW:
            # we cannot pack so start a new row
            target_locations.append(temp_indices)
            temp_indices = []
            marker = bucket_size

        temp_indices.append((marker - cur_len, marker))
        marker -= cur_len

    # push the last row into the target locations
    target_locations.append(temp_indices)

    packed_batch_size = len(target_locations)

    # initialize attention mask
    packed_shape = (packed_batch_size, bucket_size)

    packed_attention_mask = torch.zeros(packed_shape, dtype=torch.bool)
    packed_input_ids = torch.full(
        packed_shape, fill_value=pad_token_id, dtype=torch.int32
    )
    packed_new_key_location = torch.zeros(packed_shape, dtype=torch.int32)
    packed_new_value_location = torch.zeros(packed_shape, dtype=torch.int32)
    position_ids = torch.ones(packed_shape, dtype=torch.long)

    # initialize causal mask
    if compact_mask:
        causal_mask = torch.zeros((packed_batch_size, bucket_size), dtype=torch.uint8)
    else:
        causal_mask = torch.zeros(
            (packed_batch_size, bucket_size, bucket_size), dtype=torch.bool
        )

    # fill the new attention mask and mark the logit locations
    for index, target_location in enumerate(target_locations):
        # record new target locations
        logit_target_location = []
        for packing_idx, (start, end) in enumerate(target_location):
            (original_index, original_start, original_end) = real_locations.pop(0)
            packed_attention_mask[index][start:end] = True
            packed_input_ids[index][start:end] = input_ids[original_index][
                original_start:original_end
            ]
            packed_new_key_location[index][start:end] = new_key_location[
                original_index
            ][original_start:original_end]
            packed_new_value_location[index][start:end] = new_value_location[
                original_index
            ][original_start:original_end]
            position_ids[index][start:end] = torch.arange(end - start)
            logit_target_location.append(end - 1)

            if compact_mask:
                mask_value = packing_idx + 1  # 0 is reserved for padding
                causal_mask[index][start:end] = mask_value
            else:
                causal_mask[index][start:end, start:end] = torch.tril(
                    torch.ones((end - start, end - start), dtype=torch.bool)
                )
        logit_target_locations.append(logit_target_location)
    return (
        packed_attention_mask,
        packed_input_ids,
        causal_mask,
        logit_target_locations,
        position_ids,
        packed_new_key_location,
        packed_new_value_location,
    )


def get_model_dimensions(config: PretrainedConfig) -> Tuple[int, int, int]:
    if isinstance(config, GPTJConfig):
        num_heads = config.n_head
        embedding_dim = config.n_embd
        num_layers = config.n_layer
        head_size = int(embedding_dim / num_heads)
    elif isinstance(config, LlamaConfig):
        num_heads = config.num_attention_heads
        embedding_dim = config.hidden_size
        head_size = int(embedding_dim / num_heads)
        # Adjust num_heads if attention is grouped-query attention (GQA)
        if num_heads > config.num_key_value_heads:
            num_heads = config.num_key_value_heads
        num_layers = config.num_hidden_layers
    else:
        raise ValueError(f"Unsupported config type: {type(config)}")
    return num_layers, num_heads, head_size


def get_key_value_blocks(
    num_layers: int,
    num_heads: int,
    head_size: int,
    num_blocks: int,
    block_size: int,
    kv_dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
    device_map: Optional[Dict[str, int]] = None,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    # Example
    # num_layers = 2
    # num_heads = 8
    # head_size = 64
    # num_blocks = 129 (1 + 2 * 16 (bucket_size) * 4 (batch_size))
    # block_size = 16
    # block_shape = (129, 16, 8, 64)
    block_shape = (
        num_blocks,
        block_size,
        num_heads,
        head_size,
    )

    key_value_blocks = []
    for layer_idx in range(num_layers):
        if device_map is not None:
            device = torch.device(f"cuda:{device_map[str(layer_idx)]}")
        key_value_blocks.append(
            (
                # key shape: (num_blocks, block_size, num_heads, head_size) = (129, 16, 8, 64)
                torch.zeros(block_shape, dtype=kv_dtype).to(device),
                # value shape: (num_blocks, block_size, num_heads, head_size) = (129, 16, 8, 64)
                torch.zeros(block_shape, dtype=kv_dtype).to(device),
            )
        )
    return key_value_blocks


def create_key_value_blocks(
    model_config: PretrainedConfig,
    bucket_size: int,
    batch_size: int,
    kv_dtype: torch.dtype,
    block_size: int,
    device: torch.device,
    device_map: Optional[Dict[str, int]] = None,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    num_layers, num_heads, head_size = get_model_dimensions(model_config)

    # num_blocks = 1 (dummy pad token) + bucket_size (max_length) * batch_size * 2 (for each key and value) # noqa
    # Example
    # bucket_size = 16
    # batch_size = 4
    # num_blocks = 1 + 2 * 16 * 4 = 129
    num_blocks = 1 + 2 * bucket_size * batch_size

    key_value_blocks = get_key_value_blocks(
        num_layers,
        num_heads,
        head_size,
        num_blocks,
        block_size,
        kv_dtype=kv_dtype,
        device=device,
        device_map=device_map,
    )
    if len(key_value_blocks) != num_layers:
        raise ValueError(
            f"Key-value blocks should be created for all layers. Got len(key_value_blocks): \
                {len(key_value_blocks)} != num_layers: {num_layers}"
        )

    return key_value_blocks


def handle_outputs(
    outputs: Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]],
) -> torch.Tensor:
    # handle outputs differently based on prefill vs decode

    # SUPPORTED_GENERATION_RETURN_DICT_TYPES[1],
    # i.e., CausalLMOutputWithCrossAttentions is not yet checked.
    if isinstance(outputs, SUPPORTED_GENERATION_RETURN_DICT_TYPES[0]):
        logits = outputs.to_tuple()[0]
    elif isinstance(outputs, Tuple):
        logits = outputs[0]
    elif isinstance(outputs, Dict):
        logits = outputs["logits"]
    else:
        raise ValueError(f"Unsupported generation output type: {type(outputs)}")
    return logits
