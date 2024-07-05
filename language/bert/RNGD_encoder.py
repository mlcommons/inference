from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch.fx import GraphModule
from torch.nn.functional import pad
from transformers import PreTrainedModel

MAX_PACKING_PER_ROW: int = 254


class BertMLPerfSubmissionEncoder:
    def __init__(
        self,
        model: PreTrainedModel,
        bucket_size: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        compact_mask: bool = False,
    ) -> None:
        self.model = model
        self.bucket_size = bucket_size
        if pad_token_id is None:
            raise ValueError(
                f"pad_token_id must be provided for {self.__class__.__name__}"
            )
        self.pad_token_id = pad_token_id
        self.compact_mask = compact_mask

    def encode(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor,
    ) -> List[Tensor]:
        # greedy attention packing bert will do left padding regardless of given input
        (
            input_ids,
            token_type_ids,
            attention_mask,
            position_ids,
            packed_target_locations,
        ) = greedy_attention_packing_bert(
            input_ids=bucket_pad(input_ids, self.bucket_size),
            token_type_ids=bucket_pad(token_type_ids, self.bucket_size),
            bucketized_attention_mask=bucket_pad(attention_mask, self.bucket_size),
            pad_token_id=self.pad_token_id,
            compact_mask=self.compact_mask,
        )

        model_kwargs = dict(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        # remove all concrete args from model_kwargs as they will not be used in the forward pass.
        if isinstance(self.model, GraphModule):
            for arg in self.model.concrete_args:
                if arg in model_kwargs:
                    del model_kwargs[arg]

        logits = self.model(**model_kwargs)

        outputs = []
        for batch_index, target_location in enumerate(packed_target_locations):
            for single_target_location in target_location:
                start, end = single_target_location
                single_logit = logits[batch_index][start:end]
                outputs.append(single_logit)

        return outputs


def bucket_pad(tensor: Tensor, bucket_size: Optional[int]) -> Tensor:
    if bucket_size is None:
        return tensor
    padding_size = bucket_size - tensor.shape[-1]
    return pad(tensor, (0, padding_size))


def greedy_attention_packing_bert(
    input_ids: torch.Tensor,
    token_type_ids: torch.Tensor,
    bucketized_attention_mask: torch.Tensor,
    pad_token_id: int,
    compact_mask: bool,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[List[Tuple[int, int]]]
]:
    """
    return  input_ids, token_type_ids, attention_mask, position_ids, target_locations
    """
    assert input_ids.shape == bucketized_attention_mask.shape

    logit_target_locations = []
    (original_batch, bucket_size) = bucketized_attention_mask.shape

    # split attention mask by batch
    batch_real_len = []
    for single_batch in bucketized_attention_mask:
        num_real_token = single_batch.sum().item()
        batch_real_len.append(num_real_token)

    # find real tokens
    # first convert all padding tensors to zero
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

    packed_input_ids = torch.full(
        packed_shape,
        fill_value=pad_token_id,
        dtype=torch.int32,
        device=input_ids.device,
    )
    packed_token_type_ids = torch.zeros(
        packed_shape, dtype=torch.int32, device=token_type_ids.device
    )
    position_ids = torch.ones(packed_shape, dtype=torch.long, device=input_ids.device)

    # initialize causal mask
    if compact_mask:
        packed_attention_mask = torch.zeros(
            (packed_batch_size, bucket_size),
            dtype=torch.uint8,
            device=bucketized_attention_mask.device,
        )
    else:
        packed_attention_mask = torch.zeros(
            (packed_batch_size, bucket_size, bucket_size),
            dtype=torch.bool,
            device=bucketized_attention_mask.device,
        )

    # fill the new attention mask and mark the logit locations
    for index, target_location in enumerate(target_locations):
        # record new target locations
        logit_target_location = []
        for packing_idx, (start, end) in enumerate(target_location):
            (original_index, original_start, original_end) = real_locations.pop(0)
            packed_input_ids[index][start:end] = input_ids[original_index][
                original_start:original_end
            ]
            packed_token_type_ids[index][start:end] = token_type_ids[original_index][
                original_start:original_end
            ]
            position_ids[index][start:end] = torch.arange(end - start)
            logit_target_location.append((start, end))

            if compact_mask:
                mask_value = packing_idx + 1  # 0 is reserved for padding
                packed_attention_mask[index][start:end] = mask_value
            else:
                packed_attention_mask[index][start:end, start:end] = torch.ones(
                    (end - start, end - start),
                    dtype=torch.bool,
                    device=bucketized_attention_mask.device,
                )

        logit_target_locations.append(logit_target_location)

    return (
        packed_input_ids,
        packed_token_type_ids,
        packed_attention_mask,
        position_ids,
        logit_target_locations,
    )


def get_neg_inf(dtype):
    if dtype == torch.float32:
        return float("-inf")
    elif dtype == torch.float64:
        return float("-inf")
    elif dtype == torch.float16:
        return torch.tensor(float("-inf"), dtype=torch.float16).item()
    elif dtype == torch.bfloat16:
        return torch.tensor(float("-inf"), dtype=torch.bfloat16).item()
    else:
        raise ValueError(
            f"Unsupported dtype: {dtype}. Supported dtypes are: \
                float32, float64, float16, bfloat16."
        )


def stack_tensors(
    tensors: List[torch.Tensor], max_shape: List[int], pad_value: Optional[float] = None
) -> torch.Tensor:
    # Check if all tensors have the same dtype
    dtypes = {tensor.dtype for tensor in tensors}
    if len(dtypes) > 1:
        raise ValueError("All tensors must have the same dtype.")

    # Raise an error if the dtype is not one of the supported types
    dtype = dtypes.pop()
    if dtype not in {torch.float32, torch.float64, torch.float16, torch.bfloat16}:
        raise ValueError(
            f"Unsupported dtype: {dtype}. \
                Supported dtypes are: float32, float64, float16, bfloat16."
        )

    # Find the maximum shape in each dimension if not provided
    if max_shape is None:
        max_shape = [max(sizes) for sizes in zip(*[tensor.shape for tensor in tensors])]

    # Determine the padding value based on the tensor data type if not provided
    if pad_value is None:
        pad_value = get_neg_inf(dtype)

    # Pad each tensor to the maximum shape
    padded_tensors = []
    for tensor in tensors:
        padding = []
        for dim_size, max_size in zip(tensor.shape[::-1], max_shape[::-1]):
            padding.extend([0, max_size - dim_size])
        padded_tensor = pad(tensor, padding, value=pad_value)
        padded_tensors.append(padded_tensor)

    # Stack the padded tensors
    stacked_tensor = torch.stack(padded_tensors)
    return stacked_tensor
