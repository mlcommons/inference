import pickle
import torch


def is_logit_same(
    logit_file_path,
    golden_model_test_sample,
    comparison_model_test_sample,
    mcm_name_to_check,
):

    golden_file_path = logit_file_path + '/golden_logits.pkl'
    comparison_model_file_path = logit_file_path + '/submission_logits.pkl'

    comparison_file = open(comparison_model_file_path, "rb")

    read_golden_file = True

    with open(golden_file_path, "rb") as golden_file:
        while read_golden_file:
            try:
                golden_result = pickle.load(golden_file)
                golden_layer_name = next(iter(golden_result))

                if (
                    mcm_name_to_check is not None
                    and not mcm_name_to_check in golden_layer_name
                ):
                    continue

                while True:
                    comparison_result = pickle.load(comparison_file)
                    comparison_layer_name = next(iter(comparison_result))

                    if golden_layer_name in comparison_layer_name:
                        read_golden_file = False
                        break

            except EOFError:
                print(
                    f"It's end of file. Please check file path {golden_file_path} again."
                )
                break

    golden_result = golden_result[golden_layer_name]
    comparison_result = comparison_result[comparison_layer_name]

    try:
        golden_output = golden_result["output_before_rounding"]
        comparison_output = comparison_result["output_before_rounding"]
    except:  # noqa: E722
        golden_output = golden_result["output"]
        comparison_output = comparison_result["output"]

    if golden_output.dtype != comparison_output.dtype:
        raise ValueError("Invalid values to compare.")

    # ---------------------------------------------------------
    # Masking only valid_seq
    # ---------------------------------------------------------
    golden_input_ids = golden_model_test_sample["input_ids"]
    comparison_input_ids = comparison_model_test_sample["input_ids"]

    batch_size = golden_input_ids.shape[0]

    for batch_idx in range(batch_size):
        max_seq_length = golden_input_ids[batch_idx].shape[0]
        golden_extract_nonzero_locations = torch.nonzero(golden_input_ids[batch_idx])

        golden_valid_seq_length = (
            int(golden_extract_nonzero_locations[-1] + 1)
            - golden_extract_nonzero_locations[0]
        )

        if golden_valid_seq_length == 0:
            raise ValueError("Invalid target locations.")

        # mlperf_submission 모델의 input preprocessing은 generator 내부에서 이루어지므로,
        # golden valid seq length를 기준으로 comparison_output의 valid location 추출.

        comparison_extract_nonzero_locations = [
            (max_seq_length - 1) - golden_valid_seq_length + 1,
            max_seq_length - 1,
        ]

        device = golden_output.device
        valid_golden_output = golden_output[
            batch_idx,
            int(golden_extract_nonzero_locations[0]) : int(
                golden_extract_nonzero_locations[-1] + 1
            ),
        ]
        valid_comparison_output = comparison_output[
            batch_idx,
            int(comparison_extract_nonzero_locations[0]) : int(
                comparison_extract_nonzero_locations[-1] + 1
            ),
        ]

        if not torch.equal(valid_golden_output, valid_comparison_output):
            raise ValueError("Logits comparison test failed.")

    return True
