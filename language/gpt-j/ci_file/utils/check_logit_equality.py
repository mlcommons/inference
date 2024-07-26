import torch
import pickle

def load_all_tensors_from_pickle(file_path, mcm_module_name):
    tensor_list = []
    with open(file_path, "rb") as file:
        while True:
            try:
                result_tensor = pickle.load(file)
                layer_name = next(iter(result_tensor))
                if mcm_module_name in layer_name:
                    tensor_list.append(result_tensor[mcm_module_name]["output_before_rounding"])
                    
            except EOFError:
                break
    return tensor_list
            

def check_mcp_dump_output(
    golden_model_file_path,
    comparison_model_file_path,
    mcm_module_name,
    is_decode,
):

    golden_tensor_list = load_all_tensors_from_pickle(golden_model_file_path, mcm_module_name)
    comparison_tensor_list = load_all_tensors_from_pickle(comparison_model_file_path, mcm_module_name)

    assert len(golden_tensor_list) == len(comparison_tensor_list)
    
    for idx in range(len(golden_tensor_list)):
        valid_seq_len = golden_tensor_list[idx].shape[1] if not is_decode else 1
        
        if golden_tensor_list[idx].shape != comparison_tensor_list[idx].shape: 
            #If true, packing would have been applied in furiosa-llm-generator due to the short length of input_ids
            is_successful = torch.equal(golden_tensor_list[idx][0, -1:, :][0].unsqueeze(0), comparison_tensor_list[idx][0, -1:, :][0].unsqueeze(0))
        else:
            is_successful = torch.equal(golden_tensor_list[idx][:, -valid_seq_len:, :], comparison_tensor_list[idx][:, -valid_seq_len:, :])

        if not is_successful:
            raise ValueError("Logits comparison test failed.")
        
    return True


def compare_logits(logit_folder_path):
    
    check_mcp_dump_output(golden_model_file_path = logit_folder_path + '/golden_prefill_logits.pkl',
                    comparison_model_file_path = logit_folder_path + '/submission_prefill_logits.pkl',
                    mcm_module_name = 'lm_head',
                    is_decode = False)
    
    check_mcp_dump_output(golden_model_file_path = logit_folder_path + '/golden_decode_logits.pkl',
                    comparison_model_file_path = logit_folder_path + '/submission_decode_logits.pkl',
                    mcm_module_name = 'lm_head',
                    is_decode = True)