import numpy as np
import torch
import re

QPARAM_PATH = [
    # bert-large
    '../../data/furiosa_llm_modles_artifacts/quantized/furiosa-ai/mlperf-bert-large/mlperf_submission/W8A8/4L/quant_param.npy',
    '../../data/furiosa_llm_modles_artifacts/quantized/furiosa-ai/mlperf-bert-large/mlperf_submission/W8A8/24L/quant_param.npy',

    # gpt-j
    '../../data/furiosa_llm_modles_artifacts/quantized/furiosa-ai/mlperf-gpt-j-6b/mlperf_submission_slice/W8A8KV8/2L/quant_param.npy',
    '../../data/furiosa_llm_modles_artifacts/quantized/furiosa-ai/mlperf-gpt-j-6b/mlperf_submission_slice/W8A8KV8/4L/quant_param.npy',
    '../../data/furiosa_llm_modles_artifacts/quantized/furiosa-ai/mlperf-gpt-j-6b/mlperf_submission_slice/W8A8KV8/28L/quant_param.npy',

    # llama2-70b
    '../../data/furiosa_llm_modles_artifacts/quantized/meta-llama/Llama-2-70b-chat-hf/mlperf_submission_slice/W8A8KV8/2L/quant_param.npy',
    '../../data/furiosa_llm_modles_artifacts/quantized/meta-llama/Llama-2-70b-chat-hf/mlperf_submission_slice/W8A8KV8/4L/quant_param.npy',
    '../../data/furiosa_llm_modles_artifacts/quantized/meta-llama/Llama-2-70b-chat-hf/mlperf_submission_slice/W8A8KV8/80L/quant_param.npy',

    # llama3.1-8b
    '../../data/furiosa_llm_modles_artifacts/quantized/meta-llama/Meta-Llama-3.1-8B-Instruct/mlperf_submission_slice/W8A8KV8/4L/quant_param.npy',
    '../../data/furiosa_llm_modles_artifacts/quantized/meta-llama/Meta-Llama-3.1-8B-Instruct/mlperf_submission_slice/W8A8KV8/32L/quant_param.npy',

    # llama3.1-70b
    '../../data/furiosa_llm_modles_artifacts/quantized/meta-llama/Meta-Llama-3.1-70B-Instruct/mlperf_submission_slice/W8A8KV8/4L/quant_param.npy',
    '../../data/furiosa_llm_modles_artifacts/quantized/meta-llama/Meta-Llama-3.1-70B-Instruct/mlperf_submission_slice/W8A8KV8/80L/quant_param.npy',
]


N_BITS=8
eps=1e-4


def away_from_zero_round(input):
    return torch.trunc(input + torch.sign(input) * 0.5)


def calculate_zp(max, min):
    merged_scale = (max - min) / (2.0**N_BITS - 1)
    merged_scale = torch.where(merged_scale > 1e-7, merged_scale, 1e-7)

    min_bound = -(2.0 ** (N_BITS - 1))
    # max_bound = 2.0 ** (N_BITS - 1) - 1

    zero_point = min_bound - away_from_zero_round(min / merged_scale)

    return zero_point


def main():

    model_list_to_be_changed = []

    for key in QPARAM_PATH:
        org = np.load(key, allow_pickle=True).item()
        for k in org.keys():
            if not 'matmul' in k:
                continue
            if not int(re.search(r'\d+', k).group(0)) % 2 == 0:
                continue
            
            if org[k]['max'] is not None:
                _max = torch.tensor(org[k]['max'])
                _min = torch.tensor(org[k]['min'])
                zero_point = calculate_zp(_max, _min)

                if not abs(zero_point).sum() == 0.0:
                    model_list_to_be_changed.append(key)
                    break

    
    print("=============")
    print("model_list to be changed : ", model_list_to_be_changed)

    for key in model_list_to_be_changed:
        org = np.load(key, allow_pickle=True).item()

        for k in org.keys():
            if not 'matmul' in k:
                continue
            if not int(re.search(r'\d+', k).group(0)) % 2 == 0:
                continue

            if org[k]['max'] is not None:
                _max = torch.tensor(org[k]['max'])
                _min = torch.tensor(org[k]['min'])
                zero_point = calculate_zp(_max, _min)

                if not abs(zero_point).sum() == 0.0:
                    print(k)
                    mask = abs(_max) > abs(_min)
                    _max = torch.where(mask, _max, -_min)
                    _min = torch.where(mask, -_max, _min)
                    _min = _min - eps

                    zero_point = calculate_zp(_max, _min)

                    if not abs(zero_point).sum() == 0.0:
                        raise ValueError("non zero zero_point is still remaining.")
                    
                    org[k]['max'] = _max.numpy()
                    org[k]['min'] = _min.numpy()

        new_file_name = key.replace('.npy', '_zp2zero.npy')
        np.save(new_file_name, org, allow_pickle=True)

        print(new_file_name, " has been saved!")


if __name__ == "__main__":
    main()



