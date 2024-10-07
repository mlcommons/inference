import os 
import pandas as pd
import torch

def gen_test_data(data_path, n_data=1):
    if not os.path.isfile(data_path):
        print("Dataset {} not found. Please check that the path is correct".format(data_path))
    

    eval_dataset = pd.read_pickle(data_path)
    
    input_tokens = eval_dataset['tok_input']
    
    data_list = []
    
    for input_token in input_tokens[:n_data]:
        test_data = {
                "input_ids": torch.tensor(input_token, dtype=torch.int32, device='cuda').view(1,-1),
                "attention_mask": torch.ones((1,len(input_token)), dtype=torch.int32, device='cuda'),
            }
        data_list.append(test_data)
    
    print(data_list[0])
    # test_data_cpu를 파일로 저장합니다.
    torch.save(data_list[0], '/home/home-mcl/shared_data/dataset/open-orca/validation/test_data.pt')
    
    
    return data_list