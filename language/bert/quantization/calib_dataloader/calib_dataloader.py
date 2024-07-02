import os
import torch
from torch.utils.data import DataLoader


class MyCollate:
    def __call__(self, batch):

        input_ids = torch.stack([item['input_ids'] for item in batch], dim=0)
        attention_mask = torch.stack([item['attention_mask'] for item in batch], dim=0)
        token_type_ids = torch.stack([item['token_type_ids'] for item in batch], dim=0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
        }

def make_dataloader(qsl, batch_size, n_calib):

    file_path = os.path.join(
        os.path.realpath(__file__)[0:os.path.realpath(__file__).find('language')], 
        'calibration', 
        'SQuAD-v1.1',
        'bert_calibration_features.txt',)
    with open(file_path, 'r') as fp:
        lines = fp.readlines()

    calib_data_indice_list = []
    for line in lines:
        numbers = [int(num) for num in line.split('\n') if num.isdigit()]
        calib_data_indice_list.extend(numbers)
    
    calib_eval_features = [qsl.eval_features[i] for i in calib_data_indice_list]

    data_list = []
    if n_calib != -1:
        calib_eval_features = calib_eval_features[0:n_calib]
    for feature in calib_eval_features:
        data_list.append({
            'input_ids': torch.LongTensor(feature.input_ids),
            'attention_mask': torch.LongTensor(feature.input_mask),
            'token_type_ids': torch.LongTensor(feature.segment_ids),
        })
    
    dataloader = DataLoader(data_list, batch_size=batch_size)

    return dataloader
