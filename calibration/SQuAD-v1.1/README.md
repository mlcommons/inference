The integers in bert_calibration_features.txt correspond to 100 randomly selected indices in the list of features generated from dev-v1.1.json using [convert_examples_to_features()](https://github.com/mlcommons/inference/blob/master/language/bert/create_squad_data.py#L249) with a doc_stride of 128 and a max_seq_len of 384.

The values in bert_calibration_qas_ids.txt correspond to 100 randomly selected qas ids in the dev-v1.1.json file.

Please only use at most 1 calibration file from this folder for calibration.

