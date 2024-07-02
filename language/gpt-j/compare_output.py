import yaml

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def compare_yaml_lists(file_path1, file_path2):
    list1 = load_yaml(file_path1)
    list2 = load_yaml(file_path2)
    count  = 0
    for idx in range(len(list1)):
        if list1[idx] != list2[idx]:
            print(f'fp_model output: {list1[idx]}')
            print(f'quantsim_model output: {list2[idx]}')
            count +=1
            
    print(f"{count} samples are different.")
# Specify the file paths for the two YAML files
file_path1 = 'output_fp.yaml'
file_path2 = 'output_quantsim_no_fakequant.yaml'

# Compare the two YAML files
compare_yaml_lists(file_path1, file_path2)