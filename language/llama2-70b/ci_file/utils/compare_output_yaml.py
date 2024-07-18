import yaml

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def compare_output_yaml(file_path_1, file_path_2):
    # Load YAML files

    yaml_content1 = load_yaml(file_path_1)
    yaml_content2 = load_yaml(file_path_2)
    
    if yaml_content1 == yaml_content2:
        print("The generators produced the same outputs")
    else:
        raise ValueError("Generation output comparison test failed.")
