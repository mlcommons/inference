import yaml
import numpy as np




result = {}
for part_idx in range(4):
    with open(f"result_qlevel_4_cnn_eval_part_{part_idx}.yaml", "r") as yaml_file:
        data = yaml.safe_load(yaml_file)
    for key, value in data.items():
        result[key] = value if key not in result.keys() else result[key]+ value

result = {k: round(np.mean(v) * 100, 4) for k, v in result.items()}
print("\nResults\n")
print(result)

