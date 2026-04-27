import json

from loader import standardize_accelerator_name


def load_system(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_accel = data.get("accelerator_model_name", "")
    accelerators_per_node = int(data.get("accelerators_per_node", 1))
    number_of_nodes = int(data.get("number_of_nodes", 1))
    total_accelerators = accelerators_per_node * number_of_nodes

    return {
        "accelerator_model_name": raw_accel,
        "accelerator_std": standardize_accelerator_name(raw_accel),
        "accelerators_per_node": accelerators_per_node,
        "number_of_nodes": number_of_nodes,
        "total_accelerators": total_accelerators,
        "system_name": data.get("system_name", ""),
        "submitter": data.get("submitter", ""),
        "framework": data.get("framework", ""),
        "operating_system": data.get("operating_system", ""),
    }
