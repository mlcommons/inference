import argparse
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--created_quant_param_path", help="path of the quant param file that is just created through calibrate.py")
    parser.add_argument("--released_quant_param_path", help = "path of the quant param file that is released")


    args = parser.parse_args()
    return args



def is_qparam_same(created_quant_param_path, released_quant_param_path, print_log=False):
    import numpy as np


    golden_qparam = np.load(created_quant_param_path, allow_pickle=True).item()
    comparison_qparam = np.load(released_quant_param_path, allow_pickle=True).item()

    failure_count = 0
    for module_name, module_qparam in comparison_qparam.items():
        try:
            golden_data = golden_qparam[module_name]
        except:
            continue

        for qparam_name, qparam in golden_data.items():
            if qparam is None:
                continue

            if not np.array_equal(module_qparam[qparam_name], qparam):
                failure_count = failure_count + 1
                if print_log:
                    print(
                        "Failed ",
                        module_name,
                        qparam_name,
                        (abs(module_qparam[qparam_name] - qparam) / abs(qparam)).max(),
                    )
            else:
                if print_log:
                    print("Passed ", module_name, qparam_name)

    if failure_count != 0:
        print(failure_count)
        raise ValueError("Qparam comparision test failed.")

    return True

if __name__ == "__main__":
    args = get_args()
    print(is_qparam_same(args.created_quant_param_path, args.released_quant_param_path))







