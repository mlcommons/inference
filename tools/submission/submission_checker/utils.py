import os


def list_dir(*path):
    path = os.path.join(*path)
    return sorted([f for f in os.listdir(
        path) if os.path.isdir(os.path.join(path, f))])


def list_files(*path):
    path = os.path.join(*path)
    return sorted([f for f in os.listdir(
        path) if os.path.isfile(os.path.join(path, f))])


def list_empty_dirs_recursively(*path):
    path = os.path.join(*path)
    return [dirpath for dirpath, dirs, files in os.walk(
        path) if not dirs and not files]


def list_dirs_recursively(*path):
    path = os.path.join(*path)
    return [dirpath for dirpath, dirs, files in os.walk(path)]


def list_files_recursively(*path):
    path = os.path.join(*path)
    return [
        os.path.join(dirpath, file)
        for dirpath, dirs, files in os.walk(path)
        for file in files
    ]


def check_extra_files(path, target_files):
    missing_files = []
    check_pass = True
    folders = list_dir(path)
    for dir in target_files.keys():
        if dir not in folders:
            check_pass = False
            missing_files.append(os.path.join(path, dir))
        else:
            files = [f.split(".")[0]
                     for f in list_files(os.path.join(path, dir))]
            for target_file in target_files[dir]:
                if target_file not in files:
                    check_pass = False
                    missing_files.append(
                        f"{os.path.join(path, dir, target_file)}.png")
            if "captions" not in files:
                missing_files.append(
                    f"{os.path.join(path, dir, 'captions.txt')}")
    return check_pass, missing_files


def split_path(m):
    return m.replace("\\", "/").split("/")


def get_boolean(s):
    if s is None:
        return False
    elif isinstance(s, bool):
        return s
    elif isinstance(s, str):
        return s.lower() == "true"
    elif isinstance(s, int):
        return bool(s)
    else:
        raise TypeError(
            f"Variable should be bool, string or int, got {type(s)} instead"
        )

def merge_two_dict(x, y):
    z = x.copy()
    for key in y:
        if key not in z:
            z[key] = y[key]
        else:
            z[key] += y[key]
    return z

def sum_dict_values(x):
    count = 0
    for key in x:
        count += x[key]
    return count