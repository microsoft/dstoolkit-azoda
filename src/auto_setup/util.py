import os


def get_lastest_iteration(base_dir, req_prefix=""):
    filenames = [
        f
        for f in os.listdir(base_dir)
        if os.path.isfile(os.path.join(base_dir, f)) and f.startswith(req_prefix)
    ]
    print("filenames", filenames)
    if not filenames:
        return ""
    return os.path.join(base_dir, sorted(filenames)[-1])
