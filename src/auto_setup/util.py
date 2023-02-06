import os


def get_lastest_iteration(base_dir: str, req_prefix: str = "") -> str:
    """Get the latest iteration of a file in a directory.

    Args:
        base_dir (str): The directory to search.
        req_prefix (str, optional): The prefix of the file to search for. Defaults to "".

    Returns:
        str: The path to the latest iteration of the file."""

    filenames = [
        file
        for file in os.listdir(base_dir)
        if os.path.isfile(os.path.join(base_dir, file)) and file.startswith(req_prefix)
    ]

    if not filenames:
        return ""

    return os.path.join(base_dir, sorted(filenames)[-1])
