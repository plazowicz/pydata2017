import os
import os.path as op


def load_files_from_dir(dir_name, ext):
    return [p for p in os.listdir(dir_name) if p.endswith(ext)]


def create_dir_if_not_exists(dir_name):
    if not op.exists(dir_name):
        os.makedirs(dir_name)
