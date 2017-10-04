import os


def load_files_from_dir(dir_name, ext):
    return [p for p in os.listdir(dir_name) if p.endswith(ext)]
