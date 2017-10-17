import os
import os.path as op


def load_files_from_dir(dir_name, ext):
    return [p for p in os.listdir(dir_name) if p.endswith(ext)]


def create_dir_if_not_exists(dir_name):
    if not op.exists(dir_name):
        os.makedirs(dir_name)


def load_celeb_files(celeb_faces_dir):
    return [op.join(celeb_faces_dir, p) for p in
            load_files_from_dir(celeb_faces_dir, ('jpg', 'png', 'jpeg'))]


def add_suffix_to_path(out_path, suffix):
        base_path = op.basename(out_path)
        dir_path = op.dirname(out_path)
        name, ext = base_path.split('.')[:-1], base_path.split('.')[-1]
        name = '.'.join(name)
        return op.join(dir_path, "%s_%s.%s" % (name, suffix, ext))
