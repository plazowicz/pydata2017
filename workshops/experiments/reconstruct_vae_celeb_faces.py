import argparse
import os.path as op

import numpy as np

from datasets.celeb import CelebDataset
from models.celeb_vae import load_enc_with_weights, load_gen_with_weights, read_gen_settings_with_weights
from workshops.vae_reconstruction import VaeImgReconstructor
from utils import net_utils
from utils import fs_utils
from utils.logger import log


def load_gen_weights(gen_params_path):
    return lambda sess, gen_input: load_gen_with_weights(sess, gen_input, gen_params_path)


def load_enc_weights(enc_params_path):
    return lambda sess, enc_input: load_enc_with_weights(sess, enc_input, enc_params_path)


def get_latent_dim(gen_params_path):
    return lambda: net_utils.read_latent_dim_from_gen_weights(gen_params_path)


@log
def get_celeb_faces_samples(celeb_faces_dir, crop_size, gen_params_path, how_many):
    img_size = read_gen_settings_with_weights(gen_params_path)
    celeb_faces_image_paths = sample_celeb_faces(celeb_faces_dir, how_many)
    get_celeb_faces_samples.logger.info("Sampled %d images paths from celeb dataset" % how_many)
    ds = CelebDataset(crop_size, img_size, celeb_faces_image_paths, how_many)
    return ds.generate_train_mb().next()


def sample_celeb_faces(celeb_faces_dir, how_many):
    import random

    celeb_files_list = [op.join(celeb_faces_dir, p) for p in
                        fs_utils.load_files_from_dir(celeb_faces_dir, ('jpg', 'png', 'jpeg'))]
    random.shuffle(celeb_files_list)

    return celeb_files_list[:how_many]


def main():
    args = parse_args()
    reconstructor = VaeImgReconstructor(load_gen_weights(args.gen_params_path),
                                        load_enc_weights(args.enc_params_path), get_latent_dim(args.gen_params_path))
    celeb_imgs = np.array(get_celeb_faces_samples(args.celeb_faces_dir, args.crop_size, args.gen_params_path,
                                                  args.how_many_samples))
    reconstructor.reconstruct_faces(args.out_path, celeb_imgs)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--how_many_samples', type=int, default=64)
    parser.add_argument('--crop_size', type=int, default=150)
    parser.add_argument('celeb_faces_dir')
    parser.add_argument('gen_params_path')
    parser.add_argument('enc_params_path')
    parser.add_argument('out_vis_path')
    return parser.parse_args()


if __name__ == '__main__':
    main()
