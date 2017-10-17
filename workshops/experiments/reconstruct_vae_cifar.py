import argparse

import numpy as np

from datasets.cifar import UnsupervisedCifarDataSet
from reconstruct_vae_celeb_faces import load_gen_weights, load_enc_weights, get_latent_dim
from workshops.vae_reconstruction import VaeImgReconstructor
from utils import img_ops


def get_cifar_samples(cifar_ds_path, how_many):
    ds = UnsupervisedCifarDataSet(cifar_ds_path, how_many)
    return ds.generate_train_mb().next()


def main():
    args = parse_args()
    reconstructor = VaeImgReconstructor(load_enc_weights(args.enc_params_path),
                                        load_gen_weights(args.gen_params_path), get_latent_dim(args.gen_params_path))
    cifar_imgs = np.array(get_cifar_samples(args.cifar_ds_path, args.how_many_samples))
    reconstructor.reconstruct_faces(args.out_vis_path, cifar_imgs,
                                    transformer=lambda x: img_ops.unbound_image_values(x[:, :, ::-1]))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--how_many_samples', type=int, default=64)
    parser.add_argument('cifar_ds_path')
    parser.add_argument('gen_params_path')
    parser.add_argument('enc_params_path')
    parser.add_argument('out_vis_path')
    return parser.parse_args()


if __name__ == '__main__':
    main()
