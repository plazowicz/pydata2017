import os.path as op

import argparse
import config
from trainers.vae_trainer import VaeTrainer
from transformers.celeb import CelebDsTransformer

from utils import fs_utils


def load_celeb_files(celeb_faces_dir):
    return [op.join(celeb_faces_dir, p) for p in
            fs_utils.load_files_from_dir(celeb_faces_dir, ('jpg', 'png', 'jpeg'))]


def train_celeb_vae(args):
    transformer = CelebDsTransformer(args.crop_size, args.image_size)
    trainer = VaeTrainer(args.latent_dir, transformer, args.out_weights_dir)

    celeb_files_list = load_celeb_files(args.celeb_faces_dir)
    trainer.train(celeb_files_list, args.epochs_num)


def main():
    args = parse_args()
    train_celeb_vae(args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('out_weights_dir')
    parser.add_argument('--celeb_faces_dir', default=config.CELEB_FACES_DIR)
    parser.add_argument('--latent_dim', type=int, default=100)
    parser.add_argument('--epochs_num', type=int, default=30)
    parser.add_argument('--crop_size', type=int, default=150)
    parser.add_argument('--image_size', type=int, default=64)
    return parser.parse_args()


if __name__ == '__main__':
    main()
