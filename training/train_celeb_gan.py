import argparse

import config
from utils.fs_utils import load_celeb_files, create_dir_if_not_exists

from trainers.gan_trainer import GanTrainer
from datasets.celeb import CelebDataset


def train_celeb_gan(args):
    celeb_files_list = load_celeb_files(args.celeb_faces_dir)
    transformer = CelebDataset(args.crop_size, args.image_size, celeb_files_list, 64)
    trainer = GanTrainer(args.latent_dim, transformer, args.out_weights_dir)
    print("Found %d celeb images" % len(celeb_files_list))
    trainer.train(celeb_files_list, args.epochs_num)


def main():
    args = parse_args()
    create_dir_if_not_exists(args.out_weights_dir)
    train_celeb_gan(args)


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
