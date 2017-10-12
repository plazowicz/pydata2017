import argparse
import config

from trainers.vae_trainer import VaeTrainer
from datasets.celeb import CelebDataset

from utils import fs_utils


def train_celeb_vae(args):
    celeb_files_list = fs_utils.load_celeb_files(args.celeb_faces_dir)
    transformer = CelebDataset(args.crop_size, args.image_size, celeb_files_list, 64)
    trainer = VaeTrainer(args.latent_dim, transformer, args.out_weights_dir)

    print("Found %d celeb images" % len(celeb_files_list))
    trainer.train(args.epochs_num)


def main():
    args = parse_args()
    fs_utils.create_dir_if_not_exists(args.out_weights_dir)
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
