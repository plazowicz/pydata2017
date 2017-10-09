import argparse

from trainers.ss.ss_gan_trainer import SSGanTrainer
from transformers.cifar import CifarDataset
from utils.train_utils import adam_learning_options


def train_cifar_gan(args):
    cifar_ds = CifarDataset(args.cifar_ds_path, 64, args.how_many_labeled)
    learning_options = adam_learning_options()
    learning_options['lr'] = args.learning_rate
    gan_cifar_trainer = SSGanTrainer(args.latent_dim, cifar_ds, 10, args.out_weights_dir, 64, learning_options)
    gan_cifar_trainer.train_ss(args.epochs_num)


def main():
    args = parse_args()
    train_cifar_gan(args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('cifar_ds_path')
    parser.add_argument('out_weights_dir')
    parser.add_argument('how_many_labeled', type=int)
    parser.add_argument('--latent_dim', type=int, default=100)
    parser.add_argument('--epochs_num', type=int, default=100)
    parser.add_argument('--learning_rate', type=float)
    return parser.parse_args()


if __name__ == '__main__':
    main()
