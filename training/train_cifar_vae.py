import argparse

from trainers.vae_trainer import VaeTrainer
from datasets.cifar import UnsupervisedCifarDataSet
from utils.train_utils import adam_learning_options


def train_cifar_gan(args):
    cifar_ds = UnsupervisedCifarDataSet(args.cifar_ds_path, 64)
    learning_options = adam_learning_options()
    learning_options['lr'] = args.learning_rate
    vae_cifar_trainer = VaeTrainer(args.latent_dim, cifar_ds, args.out_weights_dir, learning_options)
    vae_cifar_trainer.train(args.epochs_num)


def main():
    args = parse_args()
    train_cifar_gan(args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('cifar_ds_path')
    parser.add_argument('out_weights_dir')
    parser.add_argument('--latent_dim', type=int, default=100)
    parser.add_argument('--epochs_num', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    return parser.parse_args()


if __name__ == '__main__':
    main()
