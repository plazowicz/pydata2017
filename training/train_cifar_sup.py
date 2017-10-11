import argparse

from models.ss.cifar10_gan import discriminator
from trainers.sup_cnn_trainer import SupCnnTrainer
from transformers.cifar import SupervisedCifarDataset
from utils.train_utils import adam_learning_options


def cifar_gan_discriminator(input_images, train_mode, reuse, classes_num):
    return discriminator(input_images, train_mode, reuse, classes_num, return_previous=False)


def train_cifar_gan(args):
    cifar_ds = SupervisedCifarDataset(args.cifar_ds_path, 64, args.how_many_labeled)
    learning_options = adam_learning_options()
    learning_options['lr'] = args.learning_rate
    cifar_trainer = SupCnnTrainer(cifar_ds, 64, cifar_gan_discriminator, learning_options)
    cifar_trainer.train(args.epochs_num, args.testing_interval)


def main():
    args = parse_args()
    train_cifar_gan(args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('cifar_ds_path')
    parser.add_argument('how_many_labeled', type=int)
    parser.add_argument('--epochs_num', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--testing_interval', type=int, default=100)
    return parser.parse_args()


if __name__ == '__main__':
    main()
