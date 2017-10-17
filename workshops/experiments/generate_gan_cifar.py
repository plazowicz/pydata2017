import argparse

from models.ss import cifar10_gan
from workshops.img_generation.gan import GanImgGenerator
from utils import img_ops


def main():
    args = parse_args()
    celeb_faces_generator = GanImgGenerator(args.gen_weights_path, cifar10_gan.load_gen_with_weights)
    out_vis_path = args.out_vis_path
    celeb_faces_generator.generate_images(args.how_many_samples, out_vis_path,
                                          transformer=lambda x: img_ops.unbound_image_values(x[:, :, ::-1]))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--how_many_samples', type=int, default=64)
    parser.add_argument('gen_weights_path')
    parser.add_argument('out_vis_path')
    return parser.parse_args()


if __name__ == '__main__':
    main()
