import argparse

from models import celeb_gan
from models.gan_generator import GanImgGenerator


def main():
    args = parse_args()
    celeb_faces_generator = GanImgGenerator(args.how_many_samples, args.out_weights_dir, args.from_iteration,
                                            celeb_gan.load_gen_with_weights)
    out_vis_path = args.out_vis_path
    celeb_faces_generator.generate_images(out_vis_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--how_many_samples', type=int, default=64)
    parser.add_argument('out_weights_dir')
    parser.add_argument('from_iteration', type=int)
    parser.add_argument('out_vis_path')
    return parser.parse_args()


if __name__ == '__main__':
    main()