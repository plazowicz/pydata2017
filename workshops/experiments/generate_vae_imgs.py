import argparse

from models import celeb_vae
from workshops.img_generation.vae import VaeImgGenerator


def main():
    args = parse_args()
    celeb_faces_generator = VaeImgGenerator(args.gen_weights_path, celeb_vae.load_gen_with_weights)
    out_vis_path = args.out_vis_path
    celeb_faces_generator.generate_images(args.how_many_samples, out_vis_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--how_many_samples', type=int, default=64)
    parser.add_argument('gen_weights_path')
    parser.add_argument('out_vis_path')
    return parser.parse_args()


if __name__ == '__main__':
    main()
