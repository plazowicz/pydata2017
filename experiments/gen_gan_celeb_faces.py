import argparse
import os.path as op

import numpy as np
import tensorflow as tf
import cv2

import config
from models.celeb_gan import load_gen_with_weights, read_settings_with_weights
from utils.logger import log
from utils import img_ops


@log
class GanCelebFacesGenerator(object):

    def __init__(self, celeb_faces_dir, samples_num, out_weights_dir, from_iteration):
        self.celeb_faces_dir = celeb_faces_dir
        self.samples_num = samples_num
        self.out_weights_dir = out_weights_dir
        self.from_iteration = from_iteration

        _, self.gen_weights_path = self.__discr_gen_weights_path()
        _, self.z_dim = read_settings_with_weights(self.gen_weights_path)

    def generate_faces(self, out_path):
        sess = tf.InteractiveSession()
        with tf.device("/gpu:%d" % config.GPU_ID):
            z = tf.random_uniform(shape=(1, self.z_dim), minval=-1, maxval=1)
            gen_out, _ = load_gen_with_weights(sess, z, 1, self.gen_weights_path)
            x_generated = gen_out.outputs

        gen_images = []
        for i in xrange(self.samples_num):
            gen_img = sess.run(x_generated)
            gen_images.append(gen_img[0])

            self.logger.info("Generated %d/%d images" % (i, self.samples_num))

        self.__save_gen_images(gen_images, out_path)

    def __discr_gen_weights_path(self):
        discr_params_path = op.join(self.out_weights_dir, "gan_discr_%d.npz" % self.from_iteration)
        gen_params_path = op.join(self.out_weights_dir, "gan_gen_%d.npz" % self.from_iteration)
        return discr_params_path, gen_params_path

    @staticmethod
    def __save_gen_images(gen_images_list, out_path):
        imgs = [img_ops.unbound_image_values(gen_img) for gen_img in gen_images_list]
        imgs = np.array(imgs)
        grid_size = int(np.sqrt(imgs.shape[0]))
        big_img = img_ops.merge_images_to_grid(imgs, (grid_size, grid_size))

        cv2.imwrite(out_path, big_img)


def main():
    args = parse_args()
    celeb_faces_generator = GanCelebFacesGenerator(args.celeb_faces_dir, args.how_many_samples,
                                                   args.out_weights_dir, args.from_iteration)
    out_vis_path = args.out_vis_path
    celeb_faces_generator.generate_faces(out_vis_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--celeb_faces_dir', default=config.CELEB_FACES_DIR)
    parser.add_argument('--how_many_samples', type=int, default=64)
    parser.add_argument('out_weights_dir')
    parser.add_argument('from_iteration', type=int)
    parser.add_argument('out_vis_path')
    return parser.parse_args()


if __name__ == '__main__':
    main()