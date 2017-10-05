import argparse
import os.path as op

import cv2
import numpy as np
import tensorflow as tf

import config
from transformers.celeb import CelebDsTransformer
from models.celeb import read_settings_from_weights, load_enc_with_weights, load_gen_with_weights
from utils import fs_utils
from utils.logger import log
from utils import img_ops


@log
class VaeCelebFacesGenerator(object):

    def __init__(self, celeb_faces_dir, samples_num, crop_size, out_weights_dir, from_iteration):
        self.celeb_faces_dir = celeb_faces_dir
        self.samples_num = samples_num
        self.out_weights_dir = out_weights_dir
        self.from_iteration = from_iteration
        self.crop_size = crop_size

        self.enc_path, self.gen_path = self.__enc_gen_weights_paths()
        self.transformer = self.__get_transformer()
        self.z_dim, self.img_size, _ = read_settings_from_weights(self.enc_path, self.gen_path)

    def reconstruct_faces(self, out_path):
        img_paths_to_reconstruct = self.__select_celeb_faces(self.samples_num)
        out_original_path = self.__add_suffix_to_path(out_path, "original")
        out_reconstr_path = self.__add_suffix_to_path(out_path, "reconstr")

        self.__save_original_images(img_paths_to_reconstruct, out_original_path)
        sess = tf.InteractiveSession()

        with tf.device("/gpu:%d" % config.GPU_ID):
            input_images = tf.placeholder(tf.float32, [1, self.img_size, self.img_size, 3])
            _, _, z_mean, z_cov_log_sq = load_enc_with_weights(sess, input_images, self.enc_path)

            eps = tf.random_normal(shape=(1, self.z_dim,), mean=0.0, stddev=1.0)
            z = z_mean + tf.multiply(tf.sqrt(tf.exp(z_cov_log_sq)), eps)

            gen_out, _ = load_gen_with_weights(sess, z, self.gen_path)
            x_reconstr = gen_out.outputs

        reconstr_images = []
        for i, img_path in enumerate(img_paths_to_reconstruct):
            preprocessed_img = self.transformer.get_img(img_path)
            reconstr_img = sess.run(x_reconstr, feed_dict={input_images: [preprocessed_img]})
            reconstr_images.append(reconstr_img[0])

            self.logger.info("Reconstructed %d/%d images" % (i, len(img_paths_to_reconstruct)))

        self.__save_gen_images(reconstr_images, out_reconstr_path)

    def generate_faces(self, out_path):
        out_gen_path = self.__add_suffix_to_path(out_path, "generated")
        sess = tf.InteractiveSession()
        with tf.device("/gpu:%d" % config.GPU_ID):
            z = tf.random_normal(shape=(1, self.z_dim), mean=0.0, stddev=1.0)
            gen_out, _ = load_gen_with_weights(sess, z, self.gen_path)
            x_generated = gen_out.outputs

        gen_images = []
        for i in xrange(self.samples_num):
            gen_img = sess.run(x_generated)
            gen_images.append(gen_img)

            self.logger.info("Generated %d/%d images" % (i, self.samples_num))

        self.__save_gen_images(gen_images, out_gen_path)

    def __select_celeb_faces(self, how_many):
        import random

        celeb_files_list = [op.join(self.celeb_faces_dir, p) for p in
                            fs_utils.load_files_from_dir(self.celeb_faces_dir, ('jpg', 'png', 'jpeg'))]
        random.shuffle(celeb_files_list)

        return celeb_files_list[:how_many]

    def __enc_gen_weights_paths(self):
        enc_path = op.join(self.out_weights_dir, "vae_enc_%d.npz" % self.from_iteration)
        gen_path = op.join(self.out_weights_dir, "vae_gen_%d.npz" % self.from_iteration)
        return enc_path, gen_path

    def __get_transformer(self):
        z_dim, img_size, _ = read_settings_from_weights(self.enc_path, self.gen_path)
        return CelebDsTransformer(self.crop_size, img_size)

    def __save_original_images(self, images_paths, out_path):
        imgs = []
        for img_path in images_paths:
            img = cv2.imread(img_path)
            cropped_img = img_ops.center_crop(img, self.crop_size, self.crop_size)
            resized_img = cv2.resize(cropped_img, (self.img_size, self.img_size))

            imgs.append(resized_img)

        imgs = np.array(imgs)
        grid_size = int(np.sqrt(imgs.shape[0]))
        big_img = img_ops.merge_images_to_grid(imgs, (grid_size, grid_size))

        cv2.imwrite(out_path, big_img)

    def __save_gen_images(self, gen_images_list, out_path):
        imgs = [self.transformer.inverse_transform(gen_img) for gen_img in gen_images_list]
        imgs = np.array(imgs)
        grid_size = int(np.sqrt(imgs.shape[0]))
        big_img = img_ops.merge_images_to_grid(imgs, (grid_size, grid_size))

        cv2.imwrite(out_path, big_img)

    @staticmethod
    def __add_suffix_to_path(out_path, suffix):
        base_path = op.basename(out_path)
        dir_path = op.dirname(out_path)
        name, ext = base_path.split('.')[:-1], base_path.split('.')[-1]
        name = '.'.join(name)
        return op.join(dir_path, "%s_%s.%s" % (name, suffix, ext))


def main():
    args = parse_args()
    celeb_faces_generator = VaeCelebFacesGenerator(args.celeb_faces_dir, args.how_many_samples, args.crop_size,
                                                   args.out_weights_dir, args.from_iteration)
    out_vis_path = args.out_vis_path
    celeb_faces_generator.reconstruct_faces(out_vis_path)
    celeb_faces_generator.generate_faces(out_vis_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--celeb_faces_dir', default=config.CELEB_FACES_DIR)
    parser.add_argument('--how_many_samples', type=int, default=64)
    parser.add_argument('--crop_size', type=int, default=150)
    parser.add_argument('out_weights_dir')
    parser.add_argument('from_iteration', type=int)
    parser.add_argument('out_vis_path')
    return parser.parse_args()


if __name__ == '__main__':
    main()
