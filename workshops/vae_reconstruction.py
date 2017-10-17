import os.path as op

import tensorflow as tf
import numpy as np
import cv2

from utils import device_utils
from utils import img_ops
from utils.logger import log
from utils import fs_utils


@log
class VaeImgReconstructor(object):

    def __init__(self, load_enc_with_weights, load_gen_with_weights_func, get_latent_dim_func):
        self.load_enc_with_weights_func = load_enc_with_weights
        self.load_gen_with_weights_func = load_gen_with_weights_func
        self.z_dim = get_latent_dim_func()
        pass

    def reconstruct_faces(self, out_path, imgs_to_reconstruct, transformer=lambda x: img_ops.unbound_image_values(x)):
        out_original_path = fs_utils.add_suffix_to_path(out_path, "original")
        self.__save_original_images(imgs_to_reconstruct, out_original_path, transformer)

        img_size = imgs_to_reconstruct.shape[1]

        sess = tf.InteractiveSession()
        with tf.device(device_utils.get_device()):
            # TODO - create placeholder for input_images
            input_images = None
            z_mean, z_cov_log_sq = None, None

            eps = tf.random_normal(shape=(1, self.z_dim), mean=0.0, stddev=1.0)

            # TODO - sample z from encoder
            z = None
            gen_out, _ = self.load_gen_with_weights_func(sess, z)

        reconstr_images = []
        for i in xrange(imgs_to_reconstruct.shape[0]):
            # TODO - reconstruct image with graph defined above
            reconstr_img = None
            reconstr_images.append(reconstr_img[0])

        img_ops.save_gen_images(reconstr_images, out_path, transformer)

    @staticmethod
    def __save_original_images(imgs_to_reconstruct, out_path, transformer):

        imgs = np.array([transformer(i) for i in imgs_to_reconstruct])
        grid_size = int(np.sqrt(imgs.shape[0]))
        big_img = img_ops.merge_images_to_grid(imgs[:grid_size*grid_size], (grid_size, grid_size))

        cv2.imwrite(out_path, big_img)
