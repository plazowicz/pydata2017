import os.path as op

import tensorflow as tf

import numpy as np
import cv2

from utils.logger import log
from utils import device_utils
from utils import img_ops
from utils import net_utils


@log
class GanImgGenerator(object):

    def __init__(self, samples_num, weights_dir, from_iteration, load_gen_weights_func,
                 get_latent_dim_func=net_utils.read_latent_dim_from_gen_weights):
        self.samples_num = samples_num

        _, self.gen_weights_path = self.__discr_gen_weights_path()
        self.gen_weights_path = net_utils.get_gan_generator_weights_path(weights_dir, from_iteration)
        self.load_gen_weights_func = load_gen_weights_func
        self.z_dim = get_latent_dim_func(self.gen_weights_path)

    def generate_faces(self, out_path, transformer=lambda x: x):
        sess = tf.InteractiveSession()
        with tf.device(device_utils.get_device()):
            z = tf.random_uniform(shape=(1, self.z_dim), minval=-1, maxval=1)
            gen_out, _ = self.load_gen_weights_func(sess, z, 1, self.gen_weights_path)
            x_generated = gen_out.outputs

        gen_images = []
        for i in xrange(self.samples_num):
            gen_img = sess.run(x_generated)
            gen_images.append(gen_img[0])

            self.logger.info("Generated %d/%d images" % (i, self.samples_num))

        self.__save_gen_images(gen_images, out_path, transformer)

    @staticmethod
    def __save_gen_images(gen_images_list, out_path, transformer):
        imgs = [transformer(img_ops.unbound_image_values(gen_img)) for gen_img in gen_images_list]
        imgs = np.array(imgs)
        grid_size = int(np.sqrt(imgs.shape[0]))
        big_img = img_ops.merge_images_to_grid(imgs, (grid_size, grid_size))

        cv2.imwrite(out_path, big_img)
