import tensorflow as tf
import numpy as np
import cv2

from utils import device_utils
from utils import img_ops
from utils.logger import log
from utils import fs_utils


@log
class VaeImgReconstructor(object):

    def __init__(self, load_enc_with_weights_func, load_gen_with_weights_func, get_latent_dim_func):
        self.load_enc_with_weights_func = load_enc_with_weights_func
        self.load_gen_with_weights_func = load_gen_with_weights_func
        self.z_dim = get_latent_dim_func()
        pass

    def reconstruct_faces(self, out_path, imgs_to_reconstruct, transformer=lambda x: img_ops.unbound_image_values(x)):
        out_original_path = fs_utils.add_suffix_to_path(out_path, "original")
        self.__save_original_images(imgs_to_reconstruct, out_original_path, transformer)

        img_size = imgs_to_reconstruct.shape[1]

        sess = tf.InteractiveSession()
        with tf.device(device_utils.get_device()):
            input_images = tf.placeholder(dtype=tf.float32, shape=[1, img_size, img_size, 3])
            _, _, z_mean, z_cov_log_sq = self.load_enc_with_weights_func(sess, input_images)
            eps = tf.random_normal(shape=(1, self.z_dim), mean=0.0, stddev=1.0)

            z = z_mean + tf.multiply(tf.sqrt(tf.exp(z_cov_log_sq)), eps)
            gen_out_layer, _ = self.load_gen_with_weights_func(sess, z)
            x_reconstr = gen_out_layer.outputs

        reconstr_images = []
        for i in xrange(imgs_to_reconstruct.shape[0]):
            reconstr_img = sess.run(x_reconstr, feed_dict={input_images: [imgs_to_reconstruct[i, :, :, :]]})
            reconstr_images.append(reconstr_img[0])

            self.logger.info("Reconstructed %d/%d images" % (i, imgs_to_reconstruct.shape[0]))

        img_ops.save_gen_images(reconstr_images, out_path, transformer)

    @staticmethod
    def __save_original_images(imgs_to_reconstruct, out_path, transformer):

        imgs = np.array([transformer(i) for i in imgs_to_reconstruct])
        grid_size = int(np.sqrt(imgs.shape[0]))
        big_img = img_ops.merge_images_to_grid(imgs[:grid_size*grid_size], (grid_size, grid_size))

        cv2.imwrite(out_path, big_img)
