import tensorflow as tf

from utils import device_utils
from utils import net_utils
from utils import img_ops
from utils.logger import log


@log
class VaeImgGenerator(object):

    # load_gen_with_weights_func(sess, latent_vector, saved_parameters_path)
    def __init__(self, gen_params_path, load_gen_with_weights_func,
                 get_latent_dim_func=net_utils.read_latent_dim_from_gen_weights):

        self.gen_params_path = gen_params_path
        self.load_gen_with_weights_func = load_gen_with_weights_func
        self.z_dim = get_latent_dim_func(self.gen_params_path)

    def generate_images(self, samples_num, out_path, transformer=lambda x: img_ops.unbound_image_values(x)):
        # TODO - instantiate tf Session
        # TODO - sample z randomly
        sess = tf.InteractiveSession()
        with tf.device(device_utils.get_device()):
            z = tf.random_uniform(shape=[1, self.z_dim], minval=-1, maxval=1, dtype=tf.float32)
            gen_out_layer, _ = self.load_gen_with_weights_func(sess, z, self.gen_params_path)
            x_generated = gen_out_layer.outputs

        gen_images = []
        for i in xrange(samples_num):
            x_gen = sess.run(x_generated)[0]
            gen_images.append(x_gen)

            self.logger.info("Generated %d/%d image" % (i, samples_num))

        img_ops.save_gen_images(gen_images, out_path, transformer)
