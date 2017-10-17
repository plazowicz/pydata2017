import os.path as op
import tensorflow as tf

from utils import device_utils
from utils import net_utils
from utils import img_ops
from utils.logger import log


@log
class VaeImgGenerator(object):

    def __init__(self, gen_params_path, load_gen_with_weights_func,
                 get_latent_dim_func=net_utils.read_latent_dim_from_gen_weights):
        # TODO - get latent dimensionality
        pass

    def generate_images(self, samples_num, out_path, transformer=lambda x: img_ops.unbound_image_values(x)):
        # TODO - instantiate tf Session
        # TODO - sample z randomly
        z = None
        gen_images = []

        # TODO - run loop and generate image
        img_ops.save_gen_images(gen_images, out_path, transformer)

