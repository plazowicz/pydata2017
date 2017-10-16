import os.path as op
import tensorlayer as tl


# Assumption that first layer is fully connected
def read_latent_dim_from_gen_weights(gen_weights_path):
    gen_params = tl.files.load_npz(path=gen_weights_path, name='')
    return gen_params[0].shape[0]


def get_gan_generator_weights_path(weights_dir, iter_num):
    return op.join(weights_dir, "gan_gen_%d.npz" % iter_num)


def get_gan_discriminator_weights_path(weights_dir, iter_num):
    return op.join(weights_dir, "gan_discr_%d.npz" % iter_num)

