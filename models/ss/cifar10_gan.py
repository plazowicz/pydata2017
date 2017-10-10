import tensorflow as tf
import tensorlayer as tl

from tensorlayer.layers import InputLayer, Conv2d, BatchNormLayer, FlattenLayer, DenseLayer, \
    ReshapeLayer, DeConv2d

from utils.logger import log

from models import celeb_gan


def generator(input_placeholder, train_mode, batch_size, reuse=False):

    image_size = 32
    return celeb_gan.generator(input_placeholder, train_mode, image_size, batch_size, reuse=reuse, filters_num=64)


def discriminator(input_placeholder, train_mode, reuse=False, classes_num=10, return_previous=False):
    filters_num = 64
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    def conv(in_layer, filt_num, name, stride=1):
        return Conv2d(in_layer, n_filter=filt_num, filter_size=(3, 3), strides=(stride, stride),
                      act=None, padding='SAME', name=name, W_init=w_init)

    def bn(in_layer, name):
        return BatchNormLayer(in_layer, act=lambda x: tl.act.lrelu(x, 0.2), is_train=train_mode,
                              gamma_init=gamma_init, name=name)

    with tf.variable_scope("discriminator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        input_layer = InputLayer(input_placeholder, name="discr/in")

        conv1_layer = bn(conv(input_layer, filters_num, 'discr/conv1'), 'discr/bn1')
        conv2_layer = bn(conv(conv1_layer, filters_num, 'discr/conv2'), 'discr/bn2')
        conv3_layer = bn(conv(conv2_layer, filters_num, 'discr/conv3', 2), 'discr/bn3')
        conv4_layer = bn(conv(conv3_layer, filters_num * 3, 'discr/conv4'), 'discr/bn4')
        conv5_layer = bn(conv(conv4_layer, filters_num * 3, 'discr/conv5'), 'discr/bn5')
        conv6_layer = bn(conv(conv5_layer, filters_num * 3, 'discr/conv6', 2), 'discr/bn6')
        conv7_layer = bn(conv(conv6_layer, filters_num * 3, 'discr/conv7'), 'discr/bn7')
        flat_layer = FlattenLayer(conv7_layer, name='discr/flatten')

        out_layer = DenseLayer(flat_layer, n_units=classes_num, act=tf.identity, name='discr/out')
        logits = out_layer.outputs
        out_layer.outputs = tf.nn.softmax(logits)

    if not return_previous:
        return out_layer, logits
    else:
        return out_layer, logits, [flat_layer, conv7_layer, conv6_layer, conv5_layer, conv4_layer, conv3_layer,
                                   conv2_layer, conv1_layer]


# @log
# def load_gen_with_weights(sess, gen_input, batch_size, gen_path, reuse=False):
#     gen_params = tl.files.load_npz(path=gen_path, name='')
#     img_size, latent_dim = read_settings_with_weights(gen_path)
#
#     load_gen_with_weights.logger.info("Loading weights for generator, img size = %d, latent dim = %d" %
#                                       (img_size, latent_dim))
#     gen_out_layer, gen_out = generator(gen_input, False, img_size, batch_size, reuse)
#
#     tl.files.assign_params(sess, gen_params, gen_out)
#     return gen_out_layer, gen_out
#
#
# def read_settings_with_weights(gen_path):
#     import math
#
#     gen_params = tl.files.load_npz(path=gen_path, name='')
#     filters_num = gen_params[-3].shape[0]
#
#     dense_layer_out, latent_dim = gen_params[0].shape[1], gen_params[0].shape[0]
#     img_size = int(math.sqrt(dense_layer_out / (filters_num * 8))) * 16
#
#     return img_size, latent_dim
