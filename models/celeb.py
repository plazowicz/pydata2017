import tensorflow as tf
import tensorlayer as tl
import numpy as np

from tensorlayer.layers import InputLayer, Conv2d, BatchNormLayer, FlattenLayer, DenseLayer, \
    ReshapeLayer, UpSampling2dLayer
from tensorflow.python.ops.image_ops_impl import ResizeMethod

from utils.logger import log


def encoder(input_placeholder, z_dim, train_mode, conv_filters_num=32, reuse=False):

    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("encoder", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        input_layer = InputLayer(input_placeholder, name='enc/input')
        conv1_layer = Conv2d(input_layer, n_filter=conv_filters_num, filter_size=(4, 4), strides=(2, 2),
                             act=None, padding='SAME', name='enc/conv1')

        bn1_layer = BatchNormLayer(conv1_layer, act=lambda x: tl.act.lrelu(x, 0.02), is_train=train_mode,
                                   gamma_init=gamma_init, name='enc/bn1')

        conv2_layer = Conv2d(bn1_layer, n_filter=2 * conv_filters_num, filter_size=(4, 4), strides=(2, 2),
                             act=None, padding='SAME', name='enc/conv2')
        bn2_layer = BatchNormLayer(conv2_layer, act=lambda x: tl.act.lrelu(x, 0.02), is_train=train_mode,
                                   gamma_init=gamma_init, name='enc/bn2')

        conv3_layer = Conv2d(bn2_layer, n_filter=4 * conv_filters_num, filter_size=(4, 4), strides=(2, 2),
                             act=None, padding='SAME', name='enc/conv3')
        bn3_layer = BatchNormLayer(conv3_layer, act=lambda x: tl.act.lrelu(x, 0.02), is_train=train_mode,
                                   gamma_init=gamma_init, name='enc/bn3')

        # mean of Z
        mean_flat_layer = FlattenLayer(bn3_layer, name='enc/mean_flatten')
        mean_out = DenseLayer(mean_flat_layer, n_units=z_dim, act=tf.identity, W_init=w_init, name='enc/mean_out_lin')
        mean_out = BatchNormLayer(mean_out, act=tf.identity, is_train=train_mode, gamma_init=gamma_init,
                                  name='enc/mean_out')

        # covariance of Z
        cov_flat_layer = FlattenLayer(bn3_layer, name='enc/cov_flatten')
        cov_out = DenseLayer(cov_flat_layer, n_units=z_dim, act=tf.identity, W_init=w_init, name='enc/cov_out_lin')
        cov_out = BatchNormLayer(cov_out, act=tf.identity, is_train=train_mode, gamma_init=gamma_init,
                                 name='enc/cov_out')
        z_mean, z_cov = mean_out.outputs, cov_out.outputs + 1e-6

    return mean_out, cov_out, z_mean, z_cov


def generator(input_placeholder, train_mode, image_size, reuse=False):
    s2, s4, s8, s16 = int(image_size / 2), int(image_size / 4), int(image_size / 8), int(image_size / 16)
    gf_dim = 32

    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("decoder", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        input_layer = InputLayer(input_placeholder, name='dec/input')
        lin_layer = DenseLayer(input_layer, n_units=gf_dim * 8 * s16 * s16, W_init=w_init,
                               act=tf.identity, name='dec/lin')
        # lin_layer.shape = (batch_size,256*4*4)
        resh1_layer = ReshapeLayer(lin_layer, shape=[-1, s16, s16, gf_dim * 8], name='decoder/reshape')
        # resh1_layer.shape = (batch_size, 4, 4, 256)
        in_bn_layer = BatchNormLayer(resh1_layer, act=lambda x: tl.act.lrelu(x, 0.2), is_train=train_mode,
                                     gamma_init=gamma_init, name='dec/in_bn')

        # upsampling
        up1_layer = UpSampling2dLayer(in_bn_layer, size=[s8, s8], is_scale=False, method=ResizeMethod.NEAREST_NEIGHBOR,
                                      align_corners=False, name='dec/up1')
        conv1_layer = Conv2d(up1_layer, gf_dim * 4, (3, 3), (1, 1), padding='SAME', W_init=w_init, name='dec/conv1')
        bn1_layer = BatchNormLayer(conv1_layer, act=lambda x: tl.act.lrelu(x, 0.2), is_train=train_mode,
                                   gamma_init=gamma_init, name='dec/bn1')
        # bn1_layer.shape = (batch_size,8,8,128)

        up2_layer = UpSampling2dLayer(bn1_layer, size=[s4, s4], is_scale=False, method=ResizeMethod.NEAREST_NEIGHBOR,
                                      align_corners=False, name='dec/up2')
        conv2_layer = Conv2d(up2_layer, gf_dim * 2, (3, 3), (1, 1), padding='SAME', W_init=w_init, name='dec/conv2')
        bn2_layer = BatchNormLayer(conv2_layer, act=lambda x: tl.act.lrelu(x, 0.2), is_train=train_mode,
                                   gamma_init=gamma_init, name='dec/bn2')
        # bn2_layer.shape = (batch_size,16,16,64)

        up3_layer = UpSampling2dLayer(bn2_layer, size=[s2, s2], is_scale=False, method=ResizeMethod.NEAREST_NEIGHBOR,
                                      align_corners=False, name='dec/up3')
        conv3_layer = Conv2d(up3_layer, gf_dim, (3, 3), (1, 1), padding='SAME', W_init=w_init, name='dec/conv3')
        bn3_layer = BatchNormLayer(conv3_layer, act=lambda x: tl.act.lrelu(x, 0.2), is_train=train_mode,
                                   gamma_init=gamma_init, name='dec/bn3_layer')
        # bn3_layer.shape = (batch_size,32,32,32)

        # no BN on last deconv
        up4_layer = UpSampling2dLayer(bn3_layer, size=[image_size, image_size], is_scale=False,
                                      method=ResizeMethod.NEAREST_NEIGHBOR,
                                      align_corners=False, name='dec/up4')
        conv4_layer = Conv2d(up4_layer, 3, (3, 3), (1, 1), padding='SAME', W_init=w_init, name='dec/conv4')
        # conv4_layer.shape = (batch_size,64,64,3)
        logits = conv4_layer.outputs
        conv4_layer.outputs = tf.nn.tanh(conv4_layer.outputs)
    return conv4_layer, logits


@log
def load_model_with_weights(sess, enc_input, gen_input, enc_path, gen_path):

    enc_params = tl.files.load_npz(path=enc_path, name='')
    gen_params = tl.files.load_npz(path=gen_path, name='')

    z_dim, img_size, filters_num = read_settings_from_weights(enc_path, gen_path)

    mean_out, cov_out, _, _ = encoder(enc_input, z_dim, False, filters_num)
    gen_out, _ = generator(gen_input, False, img_size)

    load_model_with_weights.logger.info("Loading weight for encoder and generator, latent dim = %d, "
                                        "image size = %d, conv filters num = %d" % (z_dim, img_size, filters_num))

    tl.files.assign_params(sess, enc_params[:24], mean_out)
    tl.files.assign_params(sess, np.concatenate((enc_params[:24], enc_params[30:]), axis=0), cov_out)

    tl.files.assign_params(sess, gen_params, gen_out)


@log
def load_enc_with_weights(sess, enc_input, enc_path):
    enc_params = tl.files.load_npz(path=enc_path, name='')
    z_dim, filters_num = read_enc_settings_with_weights(enc_path)

    mean_out, cov_out, z_mean, z_cov = encoder(enc_input, z_dim, False, filters_num, True)
    load_enc_with_weights.logger.info("Loading weights for encoder, latent dim = %d, conv filters num = %d"
                                      % (z_dim, filters_num))

    tl.files.assign_params(sess, enc_params[:24], mean_out)
    tl.files.assign_params(sess, np.concatenate((enc_params[:24], enc_params[30:]), axis=0), cov_out)

    return mean_out, cov_out, z_mean, z_cov


@log
def load_gen_with_weights(sess, gen_input, gen_path):
    gen_params = tl.files.load_npz(path=gen_path, name='')
    img_size = read_gen_settings_with_weights(gen_path)

    load_gen_with_weights.logger.info("Loading weights for generator, img size = %d" % img_size)

    gen_out, gen_logits = generator(gen_input, False, img_size, True)
    tl.files.assign_params(sess, gen_params, gen_out)

    return gen_out, gen_logits


def read_enc_settings_with_weights(enc_path):
    enc_params = tl.files.load_npz(path=enc_path, name='')
    filters_num = enc_params[0].shape[-1]
    z_dim = enc_params[-1].shape[0]

    return z_dim, filters_num


def read_gen_settings_with_weights(gen_path):
    import math

    gen_params = tl.files.load_npz(path=gen_path, name='')
    img_size = int(math.sqrt(gen_params[0].shape[1] / (32 * 8))) * 16
    return img_size


def read_settings_from_weights(enc_path, gen_path):
    import math

    enc_params = tl.files.load_npz(path=enc_path, name='')
    gen_params = tl.files.load_npz(path=gen_path, name='')

    first_conv_filter = enc_params[0]
    filters_num = first_conv_filter.shape[-1]
    z_dim = gen_params[0].shape[0]
    img_size = int(math.sqrt(gen_params[0].shape[1] / (32 * 8))) * 16

    return z_dim, img_size, filters_num
