import tensorflow as tf
import tensorlayer as tl

from tensorlayer.layers import InputLayer, Conv2d, BatchNormLayer, FlattenLayer, DenseLayer, \
    ReshapeLayer, UpSampling2dLayer
from tensorflow.python.ops.image_ops_impl import ResizeMethod


def encoder(input_placeholder, z_dim, train_mode, conv_filters_num=32, reuse=False):

    input_layer = InputLayer(input_placeholder, name='enc/input')
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("encoder", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

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

        conv4_layer = Conv2d(bn3_layer, n_filter=8 * conv_filters_num, filter_size=(4, 4), strides=(2, 2),
                             act=None, padding='SAME', name='enc/conv4')
        bn4_layer = BatchNormLayer(conv4_layer, act=lambda x: tl.act.lrelu(x, 0.02), is_train=train_mode,
                                   gamma_init=gamma_init, name='enc/bn4')

        # mean of Z
        mean_flat_layer = FlattenLayer(bn4_layer, name='enc/mean_flatten')
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


class CelebVae(object):

    def __init__(self, latent_dim, image_size, batch_size, filters_num=64):
        self.input_images = tf.placeholder(tf.float32, [batch_size, image_size, image_size, 3], 'input_images')
        self.mean_out, self.cov_out, self.z_mean, self.z_cov = encoder(self.input_images, True,
                                                                       {'latent_dim': latent_dim,
                                                                        'filters_num': filters_num})
