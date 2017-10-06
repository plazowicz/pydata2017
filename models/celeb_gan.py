import tensorflow as tf
import tensorlayer as tl

from tensorlayer.layers import InputLayer, Conv2d, BatchNormLayer, FlattenLayer, DenseLayer, \
    ReshapeLayer, DeConv2d

from utils.logger import log


def generator(input_placeholder, train_mode, image_size, batch_size, reuse=False):
    filters_num = 128

    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    s2, s4, s8, s16 = int(image_size / 2), int(image_size / 4), int(image_size / 8), int(image_size / 16)

    with tf.variable_scope("generator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        input_layer = InputLayer(input_placeholder, name='gen/in')
        lin_layer = DenseLayer(input_layer, n_units=filters_num * 8 * s16 * s16, W_init=w_init,
                               act=tf.identity, name='gen/lin')

        resh1_layer = ReshapeLayer(lin_layer, shape=[-1, s16, s16, filters_num * 8], name='gen/reshape')

        in_bn_layer = BatchNormLayer(resh1_layer, act=tf.nn.relu, is_train=train_mode,
                                     gamma_init=gamma_init, name='dec/in_bn')
        # in_bn_layer.shape = (batch_size, 4, 4, 1024)
        up1_layer = DeConv2d(in_bn_layer, filters_num * 4, (5, 5), out_size=(s8, s8), strides=(2, 2),
                             padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='gen/up1')

        bn1_layer = BatchNormLayer(up1_layer, act=tf.nn.relu, is_train=train_mode,
                                   gamma_init=gamma_init, name='dec/bn1')

        # bn1_layer.shape = (batch_size, 8, 8, 512)
        up2_layer = DeConv2d(bn1_layer, filters_num * 2, (5, 5), out_size=(s4, s4), strides=(2, 2),
                             padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='gen/up2')
        bn2_layer = BatchNormLayer(up2_layer, act=tf.nn.relu, is_train=train_mode,
                                   gamma_init=gamma_init, name='dec/bn2')
        # bn2_layer.shape = (batch_size, 16, 16, 256)

        up3_layer = DeConv2d(bn2_layer, filters_num, (5, 5), out_size=(s2, s2), strides=(2, 2),
                             padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='gen/up3')
        bn3_layer = BatchNormLayer(up3_layer, act=tf.nn.relu, is_train=train_mode,
                                   gamma_init=gamma_init, name='dec/bn3')
        # bn3_layer.shape = (batch_size, 32, 32, 128)
        up4_layer = DeConv2d(bn3_layer, 3, (5, 5), out_size=(image_size, image_size), strides=(2, 2),
                             padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='gen/up4')

        up4_layer.outputs = tf.nn.tanh(up4_layer.outputs)

    return up4_layer, up4_layer.outputs


def discriminator(input_placeholder, train_mode, reuse=False):
    filters_num = 64
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("discriminator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        input_layer = InputLayer(input_placeholder, name="discr/in")
        conv1_layer = Conv2d(input_layer, n_filter=filters_num, filter_size=(5, 5), strides=(2, 2),
                             act=lambda x: tl.act.lrelu(x, 0.2), padding='SAME', name='discr/conv1', W_init=w_init)

        conv2_layer = Conv2d(conv1_layer, n_filter=filters_num*2, filter_size=(5, 5), strides=(2, 2),
                             act=None, padding='SAME', name='discr/conv2', W_init=w_init)
        bn2_layer = BatchNormLayer(conv2_layer, act=lambda x: tl.act.lrelu(x, 0.2), is_train=train_mode,
                                   gamma_init=gamma_init, name='discr/bn2')

        conv3_layer = Conv2d(bn2_layer, n_filter=filters_num*4, filter_size=(5, 5), strides=(2, 2),
                             act=None, padding='SAME', name='discr/conv3', W_init=w_init)
        bn3_layer = BatchNormLayer(conv3_layer, act=lambda x: tl.act.lrelu(x, 0.2), is_train=train_mode,
                                   gamma_init=gamma_init, name='discr/bn3')

        conv4_layer = Conv2d(bn3_layer, n_filter=filters_num*8, filter_size=(5, 5), strides=(2, 2),
                             act=None, padding='SAME', name='discr/conv4', W_init=w_init)
        bn4_layer = BatchNormLayer(conv4_layer, act=lambda x: tl.act.lrelu(x, 0.2), is_train=train_mode,
                                   gamma_init=gamma_init, name='discr/bn4')

        flat_layer = FlattenLayer(bn4_layer, name='discr/flatten')

        out_layer = DenseLayer(flat_layer, n_units=1, W_init=w_init, act=tf.identity, name='discr/out')
        logits = out_layer.outputs

        out_layer.outputs = tf.nn.sigmoid(out_layer.outputs)

    return out_layer, logits


@log
def load_gen_with_weights(sess, gen_input, batch_size, gen_path, reuse=False):
    gen_params = tl.files.load_npz(path=gen_path, name='')
    img_size, latent_dim = read_settings_with_weights(gen_path)

    load_gen_with_weights.logger.info("Loading weights for generator, img size = %d, latent dim = %d" %
                                      (img_size, latent_dim))
    gen_out_layer, gen_out = generator(gen_input, False, img_size, batch_size, reuse)

    tl.files.assign_params(sess, gen_params, gen_out)
    return gen_out_layer, gen_out


def read_settings_with_weights(gen_path):
    import math

    gen_params = tl.files.load_npz(path=gen_path, name='')
    filters_num = gen_params[-3].shape[0]

    dense_layer_out, latent_dim = gen_params[0].shape[1], gen_params[0].shape[0]
    img_size = int(math.sqrt(dense_layer_out / (filters_num * 8))) * 16

    return img_size, latent_dim
