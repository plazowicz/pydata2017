import os.path as op

import tensorlayer as tl
import tensorflow as tf
import numpy as np

import config
from utils.logger import log
from utils.train_utils import adam_learning_options
from models.celeb_gan import generator, discriminator


@log
class GanTrainer(object):
    def __init__(self, latent_dim, transformer, out_weights_dir, train_options=adam_learning_options()):
        self.out_weights_dir = out_weights_dir
        self.transformer = transformer
        self.latent_dim = latent_dim
        self.batch_size = 64

        self.logger.info("Creating output weights directory %s" % self.out_weights_dir)
        tl.files.exists_or_mkdir(out_weights_dir)

        self.img_size = transformer.img_size()
        self.train_options = train_options
        self.weights_dump_interval = config.WEIGHTS_DUMP_INTERVAL

        self.input_images = tf.placeholder(tf.float32, shape=[self.batch_size, self.img_size, self.img_size, 3],
                                           name='input_images')
        self.z = tf.placeholder(tf.float32, shape=[None, self.latent_dim], name='z')

    def get_loss(self):
        g_out_layer, g_out = generator(self.z, True, self.img_size, self.batch_size)
        d_out_layer, d_logits = discriminator(self.input_images, True)
        d_fake_out_layer, d_fake_logits = discriminator(g_out, True, reuse=True)

        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits,
                                                                             labels=tf.ones_like(d_logits)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits,
                                                                             labels=tf.zeros_like(d_fake_logits)))
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits,
                                                                        labels=tf.ones_like(d_fake_logits)))
        d_loss = d_loss_real + d_loss_fake

        return {'d_loss': d_loss, 'g_loss': g_loss, 'd_loss_real': d_loss_real,
                'd_loss_fake': d_loss_fake}, d_out_layer, g_out_layer

    def extract_params(self, d_out_layer, g_out_layer):
        d_vars = tl.layers.get_variables_with_name("discriminator", True, True)
        g_vars = tl.layers.get_variables_with_name("generator", True, True)

        self.logger.info("Discriminator:")
        d_out_layer.print_params(False)
        self.logger.info("Generator:")
        g_out_layer.print_params(False)

        return d_vars, g_vars

    def train(self, images_file_list, epochs_num):
        lr = self.train_options['lr']

        self.logger.info("Composing GAN computation graph ...")
        with tf.device("/gpu:%d" % config.GPU_ID):
            losses_exprs, d_out_layer, g_out_layer = self.get_loss()

            d_vars, g_vars = self.extract_params(d_out_layer, g_out_layer)

            d_optimizer = tf.train.AdamOptimizer(lr, beta1=self.train_options['beta1']).minimize(
                losses_exprs['d_loss'], var_list=d_vars)
            g_optimizer = tf.train.AdamOptimizer(lr, beta1=self.train_options['beta1']).minimize(
                losses_exprs['g_loss'], var_list=g_vars)

        self.logger.info("GAN graph is ready")
        sess = tf.InteractiveSession()
        batches_num = len(images_file_list) // self.batch_size

        iter_counter = 0

        for epoch in xrange(epochs_num):
            try:
                self.logger.info("Started epoch %d/%d" % (epoch, epochs_num))
                batch_counter, mini_batch_it = 0, tl.iterate.minibatches(images_file_list, targets=images_file_list,
                                                                         batch_size=self.batch_size)
                while True:
                    batch_img_files, _ = mini_batch_it.next()
                    batch_z = np.random.uniform(-1, 1, [self.batch_size, self.latent_dim])
                    d_loss_val = self.__run_discr_minibatch(sess, d_optimizer, batch_img_files, losses_exprs, batch_z)
                    _ = self.__run_gen_minibatch(sess, g_optimizer, losses_exprs, batch_z)
                    g_loss_val = self.__run_gen_minibatch(sess, g_optimizer, losses_exprs, batch_z)

                    self.logger.info("Epoch: %d/%d, batch: %d/%d, Discr loss: %.8f, Gen loss: %.8f" %
                                     (epoch, epochs_num, batch_counter, batches_num, d_loss_val, g_loss_val))

                    if iter_counter % self.weights_dump_interval == 0 and iter_counter > 0:
                        self.logger.info("Iteration %d, dumping parameters ..." % iter_counter)
                        self.__dump_weights(iter_counter, d_out_layer, g_out_layer, sess)

                    batch_counter += 1
                    iter_counter += 1
            except StopIteration:
                self.logger.info("Finished epoch %d" % epoch)

    def __run_discr_minibatch(self, sess, d_optimizer, batch_img_files, losses_exprs, batch_z):
        batch_imgs = [self.transformer.get_img(img_file) for img_file in batch_img_files]
        batch_imgs = np.array(batch_imgs)
        d_loss = losses_exprs['d_loss']

        _, d_loss_val = sess.run([d_optimizer, d_loss], feed_dict={self.input_images: batch_imgs,
                                                                   self.z: batch_z})
        return d_loss_val

    def __run_gen_minibatch(self, sess, g_optimizer, batch_z, losses_exprs):
        g_loss = losses_exprs['g_loss']
        _, g_loss_val = sess.run([g_optimizer, g_loss], feed_dict={self.z: batch_z})

        return g_loss_val

    def __dump_weights(self, iter_counter, d_out, g_out, sess):
        discr_weights_out_file = op.join(self.out_weights_dir, "gan_discr_%d.npz" % iter_counter)
        gen_weights_out_file = op.join(self.out_weights_dir, "gan_gen_%d.npz" % iter_counter)

        discr_params, gen_params = d_out.all_params, g_out.all_params
        tl.files.save_npz(discr_params, name=discr_weights_out_file, sess=sess)
        tl.files.save_npz(gen_params, name=gen_weights_out_file, sess=sess)
