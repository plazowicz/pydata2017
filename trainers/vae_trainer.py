import os.path as op

import tensorflow as tf
import tensorlayer as tl
import numpy as np

from models.celeb import encoder, generator
from utils.train_utils import adam_learning_options
from utils.logger import log
import config


@log
class VaeTrainer(object):
    def __init__(self, latent_dim, transformer, out_weights_dir,
                 train_options=adam_learning_options()):
        self.out_weights_dir = out_weights_dir
        self.transformer = transformer
        self.train_options = train_options
        self.latent_dim = latent_dim
        self.batch_size = 64

        self.logger.info("Creating output weights directory %s" % self.out_weights_dir)
        tl.files.exists_or_mkdir(out_weights_dir)
        self.img_size = self.transformer.img_size()
        self.input_images = tf.placeholder(tf.float32, shape=[self.batch_size, self.img_size, self.img_size, 3],
                                           name='input_images')

        self.weights_dump_interval = config.WEIGHTS_DUMP_INTERVAL

    def get_loss(self):
        eps = tf.random_normal(shape=(self.batch_size, self.latent_dim), mean=0.0, stddev=1.0)

        mean_out, cov_out, z_mean, z_cov_log_sq = encoder(self.input_images, train_mode=True, z_dim=self.latent_dim)
        z = z_mean + tf.multiply(tf.sqrt(tf.exp(z_cov_log_sq)), eps)

        gen_out, _ = generator(z, train_mode=True, image_size=self.img_size)

        sse_loss = tf.reduce_mean(tf.square(gen_out.outputs - self.input_images))
        kl_loss = tf.reduce_mean(- 0.5 * tf.reduce_sum(1 + z_cov_log_sq - tf.square(z_mean)
                                                       - tf.exp(z_cov_log_sq), 1))

        vae_loss = sse_loss + kl_loss
        return {'vae': vae_loss, 'sse': sse_loss, 'kl': kl_loss}, mean_out, cov_out, gen_out

    def extract_params(self, mean_out, cov_out, gen_out):
        e_vars = tl.layers.get_variables_with_name('encoder', True, True)
        g_vars = tl.layers.get_variables_with_name('decoder', True, True)
        vae_vars = e_vars + g_vars

        self.logger.info("Encoder:")
        mean_out.print_params(False)
        cov_out.print_params(False)
        self.logger.info("Generator:")
        gen_out.print_params(False)

        return vae_vars

    def train(self, images_file_list, epochs_num):
        lr = self.train_options['lr']
        self.logger.info("Composing VAE loss computation graph ...")
        losses_exprs, mean_out, cov_out, gen_out = self.get_loss()
        self.logger.info("VAE loss graph is ready")

        vae_vars = self.extract_params(mean_out, cov_out, gen_out)

        optimizer = tf.train.AdamOptimizer(lr, beta1=self.train_options['beta1']).minimize(losses_exprs['vae'],
                                                                                           var_list=vae_vars)
        sess = tf.InteractiveSession()
        batches_num = len(images_file_list) // self.batch_size

        iter_counter = 0

        for epoch in xrange(epochs_num):
            try:
                self.logger.info("Started %d/%d epoch" % (epoch, epochs_num))
                batch_counter, mini_batch_it = 0, tl.iterate.minibatches(images_file_list, targets=images_file_list,
                                                                         batch_size=self.batch_size)
                while True:
                    batch_img_files, _ = mini_batch_it.next()
                    losses_vals = self.__run_minibatch(sess, optimizer, batch_img_files, losses_exprs)

                    self.logger.info("Epoch: %d/%d, batch: %d/%d, VAE loss: %.8f, SSE loss: %.8f, KL loss: .%8f" %
                                     (epoch, epochs_num, batch_counter, batches_num, losses_vals['vae'],
                                      losses_vals['sse'], losses_vals['kl']))

                    if iter_counter % self.weights_dump_interval == 0 and iter_counter > 0:
                        self.logger.info("Iteration %d, dumping parameters ..." % iter_counter)
                        self.__dump_weights(iter_counter, mean_out, cov_out, gen_out, sess)

                    batch_counter += 1
                    iter_counter += 1
            except StopIteration:
                self.logger.info("Finished epoch %d" % epoch)

            except Exception as e:
                raise e

    def __run_minibatch(self, sess, optimizer, batch_img_files, losses_exprs):
        batch_img = [self.transformer.get_img(batch_img_files)]
        batch_img = np.array(batch_img, dtype=np.float32)

        vae_loss, sse_loss, kl_loss = losses_exprs['vae'], losses_exprs['sse'], losses_exprs['kl']
        vae_val, sse_val, kl_val = sess.run([vae_loss, sse_loss, kl_loss, optimizer],
                                            feed_dict={self.input_images: batch_img})

        return {'vae': vae_val, 'sse': sse_val, 'kl': kl_val}

    def __dump_weights(self, iter_counter, enc_mean_out, enc_cov_out, gen_out, sess):
        enc_weights_out_file = op.join(self.out_weights_dir, "vae_enc_%d.npz" % iter_counter)
        gen_weights_out_file = op.join(self.out_weights_dir, "vae_gen_%d.npz" % iter_counter)

        enc_params = enc_mean_out.all_params + enc_cov_out.all_params
        enc_params = tl.layers.list_remove_repeat(enc_params)

        tl.files.save_npz(enc_params, name=enc_weights_out_file, sess=sess)
        tl.files.save_npz(gen_out.all_params, name=gen_weights_out_file, sess=sess)