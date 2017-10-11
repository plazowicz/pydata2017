import tensorflow as tf
import tensorlayer as tl
import numpy as np

from trainers.gan_trainer import GanTrainer
from models.ss.cifar10_gan import generator, discriminator
import config
from utils.train_utils import adam_learning_options


class SSGanTrainer(GanTrainer):
    def __init__(self, latent_dim, dataset, classes_num, out_weights_dir, batch_size=64,
                 train_options=adam_learning_options()):
        super(SSGanTrainer, self).__init__(latent_dim, dataset, out_weights_dir, batch_size, train_options)

        self.dataset = dataset
        self.classes_num = classes_num
        self.testing_interval = config.TESTING_INTERVAL
        self.testing_iterations = config.TESTING_ITERATIONS

        self.labels = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.classes_num])
        self.unlabeled_images = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.img_size,
                                                                        self.img_size, 3], name='unlabeled_images')

    def get_ss_loss(self):

        g_out_layer, g_out = generator(self.z, True, self.batch_size)
        d_lab_out_layer, d_lab_logits, d_lab_previous = discriminator(self.input_images, True, reuse=False,
                                                                      classes_num=self.classes_num,
                                                                      return_previous=True)
        _, d_unlab_logits, d_unlab_previous = discriminator(self.unlabeled_images, True, reuse=True,
                                                            classes_num=self.classes_num, return_previous=True)
        _, d_fake_logits, d_fake_previous = discriminator(g_out, True, reuse=True, classes_num=self.classes_num,
                                                          return_previous=True)

        unlab_log_logits = tf.reduce_logsumexp(d_unlab_logits, axis=1)
        fake_log_logits = tf.reduce_logsumexp(d_fake_logits, axis=1)

        lab_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=d_lab_logits, labels=self.labels))
        unlab_loss = 0.5 * tf.reduce_mean(-unlab_log_logits + tf.nn.softplus(unlab_log_logits)) + 0.5 * tf.reduce_mean(
            tf.nn.softplus(fake_log_logits))

        real_features = tf.reduce_mean(d_unlab_previous[0].outputs, axis=0)
        fake_features = tf.reduce_mean(d_fake_previous[0].outputs, axis=0)
        g_loss = tf.reduce_mean(tf.abs(real_features - fake_features))

        d_loss = lab_loss + unlab_loss

        return {'d_loss': d_loss, 'g_loss': g_loss}, d_lab_out_layer, g_out_layer

    def train_ss(self, epochs_num):
        iter_counter, beta1 = 0, self.train_options['beta1']
        global_step = tf.Variable(0, trainable=False)
        decay_steps = (2*(epochs_num // 5) * int(self.dataset.size() / float(self.batch_size)))

        lr = tf.train.exponential_decay(learning_rate=self.train_options['lr'], decay_steps=decay_steps, decay_rate=0.5,
                                        staircase=True, global_step=global_step)

        self.logger.info("Composing GAN computation graph ...")
        with tf.device("/gpu:%d" % config.GPU_ID):
            losses_exprs, d_out_layer, g_out_layer = self.get_ss_loss()

            test_d_out_layer, _ = discriminator(self.input_images, False, reuse=True, classes_num=self.classes_num)
            d_vars, g_vars = self.extract_params(d_out_layer, g_out_layer)
            d_param_avg = [tf.Variable(0. * p) for p in d_vars]
            d_optimizer = tf.contrib.layers.optimize_loss(loss=losses_exprs['d_loss'], global_step=global_step,
                                                          learning_rate=lr,
                                                          optimizer=tf.train.AdamOptimizer(beta1=beta1),
                                                          variables=d_vars)
            g_optimizer = tf.contrib.layers.optimize_loss(loss=losses_exprs['g_loss'], global_step=global_step,
                                                          learning_rate=lr,
                                                          optimizer=tf.train.AdamOptimizer(beta1=beta1),
                                                          variables=g_vars)

        self.logger.info("GAN graph is ready")
        sess = tf.InteractiveSession()

        tl.layers.initialize_global_variables(sess)
        batches_num = self.dataset.size() // self.batch_size

        for epoch in xrange(epochs_num):
            self.logger.info("Started epoch %d/%d" % (epoch, epochs_num))

            for batch_counter, (train_lab_ex, train_unlab_ex, train_labels) in enumerate(
                    self.dataset.generate_train_mb()):
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.latent_dim]).astype(np.float32)

                d_loss_val = self.run_ss_discr_minibatch(sess, d_optimizer, train_lab_ex, train_unlab_ex, train_labels,
                                                         losses_exprs, batch_z)
                d_param_avg = [tf.assign_add(a, 0.0001*(p-a)) for a, p in zip(d_param_avg, d_vars)]

                self.run_ss_gen_minibatch(sess, g_optimizer, train_unlab_ex, train_labels, losses_exprs, batch_z)
                self.run_ss_gen_minibatch(sess, g_optimizer, train_unlab_ex, train_labels, losses_exprs, batch_z)
                g_loss_val = self.run_ss_gen_minibatch(sess, g_optimizer, train_unlab_ex, train_labels,
                                                       losses_exprs, batch_z)

                self.logger.info("Epoch: %d/%d, batch: %d/%d, labeled examples: %d, unlabeled_examples: %d, "
                                 "Discr loss: %.8f, Gen loss: %.8f, global_step: %d, learning rate: %.8f" %
                                 (epoch, epochs_num, batch_counter, batches_num, train_lab_ex.shape[0],
                                  train_unlab_ex.shape[0], d_loss_val, g_loss_val,
                                  sess.run(global_step), sess.run(lr)))

                if iter_counter % self.weights_dump_interval == 0 and iter_counter > 0:
                    self.logger.info("Iteration %d, dumping parameters ..." % iter_counter)
                    self.dump_weights(iter_counter, d_out_layer, g_out_layer, sess)

                if iter_counter % self.testing_interval == 0:
                    self.logger.info("Iteration %d, testing net ..." % iter_counter)
                    self.__test_discriminator(self.testing_iterations, sess, test_d_out_layer)

                iter_counter += 1

            self.logger.info("Finished epoch %d" % epoch)

    def run_ss_discr_minibatch(self, sess, d_optimizer, lab_train_examples, unlab_train_examples,
                               train_labels, losses_exprs, batch_z):

        d_loss = losses_exprs['d_loss']
        _, d_loss_val = sess.run([d_optimizer, d_loss], feed_dict={self.input_images: lab_train_examples,
                                                                   self.unlabeled_images: unlab_train_examples,
                                                                   self.labels: train_labels,
                                                                   self.z: batch_z})

        return d_loss_val

    def run_ss_gen_minibatch(self, sess, g_optimizer, unlab_train_examples, train_labels, losses_exprs, batch_z):
        g_loss = losses_exprs['g_loss']
        _, g_loss_val = sess.run([g_optimizer, g_loss], feed_dict={self.unlabeled_images: unlab_train_examples,
                                                                   self.labels: train_labels,
                                                                   self.z: batch_z})
        return g_loss_val

    def __test_discriminator(self, iter_num, sess, d_out_layer):
        acc = 0.
        acc_func = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(d_out_layer.outputs, 1),
                                                   tf.argmax(self.labels, 1)), tf.float32))

        for test_iter, (test_batch, test_labels) in enumerate(self.dataset.generate_test_mb()):
            if test_iter == iter_num:
                break
            batch_acc = sess.run(acc_func, feed_dict={self.labels: test_labels, self.input_images: test_batch})
            self.logger.info("Test iteration: %d, accuracy = %.8f" % (test_iter, batch_acc))
            acc += batch_acc

        self.logger.info("Accuracy for %d iterations: %.8f" % (iter_num, acc / iter_num))
        return acc / iter_num
