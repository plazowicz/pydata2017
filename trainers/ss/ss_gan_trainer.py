import tensorflow as tf
import tensorlayer as tl
import numpy as np

from trainers.gan_trainer import GanTrainer
from models.ss.cifar10_gan import generator, discriminator
import config
from utils.train_utils import adam_learning_options


class SSGanTrainer(GanTrainer):

    def __init__(self, latent_dim, classes_num, out_weights_dir, batch_size=64, train_options=adam_learning_options()):
        super(SSGanTrainer).__init__(latent_dim, None, out_weights_dir, batch_size, train_options)

        self.classes_num = classes_num
        self.labels = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, classes_num])
        self.testing_interval = config.TESTING_INTERVAL
        self.testing_iterations = config.TESTING_ITERATIONS

    def get_ss_loss(self):
        alpha = 0.9
        real_labels = tf.concat([self.labels, tf.zeros([self.batch_size, 1])], axis=1)
        fake_labels = tf.concat([(1-alpha)*tf.ones([self.batch_size, self.classes_num])/self.classes_num,
                                 alpha*tf.ones([self.batch_size, 1])], axis=1)
        inv_fake_labels = tf.concat([alpha*tf.ones([self.batch_size, self.classes_num]),
                                     (1-alpha)*tf.ones([self.batch_size, 1])/self.classes_num], axis=1)

        g_out_layer, g_out = generator(self.z, True, self.batch_size)
        d_out_layer, d_logits = discriminator(self.input_images, True, reuse=False, classes_num=self.classes_num)
        d_fake_out_layer, d_fake_logits = discriminator(g_out, True, reuse=True, classes_num=self.classes_num)

        # labels will be set for both labeled and unlabeled examples
        d_loss_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=d_logits, labels=real_labels))
        d_loss_fake = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=d_fake_logits, labels=fake_labels))
        g_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=d_fake_logits, labels=inv_fake_labels))

        d_loss = d_loss_real + d_loss_fake

        return {'d_loss': d_loss, 'g_loss': g_loss, 'd_loss_real': d_loss_real,
                'd_loss_fake': d_loss_fake}, d_out_layer, g_out_layer

    def train_ss(self, dataset, epochs_num):
        iter_counter, beta1 = 0, self.train_options['beta1']
        global_step = tf.Variable(0, trainable=False)

        lr = tf.train.exponential_decay(learning_rate=self.train_options['lr'], decay_steps=10000, decay_rate=0.5,
                                        staircase=True, global_step=global_step)

        self.logger.info("Composing GAN computation graph ...")
        with tf.device("/gpu:%d" % config.GPU_ID):
            losses_exprs, d_out_layer, g_out_layer = self.get_ss_loss()

            test_d_out_layer = discriminator(self.input_images, False, reuse=True, classes_num=self.classes_num)
            d_vars, g_vars = self.extract_params(d_out_layer, g_out_layer)
            d_optimizer = tf.contrib.layers.optimize_loss(loss=losses_exprs['d_loss'], learning_rate=lr * 0.5,
                                                          optimizer=tf.train.AdamOptimizer(beta1=beta1),
                                                          clip_gradients=20.0, variables=d_vars)
            g_optimizer = tf.contrib.layers.optimize_loss(loss=losses_exprs['g_loss'], learning_rate=lr,
                                                          optimizer=tf.train.AdamOptimizer(beta1=beta1),
                                                          clip_gradients=20.0, variables=g_vars)

        self.logger.info("GAN graph is ready")
        sess = tf.InteractiveSession()

        tl.layers.initialize_global_variables(sess)
        batches_num = dataset.size() // self.batch_size

        for epoch in xrange(epochs_num):
            self.logger.info("Started epoch %d/%d" % (epoch, epochs_num))

            for batch_counter, (train_examples, train_labels) in enumerate(dataset.generate_mb(ds_type='train')):
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.latent_dim]).astype(np.float32)
                d_loss_val = self.run_ss_discr_minibatch(sess, d_optimizer, train_examples, train_labels,
                                                         losses_exprs, batch_z)
                _ = self.run_gen_minibatch(sess, g_optimizer, train_examples, batch_z)
                g_loss_val = self.run_gen_minibatch(sess, g_optimizer, train_examples, batch_z)

                self.logger.info("Epoch: %d/%d, batch: %d/%d, Discr loss: %.8f, Gen loss: %.8f, global_step: %d" %
                                 (epoch, epochs_num, batch_counter, batches_num, d_loss_val, g_loss_val,
                                  sess.run(global_step)))
                if iter_counter % self.weights_dump_interval == 0 and iter_counter > 0:
                    self.logger.info("Iteration %d, dumping parameters ..." % iter_counter)
                    self.__dump_weights(iter_counter, d_out_layer, g_out_layer, sess)

                if iter_counter % self.testing_interval:
                    self.logger.info("Iteration %d, testing net ..." % iter_counter)
                    self.__test_discriminator(self.testing_iterations, dataset, sess, test_d_out_layer)

                iter_counter += 1
                sess.run(tf.assign_add(global_step, 1))

            self.logger.info("Finished epoch %d" % epoch)

    def run_ss_discr_minibatch(self, sess, d_optimizer, train_examples, train_labels, losses_exprs, batch_z):
        d_loss = losses_exprs['d_loss']

        _, d_loss_val = sess.run([d_optimizer, d_loss], feed_dict={self.input_images: train_examples,
                                                                   self.labels: train_labels, self.z: batch_z})

        return d_loss_val

    def __test_discriminator(self, iter_num, dataset, sess, d_out_layer):
        acc = 0.
        predictions = tf.placeholder(tf.float32, shape=[self.batch_size, self.classes_num])
        acc_func = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predictions, 1), tf.argmax(self.labels)), tf.float32))

        for test_iter, (test_batch, test_labels) in enumerate(dataset.generate_mb(ds_type='test')):
            if test_iter == iter_num:
                break
            test_predictions = sess.run(d_out_layer.outputs, feed_dict={self.input_images: test_batch})
            batch_acc = sess.run(acc_func, feed_dict={self.labels: test_labels, predictions: test_predictions})
            self.logger.info("Test iteration: %d, accuracy = %.8f" % (test_iter, batch_acc))
            acc += batch_acc

        self.logger.info("Accuracy for %d iterations: .8f" % acc)
        return acc/iter_num




