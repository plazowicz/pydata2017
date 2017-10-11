import tensorflow as tf
import tensorlayer as tl
import config

from utils.train_utils import adam_learning_options
from utils.logger import log


@log
class SupCnnTrainer(object):
    def __init__(self, dataset, batch_size, create_model_func, train_options=adam_learning_options()):
        self.dataset = dataset
        self.batch_size = batch_size
        self.train_options = train_options

        self.img_size = dataset.img_size()
        self.classes_num = dataset.classes_num()

        self.input_images = tf.placeholder(tf.float32, shape=[batch_size, self.img_size, self.img_size, 3],
                                           name="input_images")
        self.labels = tf.placeholder(tf.float32, shape=[batch_size, self.classes_num])
        self.create_model_func = create_model_func

        self.logger.info("Initialization finished, dataset size = %d" % dataset.size())

    def train(self, epochs_num, testing_interval):
        init_lr = self.train_options['lr']
        global_step = tf.Variable(0, trainable=False)
        ds_size = self.dataset.size()
        batches_num = int(ds_size / float(self.batch_size))
        decay_steps = (5 * int(ds_size / float(self.batch_size)))
        self.logger.info("Decay steps = %d" % decay_steps)

        lr = tf.train.exponential_decay(init_lr, global_step=global_step, decay_steps=decay_steps,
                                        decay_rate=0.5, staircase=True)

        self.logger.info("Composing CNN graph ...")
        with tf.device("/gpu:%d" % config.GPU_ID):
            model_out_layer, model_logits = self.create_model_func(self.input_images, True, False, self.classes_num)
            model_out_layer.print_params(False)
            test_model_out_layer, _ = self.create_model_func(self.input_images, False, True, self.classes_num)

            softmax_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels,
                                                                                  logits=model_logits))
            cnn_vars = tl.layers.get_variables_with_name("discriminator", True, True)
            optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=self.train_options['beta1']).minimize(
                softmax_loss, var_list=cnn_vars, global_step=global_step)

        sess = tf.InteractiveSession()
        tl.layers.initialize_global_variables(sess)

        for epoch in xrange(epochs_num):
            self.logger.info("Epoch %d/%d" % (epoch, epochs_num))

            for batch_counter, (train_examples, train_labels) in enumerate(self.dataset.generate_train_mb()):
                _, softmax_loss_val = sess.run([optimizer, softmax_loss], feed_dict={self.input_images: train_examples,
                                                                                     self.labels: train_labels})

                current_step = global_step.eval(sess)
                self.logger.info("Epoch: %d/%d, batch: %d/%d, Class. loss: %.8f, global step: %d, lr: %.8f" % (epoch,
                    epochs_num, batch_counter, batches_num, softmax_loss_val, current_step, sess.run(lr)))

                if current_step % testing_interval == 0:
                    self.__test_cnn(sess, test_model_out_layer)

    def __test_cnn(self, sess, out_layer):
        acc = 0.
        acc_func = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out_layer.outputs, 1),
                                                   tf.argmax(self.labels, 1)), tf.float32))

        iter_num = 0
        for test_iter, (test_batch, test_labels) in enumerate(self.dataset.generate_test_mb()):
            batch_acc = sess.run(acc_func, feed_dict={self.labels: test_labels, self.input_images: test_batch})
            self.logger.info("Test iteration: %d, accuracy = %.8f" % (test_iter, batch_acc))
            acc += batch_acc
            iter_num += 1

        self.logger.info("Accuracy for %d iterations: %.8f" % (iter_num, acc / iter_num))
        return acc / iter_num
