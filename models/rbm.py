import time

import numpy as np
import theano
import theano.tensor as T

from utils.rbm_utils import Sampler


class BinaryRBM(object):
    def __init__(self, nb_visible, nb_hidden, weights=None, hidden_bias=None, visible_bias=None, sampler=None,
                 momentum_factor=0.9, learning_rate=0.1, batch_size=128, persistent=False, k=1,
                 hyper_params_strategy=lambda lr, k, it: (lr, k)):
        self.hyper_params_strategy = hyper_params_strategy
        self.learning_rate = learning_rate
        self.momentum_factor = momentum_factor
        self.nb_hidden = nb_hidden
        self.nb_visible = nb_visible
        self.k = k
        self.batch_size = batch_size

        # if no sampler provided then we use default sampler
        self.sampler = sampler or Sampler()

        # weights & biases setup

        if weights is None:
            # According to Hinton, this is very good weights initialization procedure
            weights = np.asarray(
                np.random.normal(
                    scale=0.01,
                    size=(self.nb_visible, self.nb_hidden)),
                dtype=theano.config.floatX)

        if hidden_bias is None:
            hidden_bias = np.zeros(self.nb_hidden, dtype=theano.config.floatX)

        if visible_bias is None:
            visible_bias = np.zeros(self.nb_visible, dtype=theano.config.floatX)

        # momentum setup

        momentum_speed = np.zeros((self.nb_visible, self.nb_hidden), dtype=theano.config.floatX)
        momentum_speed_hidden_bias = np.zeros(self.nb_hidden, dtype=theano.config.floatX)
        momentum_speed_visible_bias = np.zeros(self.nb_visible, dtype=theano.config.floatX)

        # initialise theano variables
        self.ts_weights = theano.shared(weights, name='weights', borrow=True)
        self.ts_hidden_bias = theano.shared(hidden_bias, name='hidden bias', borrow=True)
        self.ts_visible_bias = theano.shared(visible_bias, name='visible bias', borrow=True)
        self.ts_momentum_speed = theano.shared(momentum_speed, name='momentum speed', borrow=True)
        self.ts_momentum_speed_hidden_bias = theano.shared(momentum_speed_hidden_bias,
                                                           name='momentum speed hidden bias', borrow=True)
        self.ts_momentum_speed_visible_bias = theano.shared(momentum_speed_visible_bias,
                                                            name='momentum speed visible bias', borrow=True)

        t_input = T.matrix('x')
        t_lr = T.scalar('lr')
        t_k = T.iscalar('k')

        if persistent:
            persistent = theano.shared(np.zeros((batch_size, nb_hidden), dtype=theano.config.floatX),
                                       name='persistent', borrow=True)
        else:
            persistent = None

        cost, updates = self.get_updates(t_input, t_lr, persistent=persistent, k=t_k)

        self.tf_train = theano.function(inputs=[t_input, t_lr, t_k], outputs=cost, updates=updates, name='train')

    def train(self, data_loader, nb_epochs=10):
        iteration = 1
        for epoch in xrange(nb_epochs):
            start = time.time()
            mean_cost = []
            for batch in data_loader.generate_batches(batch_size=self.batch_size):
                mean_cost += [self.tf_train(batch, self.learning_rate, self.k)]
                iteration += 1
                self.learning_rate, self.k = self.hyper_params_strategy(self.learning_rate, self.k, iteration)
            end = time.time()
            print "Epoch # %d: cost = %f, time = %f seconds" % (epoch + 1, np.array(mean_cost).mean(), end - start)

    @property
    def parameters(self):
        return {'weights': self.ts_weights.get_value(), 'hidden_bias': self.ts_hidden_bias.get_value(),
                'visible_bias': self.ts_visible_bias.get_value()}

    @parameters.setter
    def parameters(self, value):
        self.ts_weights.set_value(value['weights'])
        self.ts_hidden_bias.set_value(value['hidden_bias'])
        self.ts_visible_bias.set_value(value['visible_bias'])

    def prop_up(self, v):
        pre_sigmoid_activation = T.dot(v, self.ts_weights) + self.ts_hidden_bias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def prop_down(self, h):
        pre_sigmoid_activation = T.dot(h, self.ts_weights.T) + self.ts_visible_bias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self, v_sample):
        pre_sigmoid_h, h = self.prop_up(v_sample)
        h_sample = self.sampler.sample_binary_state(h)
        return [pre_sigmoid_h, h, h_sample]

    def sample_v_given_h(self, h_sample):
        pre_sigmoid_v, v = self.prop_down(h_sample)
        v_sample = self.sampler.sample_binary_state(v)
        return [pre_sigmoid_v, v, v_sample]

    def gibbs_hvh(self, h_sample):
        pre_sigmoid_v, v, v_sample = self.sample_v_given_h(h_sample)
        pre_sigmoid_h, h, h_sample = self.sample_h_given_v(v_sample)
        return [pre_sigmoid_v, v, v_sample, pre_sigmoid_h, h, h_sample]

    def gibbs_vhv(self, v_sample):
        pre_sigmoid_h, h, h_sample = self.sample_h_given_v(v_sample)
        pre_sigmoid_v, v, v_sample = self.sample_v_given_h(h_sample)
        return [pre_sigmoid_h, h, h_sample, pre_sigmoid_v, v, v_sample]

    def free_energy(self, v_sample):
        wx_b = T.dot(v_sample, self.ts_weights) + self.ts_hidden_bias
        visible_bias_term = T.dot(v_sample, self.ts_visible_bias)
        hidden_bias_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_bias_term - visible_bias_term

    def get_updates(self, input, lr, persistent=None, k=1):
        # compute positive phase
        pre_sigmoid_h, h, h_sample = self.sample_h_given_v(input)

        # for CD, we use the newly generate hidden sample
        # for PCD, we initialize from the old state of the chain
        if persistent is None:
            chain_start = h_sample
        else:
            chain_start = persistent

        (
            [
                pre_sigmoid_nvs,
                nv_means,
                nv_samples,
                pre_sigmoid_nhs,
                nh_means,
                nh_samples
            ],
            updates
        ) = theano.scan(
            self.gibbs_hvh,
            # the None are place holders, saying that
            # chain_start is the initial state corresponding to the
            # 6th output
            outputs_info=[None, None, None, None, None, chain_start],
            n_steps=k,
            name="gibbs_hvh"
        )

        chain_end = nv_samples[-1]

        cost = T.mean(self.free_energy(input)) - T.mean(self.free_energy(chain_end))

        dw, dhb, dvb = T.grad(cost, [self.ts_weights, self.ts_hidden_bias, self.ts_visible_bias],
                              consider_constant=[chain_end])

        momentum_update = self.ts_momentum_speed * self.momentum_factor - lr * dw
        momentum_hidden_update = self.ts_momentum_speed_hidden_bias * self.momentum_factor - lr * dhb
        momentum_visible_update = self.ts_momentum_speed_visible_bias * self.momentum_factor - lr * dvb

        for k, v in [(self.ts_momentum_speed, momentum_update), (self.ts_weights, self.ts_weights + momentum_update),
                     (self.ts_momentum_speed_hidden_bias, momentum_hidden_update),
                     (self.ts_hidden_bias, self.ts_hidden_bias + momentum_hidden_update),
                     (self.ts_momentum_speed_visible_bias, momentum_visible_update),
                     (self.ts_visible_bias, self.ts_visible_bias + momentum_visible_update)]:
            updates[k] = v

        if persistent:
            # Note that this works only if persistent is a shared variable
            updates[persistent] = nh_samples[-1]
            # pseudo-likelihood is a better proxy for PCD
            monitoring_cost = self.get_pseudo_likelihood_cost(input, updates)
        else:
            # reconstruction cross-entropy is a better proxy for CD
            monitoring_cost = self.get_reconstruction_cost(input, pre_sigmoid_nvs[-1])
        return monitoring_cost, updates

    def get_pseudo_likelihood_cost(self, input, updates):
        """Stochastic approximation to the pseudo-likelihood"""

        # index of bit i in expression p(x_i | x_{\i})
        bit_i_idx = theano.shared(value=0, name='bit_i_idx')

        # binarize the input image by rounding to nearest integer
        xi = T.round(input)

        # calculate free energy for the given bit configuration
        fe_xi = self.free_energy(xi)

        # flip bit x_i of matrix xi and preserve all other bits x_{\i}
        # Equivalent to xi[:,bit_i_idx] = 1-xi[:, bit_i_idx], but assigns
        # the result to xi_flip, instead of working in place on xi.
        xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])

        # calculate free energy with bit flipped
        fe_xi_flip = self.free_energy(xi_flip)

        # equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i})))
        cost = T.mean(self.nb_visible * T.log(T.nnet.sigmoid(fe_xi_flip -
                                                             fe_xi)))

        # increment bit_i_idx % number as part of updates
        updates[bit_i_idx] = (bit_i_idx + 1) % self.nb_visible

        return cost

    @staticmethod
    def get_reconstruction_cost(input, pre_sigmoid_nv):
        cross_entropy = T.mean(
            T.sum(
                input * T.log(T.nnet.sigmoid(pre_sigmoid_nv)) +
                (1 - input) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv)),
                axis=1
            )
        )
        return cross_entropy
