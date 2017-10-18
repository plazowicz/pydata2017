import argparse

import numpy as np
import theano
from datasets.mnist import MNISTLoader
from models.rbm import BinaryRBM
from utils.rbm_utils import Sampler, save_parameters, tile_raster_images
from PIL import Image


def train_rbm(args):
    sampler = Sampler(mode='cpu')
    data_loader = MNISTLoader(args.mnist_ds_path)
    nb_visible = 784
    nb_hidden = args.latent_dim
    n_samples = 10
    img_size = 28

    print "Compiling rbm"

    rbm = BinaryRBM(nb_visible, nb_hidden, learning_rate=args.learning_rate, momentum_factor=args.momentum,
                    persistent=True, batch_size=args.batch_size, k=args.k, sampler=sampler)

    rbm.train(data_loader, nb_epochs=args.epochs_num)
    save_parameters(rbm.parameters, args.out_weights_dir)

    # find out the number of test samples
    number_of_test_samples = data_loader.test_x.shape[0]

    # pick random test examples, with which to initialize the persistent chain
    n_chains = 20
    rng = np.random.RandomState(123)
    test_idx = rng.randint(number_of_test_samples - n_chains)
    persistent_vis_chain = theano.shared(
        np.asarray(
            data_loader.test_x[test_idx:test_idx + n_chains],
            dtype=theano.config.floatX
        )
    )
    # end-snippet-6 start-snippet-7
    plot_every = 1000
    # define one step of Gibbs sampling (mf = mean-field) define a
    # function that does `plot_every` steps before returning the
    # sample for plotting
    (
        [
            presig_hids,
            hid_mfs,
            hid_samples,
            presig_vis,
            vis_mfs,
            vis_samples
        ],
        updates
    ) = theano.scan(
        rbm.gibbs_vhv,
        outputs_info=[None, None, None, None, None, persistent_vis_chain],
        n_steps=plot_every,
        name="gibbs_vhv"
    )

    # add to updates the shared variable that takes care of our persistent
    # chain :.
    updates.update({persistent_vis_chain: vis_samples[-1]})
    # construct the function that implements our persistent chain.
    # we generate the "mean field" activations for plotting and the actual
    # samples for reinitializing the state of our persistent chain
    sample_fn = theano.function(
        [],
        [
            vis_mfs[-1],
            vis_samples[-1]
        ],
        updates=updates,
        name='sample_fn'
    )

    # create a space to store the image for plotting ( we need to leave
    # room for the tile_spacing as well)
    image_data = np.zeros(
        ((img_size + 1) * n_samples + 1, (img_size + 1) * n_chains - 1),
        dtype='uint8'
    )
    for idx in range(n_samples):
        # generate `plot_every` intermediate samples that we discard,
        # because successive samples in the chain are too correlated
        vis_mf, vis_sample = sample_fn()
        print(' ... plotting sample %d' % idx)
        image_data[(img_size + 1) * idx:(img_size + 1) * idx + img_size, :] = tile_raster_images(
            X=vis_mf,
            img_shape=(img_size, img_size),
            tile_shape=(1, n_chains),
            tile_spacing=(1, 1)
        )

    # construct image
    image = Image.fromarray(image_data)
    image.save(args.output_samples_path)


def main():
    args = parse_args()
    train_rbm(args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mnist_ds_path')
    parser.add_argument('out_weights_dir')
    parser.add_argument('output_samples_path')
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--epochs_num', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--momentum', type=float, default=0.)
    parser.add_argument('--k', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    main()
