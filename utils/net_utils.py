import tensorlayer as tl


# Assumption that first layer is fully connected
def read_latent_dim_from_gen_weights(gen_weights_path):
    gen_params = tl.files.load_npz(path=gen_weights_path, name='')
    return gen_params[0].shape[0]
