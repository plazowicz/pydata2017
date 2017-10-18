import cPickle


class MNISTLoader(object):
    def __init__(self, data_path):
        with open(data_path, 'rb') as f:
            train_set, valid_set, test_set = cPickle.load(f)
        self.train_x, self.train_y = train_set
        self.valid_x, self.valid_y = valid_set
        self.test_x, self.test_y = test_set

    def generate_batches(self, batch_size=128, mode='train'):
        x = getattr(self, '%s_x' % mode)
        for i in xrange(0, x.shape[0], batch_size):
            current_batch_size = min(batch_size, x.shape[0] - i)
            yield x[i:i + current_batch_size]