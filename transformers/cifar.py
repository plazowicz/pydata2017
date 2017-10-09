import os.path as op
import cPickle as pickle
import random

import numpy as np

from utils.logger import log
from utils.img_ops import bound_image_values


@log
class CifarDataset(object):
    def __init__(self, cifar_ds_path, batch_size, how_many_labeled):
        self.cifar_ds_path = cifar_ds_path
        self.how_many_labeled = how_many_labeled

        self.train_data, self.train_labels = self.__read_train_data()
        self.test_data, self.test_labels = self.__read_batch_file_into_array(op.join(cifar_ds_path, 'test_batch'))
        self.batch_size = batch_size

        self.logger.info(
            "Loaded CIFAR dataset, train size = %d, test size = %d, labeled = %d" % (self.train_data.shape[0],
                                                                                     self.test_data.shape[0],
                                                                                     self.how_many_labeled))

    def generate_mb(self, ds_type):
        assert ds_type in ['train', 'test']
        train_size = self.train_data.shape[0]
        ds_to_generate, ds_labels = (self.train_data, self.train_labels) if ds_type == 'train' else \
            (self.test_data, self.test_labels)

        for i in xrange(0, train_size, self.batch_size):
            data_batch = ds_to_generate[i: i + self.batch_size, :, :, :]
            labels_batch = ds_labels[i: i + self.batch_size]
            yield bound_image_values(data_batch).astype(np.float32), self.__process_labels(labels_batch)

    def __read_train_data(self):
        train_batch_files = [op.join(self.cifar_ds_path, p) for p in ["data_batch_%d" % i for i in xrange(1, 6)]]
        train_data, train_labels = [], []
        for bf in train_batch_files:
            bf_data, bf_labels = self.__read_batch_file_into_array(bf)
            train_data.append(bf_data)
            train_labels.append(bf_labels)

        train_data, train_labels = np.concatenate(train_data, axis=0), np.concatenate(train_data, axis=0)
        return train_data, self.__nullify_labels(train_labels)

    def __process_labels(self, labels):
        final_labels = np.zeros((self.batch_size, 10), dtype=np.int32)
        for i, l in enumerate(labels):
            if l is None:
                final_labels[i, :] = 1
            else:
                final_labels[i, l] = 1
        return final_labels

    def __nullify_labels(self, labels):
        ex_per_class = int(self.how_many_labeled / 10)
        indices = xrange(len(labels))
        random.shuffle(indices)
        classes_indices = {c: (0, []) for c in xrange(10)}
        for indx in indices:
            if all(len(class_indices) == ex_per_class for _, class_indices in classes_indices.iteritems()):
                break
            indx_label = labels[indx]
            indx_label_ex_num, indx_label_indices = classes_indices[indx_label]
            if indx_label_ex_num == ex_per_class:
                continue
            indx_label_indices.append(indx)
            classes_indices[indx_label] = (indx_label_ex_num + 1, indx_label_indices)

        train_indices = set(reduce(lambda acc, elem: acc + elem._1, classes_indices.values(), []))
        self.logger.info("Collected %d examples per class, labeled examples = %d" % (ex_per_class, len(train_indices)))
        nullified_labels = [l if i in train_indices else None for i, l in enumerate(labels)]
        return nullified_labels

    @staticmethod
    def __read_batch_file_into_array(batch_file):
        with open(batch_file, 'rb') as f:
            batch_dict = pickle.load(f)
        data = batch_dict['data']
        labels = batch_dict['labels']

        data = data.reshape((data.shape[0], 3, 32, 32))
        data = data.transpose((0, 2, 3, 1))
        return data, labels
