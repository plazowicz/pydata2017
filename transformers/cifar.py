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

        train_data, train_labels = self.__read_train_data()
        self.lab_data, self.unlab_data, self.labels = self.__split_into_lab_unlab(train_data, train_labels)
        self.test_data, self.test_labels = self.__read_batch_file_into_array(op.join(cifar_ds_path, 'test_batch'))
        self.batch_size = batch_size

        self.logger.info(
            "Loaded CIFAR dataset, train size = %d, test size = %d, labeled = %d" % (train_data.shape[0],
                                                                                     self.test_data.shape[0],
                                                                                     self.how_many_labeled))

        self.rng = np.random.RandomState()

    def classes_num(self):
        return 10

    def generate_train_mb(self):
        train_size = self.unlab_data.shape[0]
        inds = self.rng.permutation(train_size)
        unlab_data = self.unlab_data[inds, :, :, :]
        lab_data, labels = self.extend_labeled_data(self.lab_data, self.labels, unlab_data)

        for i in xrange(0, train_size, self.batch_size):
            lab_data_batch = lab_data[i: i + self.batch_size, :, :, :]
            unlab_data_batch = unlab_data[i: i + self.batch_size, :, :, :]
            labels_batch = labels[i: i + self.batch_size]
            lab_data_batch, labels_batch = self.fill_batch(lab_data_batch, labels_batch, lab_data, labels)

            if unlab_data_batch.shape[0] < self.batch_size:
                unlab_data_batch = np.concatenate((unlab_data_batch, unlab_data[0: self.batch_size -
                                                                                   unlab_data_batch.shape[0]]), axis=0)

            yield bound_image_values(lab_data_batch).astype(np.float32), \
                  bound_image_values(unlab_data_batch).astype(np.float32), self.one_hot_labels(labels_batch)

    def generate_test_mb(self):
        for i in xrange(0, self.test_data.shape[0], self.batch_size):
            data_batch = self.test_data[i: i + self.batch_size, :, :, :]
            labels_batch = self.test_labels[i: i + self.batch_size]

            data_batch, labels_batch = self.fill_batch(data_batch, labels_batch, self.test_data, self.test_labels)
            data_batch = bound_image_values(data_batch).astype(np.float32)
            yield data_batch, self.one_hot_labels(labels_batch)

    def fill_batch(self, data_batch, labels_batch, data, labels):
        if data_batch.shape[0] < self.batch_size:
            labels_batch = np.concatenate((labels_batch, labels[0: self.batch_size - data_batch.shape[0]]), axis=0)
            data_batch = np.concatenate((data_batch, data[0: self.batch_size - data_batch.shape[0]]), axis=0)
        return data_batch, labels_batch

    def extend_labeled_data(self, lab_data, labels, unlab_data):
        lab_x, lab_y = [], []
        for t in xrange(int(np.ceil(unlab_data.shape[0] / float(lab_data.shape[0])))):
            inds = self.rng.permutation(lab_data.shape[0])
            lab_x.append(lab_data[inds, :, :, :])
            lab_y.append(labels[inds])
        lab_x = np.concatenate(lab_x, axis=0)
        lab_y = np.concatenate(lab_y, axis=0)
        return lab_x, lab_y

    def img_size(self):
        return 32

    def size(self):
        return self.unlab_data.shape[0]

    def __read_train_data(self):
        train_batch_files = [op.join(self.cifar_ds_path, p) for p in ["data_batch_%d" % i for i in xrange(1, 6)]]
        train_data, train_labels = [], []
        for bf in train_batch_files:
            bf_data, bf_labels = self.__read_batch_file_into_array(bf)
            train_data.append(bf_data)
            train_labels.append(bf_labels)

        train_data, train_labels = np.concatenate(train_data, axis=0), np.concatenate(train_labels, axis=0)
        return train_data, self.__nullify_labels(train_labels)

    @staticmethod
    def one_hot_labels(labels):
        final_labels = np.zeros((len(labels), 10), dtype=np.float32)
        for i, l in enumerate(labels):
            final_labels[i, l] = 1.
        return final_labels

    def __nullify_labels(self, labels):
        ex_per_class = int(self.how_many_labeled / 10)
        indices = list(xrange(len(labels)))
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

        train_indices = set(reduce(lambda acc, elem: acc + elem[1], classes_indices.values(), []))
        self.logger.info("Collected %d examples per class, labeled examples = %d" % (ex_per_class, len(train_indices)))
        nullified_labels = [l if i in train_indices else None for i, l in enumerate(labels)]
        return nullified_labels

    @staticmethod
    def __split_into_lab_unlab(data, labels):
        labeled_indices = [i for i, l in enumerate(labels) if l is not None]
        unlabeled_data = data.copy()
        labeled_data = data[labeled_indices, :, :, :]
        labels = np.array(labels)[labeled_indices]
        return labeled_data, unlabeled_data, labels

    @staticmethod
    def __read_batch_file_into_array(batch_file):
        with open(batch_file, 'rb') as f:
            batch_dict = pickle.load(f)
        data = batch_dict['data']
        labels = batch_dict['labels']

        data = data.reshape((data.shape[0], 3, 32, 32))
        data = data.transpose((0, 2, 3, 1))
        return data, labels


class SupervisedCifarDataset(CifarDataset):

    def generate_train_mb(self):
        train_size = self.lab_data.shape[0]
        for i in xrange(0, train_size, self.batch_size):
            lab_data_batch = self.lab_data[i: i + self.batch_size, :, :, :]
            labels_batch = self.labels[i: i + self.batch_size]
            lab_data_batch, labels_batch = self.fill_batch(lab_data_batch, labels_batch, self.lab_data, self.labels)
            yield bound_image_values(lab_data_batch).astype(np.float32), self.one_hot_labels(labels_batch)

    def size(self):
        return self.lab_data.shape[0]
