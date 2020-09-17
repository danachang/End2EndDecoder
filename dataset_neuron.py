import os.path as osp
import numpy as np


class Dataset(object):

    def __init__(self, ids, labels, is_train=True, name='default'):
        self._ids = list(ids)
        self._labels = labels
        self.name = name
        self.is_train = is_train

    def get_data(self, id):
        activity = np.load(id)
        label = self._labels[id]
        return activity, label

    def get_data_ROI(self, id):
        activity = np.load(id)
        activity = activity/255.*2-1
        label = self._labels[id]
        return activity, label

    @property
    def ids(self):
        return self._ids

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        return 'Dataset (%s, %d examples)' % (
            self.name,
            len(self)
        )


def create_default_splits(path, is_train=True):
    train_ids, train_labels = get_activity_path_and_label(osp.join(path, 'train'))
    val_ids, val_labels = get_activity_path_and_label(osp.join(path, 'val'))
    test_ids, test_labels = get_activity_path_and_label(osp.join(path, 'test'))
    dataset_train = Dataset(train_ids, train_labels, name='train', is_train=True)
    dataset_val = Dataset(val_ids, val_labels, name='val', is_train=False)
    dataset_test = Dataset(test_ids, test_labels, name='test', is_train=False)
    return dataset_train, dataset_val, dataset_test


def get_activity_path_and_label(path):
    ids = []
    labels = {}
    with open(osp.join(path, 'label.txt')) as f:
        lines = [line.strip() for line in f.readlines()]
        for line in lines:
            newline = list(filter(str.strip, line.split(' ')))
            id = osp.join(path, newline[0])
            ids.append(id)
            labels[id] = np.array([float(n) for n in newline[1:]])

    rs = np.random.RandomState(123)
    rs.shuffle(ids)
    return ids, labels
