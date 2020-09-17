from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import scipy.ndimage as ndi
from six.moves import xrange
from pprint import pprint
import tensorflow as tf
import matplotlib.cm as cm
import matplotlib.colors as mplc
from matplotlib import pyplot as plt
import tensorflow.contrib.slim as slim
from tensorflow.python.util import deprecation

from util import log
from config import argparser
from model import Model
from ops import flatten

class getEnsembler(object):

    def __init__(self, config, model, dataset, dataset_type_str):
        self.config = config
        self.model = model
        self.train_dir = config.train_dir
        self.dataset_type_str = dataset_type_str
        self.summary_file = dataset_type_str + '_' + config.summary_file
        self.summary_model_file = dataset_type_str + '_' + config.summary_model_file
        self.summary_indv_file = dataset_type_str + '_' + config.summary_indv_file
        log.infov("Train_dir path = %s", self.train_dir)

        # --- input ops ---
        self.batch_size = config.batch_size
        self.dataset_path = config.dataset_path
        self.dataset = dataset

        # -- session --
        tf.set_random_seed(1234)

        session_config = tf.ConfigProto(
            #allow_soft_placement=True,
            #gpu_options=tf.GPUOptions(allow_growth=True),
            device_count={'GPU': 0},
        )
        self.session = tf.Session(config=session_config)

        # --- checkpoint and monitoring ---
        self.saver = tf.train.Saver(max_to_keep=100)

        self.checkpoint = config.checkpoint
        if self.checkpoint is None and self.train_dir:
            self.checkpoint = tf.train.latest_checkpoint(self.train_dir)
            log.infov("Checkpoint path : %s", self.checkpoint)
        elif self.checkpoint is None:
            log.warn("No checkpoint is given. Just random initialization.")
            self.session.run(tf.global_variables_initializer())
        elif self.train_dir:
            self.checkpoint = os.path.join(self.train_dir, self.checkpoint)
            log.infov("Checkpoint path : %s", self.checkpoint)
        else:
            log.infov("Checkpoint path : %s", self.checkpoint)

        # --- vars ---
        reader = tf.train.NewCheckpointReader(self.checkpoint)
        self.model_vars = list(reader.get_variable_to_shape_map().keys())

        # -- directory setup --
        if self.train_dir is None:
            train_dir_base = os.path.basename(os.path.dirname(self.checkpoint))
            train_dir_top = os.path.dirname(os.path.dirname(self.checkpoint))
            train_dir_top = os.path.dirname(os.path.dirname(os.path.dirname(train_dir_top)))
        else:
            train_dir_base = os.path.basename(self.train_dir)
            train_dir_top = os.path.dirname(self.train_dir)
            train_dir_top = os.path.dirname(os.path.dirname(os.path.dirname(train_dir_top)))

        checkpoint_base = os.path.basename(self.checkpoint)

        self.ensemble_dir = '%s/ensemble_dir/%s/%s/%s' %(train_dir_top,
                            self.config.prefix, train_dir_base, checkpoint_base)

        #self.ensemble_dir = './ensemble_dir/%s/%s/%s' %(self.config.prefix,
        #                                    train_dir_base, checkpoint_base)

        if not os.path.exists(self.ensemble_dir):
            log.infov("create ensemble_dir: %s", self.ensemble_dir)
            os.makedirs(self.ensemble_dir)
        else:
            log.infov("ensemble_dir exists: %s", self.ensemble_dir)

    def ensemble_run(self):
        # load checkpoint
        if self.checkpoint:
            self.saver.restore(self.session, self.checkpoint)
            log.info("Loaded from checkpoint!")

        log.infov("Start 1-epoch Inference and Evaluation")
        log.info("# of examples = %d", len(self.dataset))

        id_list = self.dataset.ids
        id_list = sorted(id_list, key=lambda x: int(x.split('/')[-1].split('.')[0].replace('t', '')))

        _ids = []
        _truelabel = []
        _predlabel = []
        _features = []
        _final_features = []
        _actv1_features = []
        _actv2_features = []
        _actv3_features = []
        _actv4_features = []
        _actv5_features = []
        _actv6_features = []
        _actv7_features = []

        try:
            step = None
            s = 0
            continue_evaluate = True
            while continue_evaluate:

                batch_id_list = id_list[self.batch_size*s:self.batch_size*(s+1)]

                if not batch_id_list:
                    print('empty batch list')
                else:
                    if len(batch_id_list) < self.batch_size:
                        self.config.batch_size = len(batch_id_list)
                        self.model = Model(self.config, is_train=False)

                    id = []
                    image = []
                    label = []

                    if self.config.arch == 'ResNet50':
                        for id_data in batch_id_list:
                            m, l = self.dataset.get_data_resnet(id_data)
                            id0 = id_data.split('/')[-1].split('.')[0]
                            image.append(m)
                            label.append(l)
                            id.append(id0)
                    else:
                        for id_data in batch_id_list:
                            m, l = self.dataset.get_data(id_data)
                            id0 = id_data.split('/')[-1].split('.')[0]
                            image.append(m)
                            label.append(l)
                            id.append(id0)

                    batch_chunk = {
                        'id': np.stack(id, axis=0),
                        'image': np.stack(image, axis=0),
                        'label': np.stack(label, axis=0)
                    }

                    feature_layer = flatten(self.model.actv_list[-1])
                    fc_layer = flatten(self.model.fc_list[-1])

                    actv1_layer = self.model.actv_list[0]
                    actv1_layer = tf.reduce_mean(actv1_layer, axis=(1, 2))

                    actv2_layer = self.model.actv_list[1]
                    actv2_layer = tf.reduce_mean(actv2_layer, axis=(1, 2))

                    actv3_layer = self.model.actv_list[2]
                    actv3_layer = tf.reduce_mean(actv3_layer, axis=(1, 2))

                    actv4_layer = self.model.actv_list[3]
                    actv4_layer = tf.reduce_mean(actv4_layer, axis=(1, 2))

                    if len(self.model.actv_list) >= 5:
                        actv5_exist = 1
                        actv5_layer = self.model.actv_list[4]
                        actv5_layer = tf.reduce_mean(actv5_layer, axis=(1, 2))

                        actv6_layer = self.model.actv_list[5]
                        actv6_layer = tf.reduce_mean(actv6_layer, axis=(1, 2))

                        actv7_layer = self.model.actv_list[6]
                        actv7_layer = tf.reduce_mean(actv7_layer, axis=(1, 2))

                        [pred_label, features, fc, actv1, actv2, actv3, \
                            actv4, actv5, actv6, actv7] = self.session.run(
                            [self.model.pred_label, feature_layer, fc_layer, \
                            actv1_layer, actv2_layer, actv3_layer, actv4_layer, \
                            actv5_layer, actv6_layer, actv7_layer],
                            feed_dict=self.model.get_feed_dict(batch_chunk)
                        )
                    else:
                        actv5_exist = 0
                        [pred_label, features, fc, actv1, actv2, actv3, actv4] = self.session.run(
                            [self.model.pred_label, feature_layer, fc_layer, \
                            actv1_layer, actv2_layer, actv3_layer, actv4_layer],
                            feed_dict=self.model.get_feed_dict(batch_chunk)
                        )

                    _ids.append(id)
                    _truelabel.append(batch_chunk['label'])
                    _predlabel.append(pred_label)
                    _features.append(features)
                    _final_features.append(fc)
                    _actv1_features.append(actv1)
                    _actv2_features.append(actv2)
                    _actv3_features.append(actv3)
                    _actv4_features.append(actv4)
                    if actv5_exist:
                        _actv5_features.append(actv5)
                        _actv6_features.append(actv6)
                        _actv7_features.append(actv7)


                s += 1
                continue_evaluate = (s < len(self.dataset)/self.batch_size)

                if not continue_evaluate:
                    ids = []
                    for elem in _ids:
                        ids.extend(elem)
                    ids = np.asarray(ids)

                    truelabel = np.vstack(_truelabel)
                    predlabel = np.vstack(_predlabel)

                    ensemble = np.vstack(_features)
                    fc_ensemble = np.vstack(_final_features)
                    actv1_ensemble = np.vstack(_actv1_features)
                    actv2_ensemble = np.vstack(_actv2_features)
                    actv3_ensemble = np.vstack(_actv3_features)
                    actv4_ensemble = np.vstack(_actv4_features)
                    if actv5_exist:
                        actv5_ensemble = np.vstack(_actv5_features)
                        actv6_ensemble = np.vstack(_actv6_features)
                        actv7_ensemble = np.vstack(_actv7_features)

                    prepath_str = self.ensemble_dir + '/' + self.dataset_type_str
                    filename = prepath_str + '_ensemble.npz'

                    if actv5_exist:
                        np.savez(filename, id=ids, label=truelabel,
                                    pred_label=predlabel,
                                    ensemble=ensemble,
                                    fc_ensemble=fc_ensemble,
                                    actv1_ensemble=actv1_ensemble,
                                    actv2_ensemble=actv2_ensemble,
                                    actv3_ensemble=actv3_ensemble,
                                    actv4_ensemble=actv4_ensemble,
                                    actv5_ensemble=actv5_ensemble,
                                    actv6_ensemble=actv6_ensemble,
                                    actv7_ensemble=actv7_ensemble)
                    else:
                        np.savez(filename, id=ids, label=truelabel,
                                    pred_label=predlabel,
                                    ensemble=ensemble,
                                    fc_ensemble=fc_ensemble,
                                    actv1_ensemble=actv1_ensemble,
                                    actv2_ensemble=actv2_ensemble,
                                    actv3_ensemble=actv3_ensemble,
                                    actv4_ensemble=actv4_ensemble)

                    log.infov(self.dataset_type_str + ' ensemble saved')

        except Exception as e:
            print(e)
            log.infov('ohohoh stop')

        log.warning('Ensemble completed')


def main():

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    config, model, dataset_train, dataset_val, dataset_test = argparser(is_train=False)
    log.warning("dataset path: %s", config.dataset_path)

    ensembler_val = getEnsembler(config, model, dataset_val, 'val')
    ensembler_val.ensemble_run()
    config.batch_size = ensembler_val.batch_size

    ensembler_train = getEnsembler(config, model, dataset_train, 'train')
    ensembler_train.ensemble_run()
    config.batch_size = ensembler_train.batch_size

    ensembler_test = getEnsembler(config, model, dataset_test, 'test')
    ensembler_test.ensemble_run()

if __name__ == '__main__':
    main()
