from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from util import log
from decoder_neuron import Decoder_Neuron


class Model(object):

    def __init__(self, config, debug_information=False, is_train=True):
        self.debug = debug_information
        self.config = config
        self.batch_size = config.batch_size
        self.l = config.l
        self.output_dim = config.output_dim
        self.output_act_fn = config.output_act_fn
        self.num_d_fc = config.num_d_fc
        self.d_norm_type = config.d_norm_type
        self.loss_type = config.loss_type

        # added for Decoder_mdl
        self.load_pretrained = config.load_pretrained
        self.arch = config.arch

        # create placeholders for the input
        self.activity = tf.placeholder(
            name='activity', dtype=tf.float32,
            shape=[self.batch_size, self.l],
        )

        self.label = tf.placeholder(
            name='label', dtype=tf.float32,
            shape=[self.batch_size, self.output_dim],
        )

        self.build(is_train=is_train)

    def get_feed_dict(self, batch_chunk):
        fd = {
            self.activity: batch_chunk['activity'],  # [bs, h, w, c]
            self.label: batch_chunk['label'],  # [bs, v] (v should be 3)
        }
        return fd

    def build(self, is_train=True):

        # Decoder {{{
        # =========
        # Input: an activity [bs, v]
        # Output: [bs, [x, y, v]]

        D = Decoder_Neuron('Decoder_Neuron',  self.output_dim, self.output_act_fn,
                        self.num_d_fc, self.d_norm_type, is_train)

        pred_label = D(self.activity)
        self.pred_label = pred_label

        # }}}

        # Build losses {{{
        # =========
        # compute loss
        if self.loss_type == 'l1':
            self.ori_loss = tf.abs(self.label - pred_label)
            self.loss = tf.reduce_mean(self.ori_loss)
        elif self.loss_type == 'l2':
            self.ori_loss = (self.label - pred_label) **2
            self.loss = tf.reduce_mean(self.ori_loss)
        else:
            raise NotImplementedError
        # }}}

        # TensorBoard summaries {{{
        # =========
        tf.summary.scalar("loss/loss", self.loss)
        # }}}

        # Output {{{
        # =========
        self.output = {
            'pred_label': pred_label
        }
        # }}}

        log.warn('\033[93mSuccessfully loaded the model.\033[0m')
