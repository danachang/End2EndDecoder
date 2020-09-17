from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from util import log
from decoder_selfAttn import Decoder
from decoder_mdl import Decoder_Mdl


class Model(object):

    def __init__(self, config,
                 debug_information=False,
                 is_train=True):
        self.debug = debug_information

        self.config = config
        self.batch_size = config.batch_size
        self.h = config.h
        self.w = config.w
        self.c = config.c
        self.output_dim = config.output_dim
        self.output_act_fn = config.output_act_fn
        self.num_d_conv = config.num_d_conv
        self.num_d_fc = config.num_d_fc
        self.d_norm_type = config.d_norm_type
        self.loss_type = config.loss_type

        # added for Decoder_mdl
        self.load_pretrained = config.load_pretrained
        self.arch = config.arch

        # create placeholders for the input
        self.image = tf.placeholder(
            name='image', dtype=tf.float32,
            shape=[self.batch_size, self.h, self.w, self.c],
        )

        self.label = tf.placeholder(
            name='label', dtype=tf.float32,
            shape=[self.batch_size, self.output_dim],
        )

        self.build(is_train=is_train)

    def get_feed_dict(self, batch_chunk):
        fd = {
            self.image: batch_chunk['image'],  # [bs, h, w, c]
            self.label: batch_chunk['label'],  # [bs, v] (v should be 3)
        }
        return fd

    def build(self, is_train=True):

        # Decoder {{{
        # =========
        # Input: an image [bs, h, w, c]
        # Output: [bs, [x, y, v]]

        if self.arch == 'ConvNet':
            D = Decoder('Decoder',  self.output_dim, self.output_act_fn,
                        self.num_d_conv, self.num_d_fc,
                        self.d_norm_type, is_train)
        else:
            D = Decoder_Mdl('Decoder_Mdl',  self.output_dim, self.output_act_fn,
                                    self.num_d_conv, self.num_d_fc,
                                    self.d_norm_type, is_train,
                                    self.load_pretrained, self.arch)

        pred_label, conv_list, actv_list, attn_list = D(self.image)
        self.pred_label = pred_label
        self.conv_list = conv_list
        self.actv_list = actv_list
        self.attn_list = attn_list
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
        tf.summary.image("image", self.image)
        # }}}

        # Output {{{
        # =========
        self.output = {
            'pred_label': pred_label
        }
        # }}}

        log.warn('\033[93mSuccessfully loaded the model.\033[0m')
