#!/usr/bin/env python
import tensorflow as tf
from ops import fc, flatten, act_str2fn
from util import log


class Decoder_Neuron(object):
    def __init__(self, name, output_dim, output_act_fn,
                 num_fc, norm_type, is_train):
        self.name = name
        self._output_dim = output_dim
        self._output_act_fn = act_str2fn(output_act_fn)
        self._num_fc = num_fc
        self._norm_type = norm_type
        self._is_train = is_train
        self._reuse = False
        self._reuse = tf.AUTO_REUSE # added for testing

    def __call__(self, input):
        with tf.variable_scope(self.name, reuse=self._reuse):
            if not self._reuse:
                log.warn(self.name)
            _ = input
            print('input tensor')
            print(_)

            # fc layers
            num_fc_channel = [256, 64, 32, 16]
            for i in range(min(self._num_fc, len(num_fc_channel))):
                _ = fc(_, num_fc_channel[i], self._is_train, info=not self._reuse,
                       norm=self._norm_type, name='fc{}'.format(i+1))

            if self._num_fc == 0:
                _ = fc(_, self._output_dim, self._is_train, info=not self._reuse,
                           activation_fn=self._output_act_fn, norm='none',
                           name='fc{}'.format(1))
            else:
                _ = fc(_, self._output_dim, self._is_train, info=not self._reuse,
                           activation_fn=self._output_act_fn, norm='none',
                           name='fc{}'.format(i+2))

            self._reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
            """
            self.allvar = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, self.name)
            log.infov('var list')
            log.infov(self.var_list)
            log.infov('all var')
            log.infov(self.allvar)
            """
            return _
