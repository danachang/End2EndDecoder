from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import sys
from six.moves import xrange
from pprint import pprint
import tensorflow as tf
import tensorflow.contrib.slim as slim

from input_ops_neuron import create_input_ops, create_input_ops_ROI
from util import log
from config_neuron import argparser
from tensorflow.python.util import deprecation
from tensorflow.python.ops import variables # add for test


class Trainer(object):

    def __init__(self, config, model, dataset):
        self.config = config
        self.model = model
        learning_hyperparameter_str = '{}_{}_bs_{}_lr_{}'.format(
            os.path.basename(config.dataset_path),
            config.loss_type, config.batch_size,
            config.learning_rate)
        model_hyperparameter_str = '{}_fc_{}_norm_act_{}'.format(
            config.num_d_fc, config.d_norm_type, config.output_act_fn)

        self.train_dir = './train_dir/%s/%s-%s' % (
            config.prefix,
            learning_hyperparameter_str + '_' + model_hyperparameter_str,
            time.strftime("%Y%m%d-%H%M%S")
        )

        os.makedirs(self.train_dir)
        log.infov("Train Dir: %s", self.train_dir)

        # --- input ops ---
        self.batch_size = config.batch_size

        if config.prefix == 'ROI':
            _, self.batch_train = create_input_ops_ROI(
                dataset, self.batch_size, is_training=True, shuffle=False)
        else:
            _, self.batch_train = create_input_ops(
                dataset, self.batch_size, is_training=True, shuffle=False)

        # --- optimizer ---
        self.global_step = tf.contrib.framework.get_or_create_global_step(graph=None)

        # --- checkpoint and monitoring ---
        all_var = tf.trainable_variables()

        # added in order to remove variables in VGG
        if self.config.load_pretrained:
            new_all_var = [v for v in all_var if 'block' not in v.name]
        else:
            new_all_var = all_var

        self.optimizer = tf.train.AdamOptimizer(
            self.config.learning_rate,
            beta1=self.config.adam_beta1, beta2=self.config.adam_beta2
        ).minimize(self.model.loss, var_list=new_all_var,
                   name='optimize_loss', global_step=self.global_step)

        """
        self.optimizer = tf.train.AdadeltaOptimizer(
            self.config.learning_rate, rho=0.95).minimize(self.model.loss,
                var_list=new_all_var, name='optimize_loss',
                global_step=self.global_step)


        self.optimizer = tf.train.GradientDescentOptimizer(
            self.config.learning_rate).minimize(self.model.loss,
                var_list=new_all_var, name='optimize_loss',
                global_step=self.global_step)

        """

        self.summary_op = tf.summary.merge_all()

        self.saver = tf.train.Saver(max_to_keep=1000)
        pretrain_saver = tf.train.Saver(var_list=all_var, max_to_keep=1)
        self.summary_writer = tf.summary.FileWriter(self.train_dir)

        self.supervisor = tf.train.Supervisor(
            logdir=self.train_dir,
            is_chief=True,
            saver=None,
            summary_op=None,
            summary_writer=self.summary_writer,
            save_summaries_secs=300,
            save_model_secs=600,
            global_step=self.global_step,
        )

        session_config = tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True),
            device_count={'GPU': 1},
        )
        self.session = self.supervisor.prepare_or_wait_for_session(config=session_config)

        def load_checkpoint(ckpt_path, saver, name=None):
            if ckpt_path is not None:
                log.info("Checkpoint path for {}: {}".format(name, ckpt_path))
                saver.restore(self.session, ckpt_path)
                log.info("Loaded the pretrain parameters " +
                         "from the provided checkpoint path.")

        load_checkpoint(
            config.checkpoint, pretrain_saver, name='All vars')

    def train(self):
        log.infov("Training Starts!")
        pprint(self.batch_train)

        step = self.session.run(self.global_step)

        for s in xrange(self.config.max_training_steps):

            if s % self.config.ckpt_save_step == 0:
                log.infov("Saved checkpoint at %d", s)
                self.saver.save(self.session, os.path.join(
                    self.train_dir, 'model'), global_step=s)

            step, summary, loss, step_time = \
                self.run_single_step(self.batch_train, step=s, is_train=True)

            if s % self.config.log_step == 0:
                self.log_step_message(step, loss, step_time)

            if s % self.config.write_summary_step == 0:
                self.summary_writer.add_summary(summary, global_step=step)

        # add for testing start_queue_runner
        #coord.request_stop()
        #coord.join(threads, stop_grace_period_secs=3)
        # end for testing start_queue_runner

    def run_single_step(self, batch, step=None, is_train=True):
        _start_time = time.time()
        #log.infov('now start to run')
        batch_chunk = self.session.run(batch)
        #log.infov('done with run')

        # below for testing what files lie in each batch (generate txt)
        """
        orig_stdout = sys.stdout
        f = open(self.config.prefix + 'out.txt', 'a')
        sys.stdout = f
        log.infov("Show Batch Chunk")
        pprint(batch_chunk['id'])
        sys.stdout = orig_stdout
        f.close()
        """
        # end for testing what files lie in each batch (generate txt)

        fetch = [self.global_step, self.summary_op,
                 self.model.loss, self.optimizer, self.model.ori_loss]

        fetch_values = self.session.run(
            fetch,
            feed_dict=self.model.get_feed_dict(batch_chunk)
        )
        [step, summary, loss, _, ori_loss] = fetch_values[:]

        _end_time = time.time()

        return step, summary, loss, (_end_time - _start_time)

    def log_step_message(self, step, loss, step_time, is_train=True):
        if step_time == 0: step_time = 0.001
        log_fn = (is_train and log.info or log.infov)
        log_fn((
            " [{split_mode:5s} step {step:4d}] " +
            "Loss: {loss:.5f} " +
            "({sec_per_batch:.3f} sec/batch, {instance_per_sec:.3f} instances/sec) "
            ).format(split_mode=(is_train and 'train' or 'val'),
                     step=step, loss=loss,
                     sec_per_batch=step_time,
                     instance_per_sec=self.batch_size / step_time))

        # below for testing what files lie in each batch (generate txt)
        """
        orig_stdout = sys.stdout
        f = open(self.config.prefix + 'out.txt', 'a')
        sys.stdout = f
        pprint("Step {step:4d}".format(step=step))
        sys.stdout = orig_stdout
        f.close()
        """
        # end for testing what files lie in each batch (generate txt)

def main():

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    config, model, dataset_train, dataset_val, dataset_test = argparser(is_train=True)

    trainer = Trainer(config, model, dataset_train)

    log.warning("dataset_path: %s, learning_rate: %f",
                config.dataset_path, config.learning_rate)
    trainer.train()

if __name__ == '__main__':
    main()
