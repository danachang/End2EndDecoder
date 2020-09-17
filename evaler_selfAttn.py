from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
from six.moves import xrange
from pprint import pprint
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.util import deprecation

from util import log
from config_selfAttn import argparser
from model_selfAttn import Model

class Evaler(object):

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

        self.global_step = tf.contrib.framework.get_or_create_global_step(graph=None)
        self.step_op = tf.no_op(name='step_no_op')

        # --- vars ---
        self.model_vars = tf.trainable_variables()
        log.warning("********* var ********** ")
        model_vars = slim.model_analyzer.analyze_vars(self.model_vars, print_info=True)
        self.num_model_params = model_vars[0]

        # -- session --
        tf.set_random_seed(1234)

        session_config = tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True),
            device_count={'GPU': 1},
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

        # -- directory setup --
        if self.train_dir is None:
            train_dir_base = os.path.basename(os.path.dirname(self.checkpoint))
        else:
            train_dir_base = os.path.basename(self.train_dir)

        checkpoint_base = os.path.basename(self.checkpoint)

        self.val_dir = './val_dir/%s/%s/%s' %(self.config.prefix,
                                              train_dir_base, checkpoint_base)
        print(self.val_dir)

    def eval_run(self):
        # load checkpoint
        if self.checkpoint:
            self.saver.restore(self.session, self.checkpoint)
            log.info("Loaded from checkpoint!")

        log.infov("Start 1-epoch Inference and Evaluation")
        log.info("# of examples = %d", len(self.dataset))

        _ids = []
        _predlabel = []
        _truelabel = []
        id_list = self.dataset.ids
        id_list = sorted(id_list, key=lambda x: int(x.split('/')[-1].split('.')[0].replace('t', '')))

        try:
            loss_avg = []
            loss_all = []
            time_all = 0
            step = None
            s = 0
            continue_evaluate = True
            while continue_evaluate:

                batch_id_list = id_list[self.batch_size*s:self.batch_size*(s+1)]

                if not batch_id_list:
                    print('empty batch list')
                #elif len(batch_id_list)<self.batch_size:
                    #print('discard the final batch')
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
                            image.append(m)
                            label.append(l)
                            id.append(id_data)
                    else:
                        for id_data in batch_id_list:
                            m, l = self.dataset.get_data(id_data)
                            image.append(m)
                            label.append(l)
                            id.append(id_data)

                    batch_chunk = {
                        'id': np.stack(id, axis=0),
                        'image': np.stack(image, axis=0),
                        'label': np.stack(label, axis=0)
                    }

                    step, step_time, ori_loss, pred_label = \
                        self.run_single_step(batch_chunk, step=s, is_train=False)

                    _ids.append(batch_chunk['id'])
                    _predlabel.append(pred_label)
                    _truelabel.append(batch_chunk['label'])

                    pred_label_list = np.vstack(_predlabel)
                    true_label_list = np.vstack(_truelabel)

                    # report losses
                    #loss_avg.append(np.average(np.array(ori_loss), axis=1))
                    loss_avg.append(np.average(ori_loss, axis=1))
                    loss_all.append(np.array(ori_loss))
                    loss_list = np.vstack(loss_all)


                time_all += step_time
                s += 1

                continue_evaluate = (s < len(self.dataset)/self.batch_size)

                self.log_step_message(
                    s, loss_list, loss_avg, time_all, id_list, true_label_list, pred_label_list,
                    val_dir=self.val_dir,
                    summary_file=self.summary_file,
                    summary_model_file=self.summary_model_file,
                    summary_indv_file=self.summary_indv_file,
                    final_step=not continue_evaluate,
                    dataset_type_str=self.dataset_type_str
                )


        except Exception as e:
            print(e)
            log.infov('ohohoh stop')

        log.warning('Evaluation completed')


    def run_single_step(self, batch_chunk, step=None, is_train=False):
        _start_time = time.time()

        [step, ori_loss, pred_label, _] = self.session.run(
            [self.global_step, self.model.ori_loss,
             self.model.pred_label, self.step_op],
            feed_dict=self.model.get_feed_dict(batch_chunk)
        )

        _end_time = time.time()

        return step, (_end_time - _start_time), ori_loss, pred_label

    def log_step_message(self, step, loss_list, loss_avg, step_time, id_list, true_label_list, pred_label_list, \
                         val_dir=None, summary_file=None, summary_model_file = None, summary_indv_file=None, \
                         final_step=False, dataset_type_str=None):

        if step_time == 0: step_time = 0.001

        if step == len(loss_avg):
            msg = (
             " [{split_mode:5s} step {step:4d}] " +
             "Total Average Loss for this Batch: {loss:.5f} "
            ).format(split_mode=dataset_type_str,
                      step=step,
                      loss=np.mean(loss_avg[step-1]))

            log.info(msg)

            loss_batch = "Average Loss for Each Sample: " + \
                        str(list(map("{:.5f}".format, loss_avg[step-1])))

        if final_step:
            if not os.path.exists(val_dir):
                print('create val_dir')
                os.makedirs(val_dir)
            else:
                print('val_dir exists')

            sumfilename = os.path.join(val_dir, summary_file)
            modelfilename = os.path.join(val_dir, summary_model_file)
            indvfilename = os.path.join(val_dir, summary_indv_file)
            np.set_printoptions(threshold=np.inf, precision=6, suppress=True)
            id_list = id_list[0:loss_list.shape[0]]
            loss_avglist = np.mean(loss_list, axis=1) # check diff with loss_avg
            loss_avglist = np.array([loss_avglist]).T

            id_num_list = [int(x.split('/')[-1].split('.')[0].replace('t', '')) for x in id_list]
            id_num_list = np.array([id_num_list]).T
            #print(id_num_list)

            total_loss_avg = np.mean(loss_avglist)
            each_loss_avg = np.mean(loss_list, axis=0)
            print('each', each_loss_avg)

            log.infov("Checkpoint: %s", self.checkpoint)
            log.infov("Dataset Path: %s", self. dataset_path)
            log.infov("Dataset: %s", self.dataset)
            log.infov("Write the summary to: %s", sumfilename)
            log.infov("Write the model summary to: %s", modelfilename)
            log.infov("Total Average Loss across %d samples: %.6f", id_num_list.shape[0], total_loss_avg)

            # model summay file writing
            formatspec = "%-50s%-50s\n"
            model_var_names = [p.name for p in self.model_vars]
            model_var_shape = [str(p.get_shape().as_list()) for p in self.model_vars]

            with open(modelfilename, 'w') as f:
                f.writelines(formatspec % ('Model_Variable_Name', 'Model_Variable_Shape'))
                f.writelines(formatspec % (n, s) for n, s in zip(model_var_names, model_var_shape))

            # individual summayry file writing
            if self.config.output_dim == 3:
                formatstr = "%-12s"*11 + "\n"
                formatstr2 = "%-12d" + "%-12.4f%-12.4f%-12.6f"*3 + "%-12.6f\n"
                liststr = ['ID', 'True_X', 'Pred_X', 'Loss_X',
                           'True_Y', 'Pred_Y', 'Loss_Y',
                          'True_Speed', 'Pred_Speed', 'Loss_Speed', 'Avg_Loss']

                x = np.vstack((true_label_list[:, 0], pred_label_list[:, 0], loss_list[:, 0])).T
                y = np.vstack((true_label_list[:, 1], pred_label_list[:, 1], loss_list[:, 1])).T
                speed = np.vstack((true_label_list[:, 2], pred_label_list[:, 2], loss_list[:, 2])).T
                listdata = np.hstack((id_num_list, x, y, speed, loss_avglist))

            elif self.config.output_dim == 2:
                formatstr = "%-12s"*8 + "\n"
                formatstr2 = "%-12d" + "%-12.4f%-12.4f%-12.6f"*2 + "%-12.6f\n"
                liststr = ['ID', 'True_X', 'Pred_X', 'Loss_X',
                           'True_Y', 'Pred_Y', 'Loss_Y', 'Avg_Loss']
                x = np.vstack((true_label_list[:, 0], pred_label_list[:, 0], loss_list[:, 0])).T
                y = np.vstack((true_label_list[:, 1], pred_label_list[:, 1], loss_list[:, 1])).T
                listdata = np.hstack((id_num_list, x, y, loss_avglist))

            elif self.config.output_dim == 1:
                formatstr = "%-12s"*5 + "\n"
                formatstr2 = "%-12d" + "%-12.4f%-12.4f%-12.6f%-12.6f\n"
                liststr = ['ID', 'True_Speed', 'Pred_Speed', 'Loss_Speed', 'Avg_Loss']
                speed = np.vstack((true_label_list[:, 0], pred_label_list[:, 0], loss_list[:, 0])).T
                listdata = np.hstack((id_num_list, speed, loss_avglist))
            else:
                raise NotImplementedError

            with open(indvfilename, 'w') as f:
                f.writelines(formatstr % tuple(liststr))
                f.writelines(formatstr2 % tuple(d) for d in listdata)


            # summary file writing
            formatspecs = "%-30s%-30s\n"
            formatspecs2 = "%-30s%-20d\n"
            formatspecs3 = "%-30s%-20.6f\n"
            with open(sumfilename, 'w') as f:
                f.writelines(formatspecs % ('Dataset_Path', self.dataset_path))
                f.writelines(formatspecs % ('Dataset_Info', self.dataset))
                f.writelines(formatspecs % ('Num_Samples', id_num_list.shape[0]))
                f.writelines(formatspecs % ('Model_Checkpoint', self.checkpoint))
                f.writelines(formatspecs2 % ('Num_Parameters', self.num_model_params))
                f.writelines(formatspecs3 % ('Total_Average_Loss', total_loss_avg))
                if self.config.output_dim == 3:
                    f.writelines(formatspecs3 % ('Average_Loss_X', each_loss_avg[0]))
                    f.writelines(formatspecs3 % ('Average_Loss_Y', each_loss_avg[1]))
                    f.writelines(formatspecs3 % ('Average_Loss_Speed', each_loss_avg[2]))
                elif self.config.output_dim == 2:
                    f.writelines(formatspecs3 % ('Average_Loss_X', each_loss_avg[0]))
                    f.writelines(formatspecs3 % ('Average_Loss_Y', each_loss_avg[1]))
                elif self.config.output_dim == 1:
                    f.writelines(formatspecs3 % ('Average_Loss_Speed', each_loss_avg[0]))
                else:
                    print('output_dim outside range.')


def main():

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    config, model, dataset_train, dataset_val, dataset_test = argparser(is_train=False)
    log.warning("dataset path: %s", config.dataset_path)

    evaler_val = Evaler(config, model, dataset_val, 'val')
    evaler_val.eval_run()

    config.batch_size = evaler_val.batch_size

    evaler_train = Evaler(config, model, dataset_train, 'train')
    evaler_train.eval_run()

    config.batch_size = evaler_val.batch_size

    evaler_train = Evaler(config, model, dataset_test, 'test')
    evaler_train.eval_run()

if __name__ == '__main__':
    main()
