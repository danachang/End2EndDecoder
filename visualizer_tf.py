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

class Visualizer(object):

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
            allow_soft_placement=True,
            #gpu_options=tf.GPUOptions(allow_growth=True),
            #device_count={'GPU': 1},
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
        else:
            train_dir_base = os.path.basename(self.train_dir)

        checkpoint_base = os.path.basename(self.checkpoint)

        self.vis_dir = './vis_tf_dir/%s/%s/%s' %(self.config.prefix,
                                              train_dir_base, checkpoint_base)

        if not os.path.exists(self.vis_dir):
            log.infov("create vis_tf_dir: %s", self.vis_dir)
            os.makedirs(self.vis_dir)
        else:
            log.infov("vis_tf_dir exists: %s", self.vis_dir)

    def vis_run(self):
        # load checkpoint
        if self.checkpoint:
            self.saver.restore(self.session, self.checkpoint)
            log.info("Loaded from checkpoint!")

        log.infov("Start 1-epoch Inference and Evaluation")
        log.info("# of examples = %d", len(self.dataset))

        id_list = self.dataset.ids
        id_list = sorted(id_list, key=lambda x: int(x.split('/')[-1].split('.')[0].replace('t', '')))

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

                    """
                    log.infov("featureMap Actv2")
                    self.plot_feature_map(batch_chunk, featurelayer='actv2')

                    log.infov("featureMap Actv3")
                    self.plot_feature_map(batch_chunk, featurelayer='actv3')

                    log.infov("featureMap Actv4")
                    self.plot_feature_map(batch_chunk, featurelayer='actv4')

                    log.infov("featureMap Actv5")
                    self.plot_feature_map(batch_chunk, featurelayer='actv5')

                    log.infov("smoothGrad")
                    start_time = time.time()
                    smoothGrad_both_list, smoothGrad_pos_list, smoothGrad_neg_list = \
                        self.get_smoothGrad_batch(batch_chunk)

                    end_time = time.time()
                    log.infov('layer time: %f', end_time - start_time)
                    self.plot_smoothGrad(smoothGrad_both_list, smoothGrad_pos_list, \
                                        smoothGrad_neg_list, batch_chunk)

                    log.infov("vanillaGrad")
                    start_time = time.time()
                    vanillaGrad_both_list, vanillaGrad_pos_list, vanillaGrad_neg_list = \
                        self.get_vanillaGrad_batch(batch_chunk)

                    end_time = time.time()
                    log.infov('layer time: %f', end_time - start_time)
                    self.plot_vanillaGrad(vanillaGrad_both_list, vanillaGrad_pos_list, \
                                        vanillaGrad_neg_list, batch_chunk)

                    log.infov("grad-CAM output actv4")
                    start_time = time.time()
                    gradcam_small_list, gradcam_pos_list, gradcam_neg_list = \
                        self.get_gradcam_batch(batch_chunk, convlayer='actv4')

                    end_time = time.time()
                    log.infov('layer time: %f', end_time - start_time)
                    self.plot_gradcam(gradcam_small_list, gradcam_pos_list, \
                                        gradcam_neg_list, batch_chunk, \
                                        convlayer='actv4')

                    log.infov("grad-CAM output conv4")
                    start_time = time.time()
                    gradcam_small_lista, gradcam_pos_lista, gradcam_neg_lista = \
                        self.get_gradcam_batch(batch_chunk, convlayer='conv4')

                    end_time = time.time()
                    log.infov('layer time: %f', end_time - start_time)
                    self.plot_gradcam(gradcam_small_lista, gradcam_pos_lista, \
                                        gradcam_neg_lista, batch_chunk, \
                                        convlayer='conv4')

                    log.infov("grad-CAM Actv2")
                    start_time = time.time()
                    gradcam_small_array, gradcam_pos_array, gradcam_neg_array = \
                        self.get_gradcam_layer(batch_chunk, targetlayer='actv2', \
                                                convlayer='conv1')
                    end_time = time.time()
                    log.infov('layer time: %f', end_time - start_time)

                    self.plot_gradcam_layer(gradcam_small_array, gradcam_pos_array, \
                                            gradcam_neg_array, batch_chunk, \
                                            targetlayer='actv2', convlayer='conv1')

                    log.infov("grad-CAM Actv3")
                    start_time = time.time()
                    gradcam_small_array, gradcam_pos_array, gradcam_neg_array = \
                        self.get_gradcam_layer(batch_chunk, targetlayer='actv3', \
                                                convlayer='conv2')
                    end_time = time.time()
                    log.infov('layer time: %f', end_time - start_time)

                    self.plot_gradcam_layer(gradcam_small_array, gradcam_pos_array, \
                                            gradcam_neg_array, batch_chunk, \
                                            targetlayer='actv3', convlayer='conv2')

                    log.infov("grad-CAM Actv4")
                    start_time = time.time()
                    gradcam_small_array, gradcam_pos_array, gradcam_neg_array = \
                        self.get_gradcam_layer(batch_chunk, targetlayer='actv4', \
                                                convlayer='conv3')
                    end_time = time.time()
                    log.infov('layer time: %f', end_time - start_time)

                    self.plot_gradcam_layer(gradcam_small_array, gradcam_pos_array, \
                                            gradcam_neg_array, batch_chunk, \
                                            targetlayer='actv4', convlayer='conv3')
                    """
                    if s >= 3:
                        log.infov("grad-CAM Actv5")
                        start_time = time.time()
                        gradcam_small_array, gradcam_pos_array, gradcam_neg_array = \
                            self.get_gradcam_layer(batch_chunk, targetlayer='actv5', \
                                                    convlayer='conv4')
                        end_time = time.time()
                        log.infov('layer time: %f', end_time - start_time)

                        self.plot_gradcam_layer(gradcam_small_array, gradcam_pos_array, \
                                                gradcam_neg_array, batch_chunk, \
                                                targetlayer='actv5', convlayer='conv4')
                    """
                    log.infov("grad-CAM FC4")
                    start_time = time.time()
                    gradcam_small_array, gradcam_pos_array, gradcam_neg_array = \
                        self.get_gradcam_fc(batch_chunk, targetlayer='fc4', \
                                                convlayer='actv4')

                    end_time = time.time()
                    log.infov('layer time: %f', end_time - start_time)

                    self.plot_gradcam_layer(gradcam_small_array, gradcam_pos_array, \
                                            gradcam_neg_array, batch_chunk, \
                                            targetlayer='fc4', convlayer='actv4')
                    """

                s += 1
                continue_evaluate = (s < len(self.dataset)/self.batch_size)


        except Exception as e:
            print(e)
            log.infov('ohohoh stop')

        log.warning('Visualization completed')

    def normalize_batch(self, batch_array):
        batch_min = batch_array.min(axis=(1, 2), keepdims=True)
        batch_max = batch_array.max(axis=(1, 2), keepdims=True)
        batch_array = (batch_array - batch_min)/(batch_max - batch_min + 1e-7)
        return batch_array

    def divide_max_batch(self, batch_array):
        max_array = batch_array.max(axis=(1, 2, 3), keepdims=True)
        batch_array = batch_array/(max_array + 1e-7)
        return batch_array

    def get_ncol_figsize(self, nfilters):
        if nfilters <= 10:
            ncols = nfilters
        elif nfilters <= 64:
            ncols = 8
        elif nfilters <= 100:
            ncols = 10
        elif nfilters <= 144:
            ncols = 12
        elif nfilters <= 196:
            ncols = 14
        elif nfilters <= 256:
            ncols = 16
        else:
            ncols = 32

        figsize = (ncols*0.6, (nfilters//ncols)*0.6)
        if ncols == 32:
            figsize = (16, nfilters//64)
        if nfilters <= 16:
            figsize = (ncols, 2)

        return ncols, figsize

    def stitch_images(self, images, margin=5, cols=5):
        h, w = images[0].shape
        n_rows = int(np.ceil(images.shape[0]/ cols))
        n_cols = min(images.shape[0], cols)

        out_w = n_cols * w + (n_cols-1) * margin
        out_h = n_rows * h + (n_rows-1) * margin

        stitched_images = np.zeros((out_h, out_w), dtype=images[0].dtype)

        for row in range(n_rows):
            for col in range(n_cols):
                img_idx = row * cols + col
                if img_idx >= images.shape[0]:
                    break

                start_h = (h+margin)*row
                start_w = (w+margin)*col
                stitched_images[start_h: start_h+h, start_w: start_w+w] = \
                    images[img_idx]

        return stitched_images

    def plot_feature_map(self, batch_chunk, featurelayer='actv2'):
        prepath_str = self.vis_dir + '/' + self.dataset_type_str

        if 'conv' in featurelayer:
            if any(featurelayer in v for v in self.model_vars):
                feature_id = int(featurelayer.split('conv')[1])
                feature_layer = self.model.conv_list[feature_id-1]
        elif 'actv' in featurelayer:
            tmp = featurelayer.replace('actv', 'conv')
            if any(tmp in c for c in self.model_vars):
                feature_id = int(featurelayer.split('actv')[1])
                feature_layer = self.model.actv_list[feature_id-1]
        elif 'attn' in featurelayer:
            if any(featurelayer in c for c in self.model_vars):
                feature_id = int(featurelayer.split('attn')[1])
                feature_layer = self.model.attn_list[feature_id-1]
                bs, h, w, _ = self.model.actv_list[feature_id-1].get_shape().as_list()
                feature_layer = tf.reshape(feature_layer, (bs, h, w, -1))

        [features] = self.session.run([feature_layer],
            feed_dict=self.model.get_feed_dict(batch_chunk)
        )

        n_items, h, w, n_filters = features.shape
        ncols, figsize = self.get_ncol_figsize(n_filters)

        factor_h = (self.config.h*1.0)/h
        factor_w = (self.config.w*1.0)/w

        features = self.normalize_batch(features)
        features = np.reshape(features, (n_items, n_filters, h, w))
        features = ndi.zoom(features, (1, 1, factor_h, factor_w), order=1)

        for item in range(n_items):
            id = batch_chunk['id'][item]
            label = batch_chunk['label'][item]
            figpath = prepath_str + '_featureMap_' + featurelayer + '_' + id + '.svg'

            feature_map = self.stitch_images(features[item, ...], cols=ncols)

            plt.figure(figsize=figsize)
            plt.axis('off')
            plt.imshow(feature_map, cmap='jet')
            plt.title('('+ self.dataset_type_str + ') ' + id + ' feature Map on ' +
                        featurelayer + ' layer \n (true: ' + str(label) + '])')
            plt.tight_layout(rect=[0, 0.03, 1, 0.9])
            plt.savefig(figpath)
            plt.close()

    def get_gradcam_layer(self, batch_chunk, targetlayer='conv2', \
                            convlayer='conv1'):

        if 'conv' in targetlayer:
            if any(targetlayer in v for v in self.model_vars):
                target_id = int(targetlayer.split('conv')[1])
                target_layer = self.model.conv_list[target_id-1]
        elif 'actv' in targetlayer:
            tmp = targetlayer.replace('actv', 'conv')
            if any(tmp in c for c in self.model_vars):
                target_id = int(targetlayer.split('actv')[1])
                target_layer = self.model.actv_list[target_id-1]
        else:
            return None, None, None

        if any(convlayer in v for v in self.model_vars):
            conv_id = int(convlayer.split('conv')[1])
            conv_layer = self.model.conv_list[conv_id-1]
        else:
            return None, None, None

        filter_dim = target_layer.shape[3]
        n_items = target_layer.shape[0]

        factor_h = (self.config.h*1.0)/conv_layer.shape[1].value
        factor_w = (self.config.w*1.0)/conv_layer.shape[2].value

        start_id = batch_chunk['id'][0]
        end_id = batch_chunk['id'][-1]

        gradcam_array_shape = (filter_dim, n_items, self.config.h, self.config.w)
        gradcam_small_array = np.zeros(gradcam_array_shape, dtype=np.float32)
        gradcam_pos_array = np.zeros(gradcam_array_shape, dtype=np.float32)
        gradcam_neg_array = np.zeros(gradcam_array_shape, dtype=np.float32)

        #gradcam_small_array_noNorm = gradcam_small_array
        #gradcam_pos_array_noNorm = gradcam_pos_array
        #gradcam_neg_array_noNorm = gradcam_neg_array

        prepath_str = self.vis_dir + '/' + self.dataset_type_str
        npfile_pos = prepath_str + '_' + targetlayer + '_gradCAM_' + \
                        convlayer + '_' + start_id + 'to' + end_id + '_increase.npz'
        npfile_neg = prepath_str + '_' + targetlayer + '_gradCAM_' + \
                        convlayer + '_' + start_id + 'to' + end_id + '_decrease.npz'
        npfile_small = prepath_str + '_' + targetlayer + '_gradCAM_' + \
                        convlayer + '_' + start_id + 'to' + end_id + '_maintain.npz'

        for dim in range(filter_dim):
            grads = tf.gradients(target_layer[..., dim], conv_layer)
            pos_grads = grads[0]
            neg_grads = -grads[0]
            inverse_grads = tf.div(tf.constant(1.), grads[0] + tf.constant(1e-7))
            small_grads = tf.abs(inverse_grads)

            [grads_small, grads_pos, grads_neg, output_conv] = self.session.run(
                [small_grads, pos_grads, neg_grads, conv_layer],
                feed_dict=self.model.get_feed_dict(batch_chunk)
            )

            grads_pos = self.divide_max_batch(grads_pos)
            grads_neg = self.divide_max_batch(grads_neg)
            grads_small = self.divide_max_batch(grads_small)

            w_pos = np.mean(grads_pos, axis=(1, 2))
            w_neg = np.mean(grads_neg, axis=(1, 2))
            w_small = np.mean(grads_small, axis=(1, 2))

            gradcam_pos = np.zeros(output_conv.shape[0:3], dtype=np.float32)
            gradcam_neg = np.zeros(output_conv.shape[0:3], dtype=np.float32)
            gradcam_small = np.zeros(output_conv.shape[0:3], dtype=np.float32)

            for ch, (wp, wn, ws) in enumerate(zip(w_pos, w_neg, w_small)):
                gradcam_pos[ch, ...] += np.sum(wp * output_conv[ch, ...], axis=2)
                gradcam_neg[ch, ...] += np.sum(wn * output_conv[ch, ...], axis=2)
                gradcam_small[ch, ...] += np.sum(ws * output_conv[ch, ...], axis=2)

            gradcam_pos = np.maximum(gradcam_pos, 0)
            gradcam_neg = np.maximum(gradcam_neg, 0)
            gradcam_small = np.maximum(gradcam_small, 0)

            gradcam_pos = ndi.zoom(gradcam_pos, (1, factor_h, factor_w), order=1)
            gradcam_neg = ndi.zoom(gradcam_neg, (1, factor_h, factor_w), order=1)
            gradcam_small = ndi.zoom(gradcam_small, (1, factor_h, factor_w), order=1)

            #gradcam_small_array_noNorm[dim, ...] = gradcam_small
            #gradcam_pos_array_noNorm[dim, ...] = gradcam_pos
            #gradcam_neg_array_noNorm[dim, ...] = gradcam_neg

            gradcam_small = self.normalize_batch(gradcam_small)
            gradcam_pos = self.normalize_batch(gradcam_pos)
            gradcam_neg = self.normalize_batch(gradcam_neg)

            gradcam_small_array[dim, ...] = gradcam_small
            gradcam_pos_array[dim, ...] = gradcam_pos
            gradcam_neg_array[dim, ...] = gradcam_neg

        #np.savez(npfile_small, id=batch_chunk['id'], gradcam=gradcam_small_array_noNorm)
        #np.savez(npfile_pos, id=batch_chunk['id'], gradcam=gradcam_pos_array_noNorm)
        #np.savez(npfile_neg, id=batch_chunk['id'], gradcam=gradcam_neg_array_noNorm)
        del grads, pos_grads, neg_grads, inverse_grads, small_grads
        del gradcam_small, gradcam_neg, gradcam_pos, conv_layer, target_layer
        return gradcam_small_array, gradcam_pos_array, gradcam_neg_array

    def plot_gradcam_layer(self, gradcam_small_array, gradcam_pos_array, \
        gradcam_neg_array, batch_chunk, targetlayer='actv4', convlayer='conv3'):

        if gradcam_pos_array is None:
            return

        prepath_str = self.vis_dir + '/' + self.dataset_type_str
        n_filters = gradcam_small_array.shape[0]
        n_items = gradcam_small_array.shape[1]

        ncols, figsize = self.get_ncol_figsize(n_filters)

        for item in range(n_items):
            id = batch_chunk['id'][item]
            label = batch_chunk['label'][item]
            figpath_pos = prepath_str + '_' + targetlayer + '_gradCAM_' + \
                            convlayer + '_' + id + '_increase.svg'
            figpath_neg = prepath_str + '_' + targetlayer + '_gradCAM_' + \
                            convlayer + '_' + id + '_decrease.svg'
            figpath_small = prepath_str + '_' + targetlayer + '_gradCAM_' + \
                            convlayer + '_' + id + '_maintain.svg'

            fig_pos = self.stitch_images(gradcam_pos_array[:, item, :, :],
                                            cols=ncols)
            fig_neg = self.stitch_images(gradcam_neg_array[:, item, :, :],
                                            cols=ncols)
            fig_small = self.stitch_images(gradcam_small_array[:, item, :, :],
                                            cols=ncols)

            plt.figure(figsize=figsize)
            plt.axis('off')
            plt.imshow(fig_pos, cmap='jet')
            plt.title('('+ self.dataset_type_str + ') ' + id + ' grad-CAM Map on ' +
                        targetlayer + ' layer [increase] \n (true: ' + str(label) +
                        '])')
            plt.tight_layout(rect=[0, 0.03, 1, 0.9])
            plt.savefig(figpath_pos)
            plt.close()

            plt.figure(figsize=figsize)
            plt.axis('off')
            plt.imshow(fig_neg, cmap='jet')
            plt.title('('+ self.dataset_type_str + ') ' + id + ' grad-CAM Map on ' +
                        targetlayer + ' layer [decrease] \n (true: ' + str(label) +
                        '])')
            plt.tight_layout(rect=[0, 0.03, 1, 0.9])
            plt.savefig(figpath_neg)
            plt.close()

            plt.figure(figsize=figsize)
            plt.axis('off')
            plt.imshow(fig_small, cmap='jet')
            plt.title('('+ self.dataset_type_str + ') ' + id + ' grad-CAM Map on ' +
                        targetlayer + ' layer [maintain] \n (true: ' + str(label) +
                        '])')
            plt.tight_layout(rect=[0, 0.03, 1, 0.9])
            plt.savefig(figpath_small)
            plt.close()

    def get_gradcam_fc(self, batch_chunk, targetlayer='fc4', \
                            convlayer='last'):

        if 'fc' in targetlayer:
            if any(targetlayer in v for v in self.model_vars):
                target_id = int(targetlayer.split('fc')[1])
                target_layer = self.model.fc_list[target_id-1]
        else:
            return None, None, None

        if any(convlayer in v for v in self.model_vars):
            conv_id = int(convlayer.split('conv')[1])
            conv_layer = self.model.conv_list[conv_id-1]
        elif 'actv' in convlayer:
            tmp = convlayer.replace('actv', 'conv')
            if any(tmp in c for c in self.model_vars):
                actv_id = int(convlayer.split('actv')[1])
                conv_layer = self.model.actv_list[actv_id-1]
        else:
            conv_layer = self.model.actv_list[-1]

        prepath_str = self.vis_dir + '/' + self.dataset_type_str
        filter_dim = target_layer.shape[1]
        n_items = target_layer.shape[0]

        factor_h = (self.config.h*1.0)/conv_layer.shape[1].value
        factor_w = (self.config.w*1.0)/conv_layer.shape[2].value

        start_id = batch_chunk['id'][0]
        end_id = batch_chunk['id'][-1]

        gradcam_array_shape = (filter_dim, n_items, self.config.h, self.config.w)
        gradcam_small_array = np.zeros(gradcam_array_shape, dtype=np.float32)
        gradcam_pos_array = np.zeros(gradcam_array_shape, dtype=np.float32)
        gradcam_neg_array = np.zeros(gradcam_array_shape, dtype=np.float32)

        for dim in range(filter_dim):
            grads = tf.gradients(target_layer[..., dim], conv_layer)
            pos_grads = grads[0]
            neg_grads = -grads[0]
            inverse_grads = tf.div(tf.constant(1.), grads[0] + tf.constant(1e-7))
            small_grads = tf.abs(inverse_grads)

            [grads_small, grads_pos, grads_neg, output_conv] = self.session.run(
                [small_grads, pos_grads, neg_grads, conv_layer],
                feed_dict=self.model.get_feed_dict(batch_chunk)
            )

            grads_pos = self.divide_max_batch(grads_pos)
            grads_neg = self.divide_max_batch(grads_neg)
            grads_small = self.divide_max_batch(grads_small)

            w_pos = np.mean(grads_pos, axis=(1, 2))
            w_neg = np.mean(grads_neg, axis=(1, 2))
            w_small = np.mean(grads_small, axis=(1, 2))

            gradcam_pos = np.zeros(output_conv.shape[0:3], dtype=np.float32)
            gradcam_neg = np.zeros(output_conv.shape[0:3], dtype=np.float32)
            gradcam_small = np.zeros(output_conv.shape[0:3], dtype=np.float32)

            for ch, (wp, wn, ws) in enumerate(zip(w_pos, w_neg, w_small)):
                gradcam_pos[ch, ...] += np.sum(wp * output_conv[ch, ...], axis=2)
                gradcam_neg[ch, ...] += np.sum(wn * output_conv[ch, ...], axis=2)
                gradcam_small[ch, ...] += np.sum(ws * output_conv[ch, ...], axis=2)

            gradcam_pos = np.maximum(gradcam_pos, 0)
            gradcam_neg = np.maximum(gradcam_neg, 0)
            gradcam_small = np.maximum(gradcam_small, 0)

            gradcam_pos = ndi.zoom(gradcam_pos, (1, factor_h, factor_w), order=1)
            gradcam_neg = ndi.zoom(gradcam_neg, (1, factor_h, factor_w), order=1)
            gradcam_small = ndi.zoom(gradcam_small, (1, factor_h, factor_w), order=1)

            gradcam_small = self.normalize_batch(gradcam_small)
            gradcam_pos = self.normalize_batch(gradcam_pos)
            gradcam_neg = self.normalize_batch(gradcam_neg)

            gradcam_small_array[dim, ...] = gradcam_small
            gradcam_pos_array[dim, ...] = gradcam_pos
            gradcam_neg_array[dim, ...] = gradcam_neg

        del grads, pos_grads, neg_grads, inverse_grads, small_grads
        del gradcam_small, gradcam_neg, gradcam_pos, conv_layer, target_layer
        return gradcam_small_array, gradcam_pos_array, gradcam_neg_array

    def get_gradcam_batch(self, batch_chunk, convlayer='last'):
        prepath_str = self.vis_dir + '/' + self.dataset_type_str
        out_dim = self.model.pred_label.shape[1]
        gradcam_small_list = []
        gradcam_pos_list = []
        gradcam_neg_list = []

        if any(convlayer in v for v in self.model_vars):
            conv_id = int(convlayer.split('conv')[1])
            conv_layer = self.model.conv_list[conv_id-1]
        elif 'actv' in convlayer:
            tmp = convlayer.replace('actv', 'conv')
            if any(tmp in c for c in self.model_vars):
                actv_id = int(convlayer.split('actv')[1])
                conv_layer = self.model.actv_list[actv_id-1]
        else:
            conv_layer = self.model.actv_list[-1]

        factor_h = (self.config.h*1.0)/conv_layer.shape[1].value
        factor_w = (self.config.w*1.0)/conv_layer.shape[2].value

        start_id = batch_chunk['id'][0]
        end_id = batch_chunk['id'][-1]

        if out_dim == 3:
            label_vars = ['x', 'y', 'speed']
        elif out_dim == 2:
            label_vars = ['x', 'y']
        elif out_dim == 1:
            label_vars = ['speed']

        for dim in range(out_dim):
            grads = tf.gradients(self.model.pred_label[..., dim], conv_layer)
            pos_grads = grads[0]
            neg_grads = -grads[0]
            inverse_grads = tf.div(tf.constant(1.), grads[0] + tf.constant(1e-7))
            small_grads = tf.abs(inverse_grads)

            [grads_small, grads_pos, grads_neg, output_conv] = self.session.run(
                [small_grads, pos_grads, neg_grads, conv_layer],
                feed_dict=self.model.get_feed_dict(batch_chunk)
            )

            grads_pos = self.divide_max_batch(grads_pos)
            grads_neg = self.divide_max_batch(grads_neg)
            grads_small = self.divide_max_batch(grads_small)

            w_pos = np.mean(grads_pos, axis=(1, 2))
            w_neg = np.mean(grads_neg, axis=(1, 2))
            w_small = np.mean(grads_small, axis=(1, 2))

            gradcam_pos = np.zeros(output_conv.shape[0:3], dtype=np.float32)
            gradcam_neg = np.zeros(output_conv.shape[0:3], dtype=np.float32)
            gradcam_small = np.zeros(output_conv.shape[0:3], dtype=np.float32)

            npfile_pos = prepath_str + '_gradCAM_' + convlayer + '_' + \
                            start_id + 'to' + end_id + '_' + \
                            label_vars[dim] + '_increase.npz'
            npfile_neg = prepath_str + '_gradCAM_' + convlayer + '_' + \
                            start_id + 'to' + end_id + '_' + \
                            label_vars[dim] + '_decrease.npz'
            npfile_small = prepath_str + '_gradCAM_' + convlayer + '_' + \
                            start_id + 'to' + end_id + '_' + \
                            label_vars[dim] + '_maintain.npz'

            for ch, (wp, wn, ws) in enumerate(zip(w_pos, w_neg, w_small)):
                gradcam_pos[ch, ...] += np.sum(wp * output_conv[ch, ...], axis=2)
                gradcam_neg[ch, ...] += np.sum(wn * output_conv[ch, ...], axis=2)
                gradcam_small[ch, ...] += np.sum(ws * output_conv[ch, ...], axis=2)

            gradcam_pos = np.maximum(gradcam_pos, 0)
            gradcam_neg = np.maximum(gradcam_neg, 0)
            gradcam_small = np.maximum(gradcam_small, 0)

            gradcam_pos = ndi.zoom(gradcam_pos, (1, factor_h, factor_w), order=1)
            gradcam_neg = ndi.zoom(gradcam_neg, (1, factor_h, factor_w), order=1)
            gradcam_small = ndi.zoom(gradcam_small, (1, factor_h, factor_w), order=1)

            np.savez(npfile_small, id=batch_chunk['id'], gradcam=gradcam_small)
            np.savez(npfile_pos, id=batch_chunk['id'], gradcam=gradcam_pos)
            np.savez(npfile_neg, id=batch_chunk['id'], gradcam=gradcam_neg)

            gradcam_small = self.normalize_batch(gradcam_small)
            gradcam_pos = self.normalize_batch(gradcam_pos)
            gradcam_neg = self.normalize_batch(gradcam_neg)

            gradcam_small_list.append(gradcam_small)
            gradcam_pos_list.append(gradcam_pos)
            gradcam_neg_list.append(gradcam_neg)

        return gradcam_small_list, gradcam_pos_list, gradcam_neg_list

    def plot_gradcam(self, gradcam_small_list, gradcam_pos_list, \
                        gradcam_neg_list, batch_chunk, convlayer='last'):

        prepath_str = self.vis_dir + '/' + self.dataset_type_str
        steer_output = ['image', 'smallgrad', 'positive', 'negative']
        n_items = batch_chunk['image'].shape[0]
        out_dim = batch_chunk['label'].shape[1]
        ncol = len(steer_output)

        layer_idx = -1
        if out_dim == 3:
            label_vars = ['x', 'y', 'speed']
        elif out_dim == 2:
            label_vars = ['x', 'y']
        elif out_dim == 1:
            label_vars = ['speed']

        figsize=(7, 2.1*out_dim)
        for item in range(n_items):
            fig = plt.subplots(out_dim, ncol, figsize=figsize)
            id = batch_chunk['id'][item]
            label = batch_chunk['label'][item]
            figpath = prepath_str + '_gradCAM_' + convlayer + '_' + id + '.svg'
            for label_dim in range(out_dim):
                for num in range(ncol):
                    plt.subplot(out_dim, ncol, num+1+label_dim*ncol)
                    ax = plt.gca()
                    if num == 0:
                        plt.imshow(batch_chunk['image'][item][..., 0], cmap='gray')
                        plt.ylabel(label_vars[label_dim])
                    elif num == 1:
                        plt.imshow(gradcam_small_list[label_dim][item, ...], cmap='jet')
                    elif num == 2:
                        plt.imshow(gradcam_pos_list[label_dim][item, ...], cmap='jet')
                    elif num == 3:
                        plt.imshow(gradcam_neg_list[label_dim][item, ...], cmap='jet')

                    ax.set_title(steer_output[num])
                    ax.axes.xaxis.set_ticks([])
                    ax.axes.yaxis.set_ticks([])
                    ax.set_aspect('equal')

            plt.suptitle('('+ self.dataset_type_str + ') ' + id +
                        ' grad-CAM Map \n (true labels: ' + str(label) + ')')
            plt.tight_layout(rect=[0, 0.03, 1, 0.9])
            plt.savefig(figpath)
            plt.close()

    def get_smoothGrad_batch(self, batch_chunk, stdev_spread=.2, nsamples=100):
        prepath_str = self.vis_dir + '/' + self.dataset_type_str
        out_dim = self.model.pred_label.shape[1]
        smoothGrad_both_list = []
        smoothGrad_pos_list = []
        smoothGrad_neg_list = []
        image_shape = batch_chunk['image'].shape[1:]

        start_id = batch_chunk['id'][0]
        end_id = batch_chunk['id'][-1]

        if out_dim == 3:
            label_vars = ['x', 'y', 'speed']
        elif out_dim == 2:
            label_vars = ['x', 'y']
        elif out_dim == 1:
            label_vars = ['speed']

        for dim in range(out_dim):
            grads = tf.gradients(self.model.pred_label[..., dim], self.model.image)
            abs_smoothGrad = tf.abs(grads[0])
            pos_smoothGrad = tf.nn.relu(grads[0])
            neg_smoothGrad = tf.nn.relu(-grads[0])

            stdev = stdev_spread * (np.max(batch_chunk['image'], axis=(1, 2)) - \
                                      np.min(batch_chunk['image'], axis=(1, 2)))

            smoothGrad_both = np.zeros_like(batch_chunk['image'])
            smoothGrad_pos = np.zeros_like(batch_chunk['image'])
            smoothGrad_neg = np.zeros_like(batch_chunk['image'])

            npfile_both = prepath_str + '_smoothGrad_' + \
                            start_id + 'to' + end_id + '_' + \
                            label_vars[dim] + '_absolute.npz'
            npfile_pos = prepath_str + '_smoothGrad_' + \
                            start_id + 'to' + end_id + '_' + \
                            label_vars[dim] + '_increase.npz'
            npfile_neg = prepath_str + '_smoothGrad_' + \
                            start_id + 'to' + end_id + '_' + \
                            label_vars[dim] + '_decrease.npz'

            for i in range(nsamples):
                noise = np.array([np.random.normal(0, s, image_shape) for s in stdev])
                image_add_noise = batch_chunk['image'] + noise
                batch_chunk_i = batch_chunk.copy()
                batch_chunk_i['image'] = image_add_noise

                [smoothGrad_both_i, smoothGrad_pos_i, smoothGrad_neg_i] = self.session.run(
                    [abs_smoothGrad, pos_smoothGrad, neg_smoothGrad],
                    feed_dict=self.model.get_feed_dict(batch_chunk_i)
                )

                smoothGrad_both += smoothGrad_both_i * smoothGrad_both_i
                smoothGrad_pos += smoothGrad_pos_i * smoothGrad_pos_i
                smoothGrad_neg += smoothGrad_neg_i * smoothGrad_neg_i

            smoothGrad_both = smoothGrad_both / nsamples
            smoothGrad_pos = smoothGrad_pos / nsamples
            smoothGrad_neg = smoothGrad_neg / nsamples

            np.savez(npfile_both, id=batch_chunk['id'], gradcam=smoothGrad_both)
            np.savez(npfile_pos, id=batch_chunk['id'], gradcam=smoothGrad_pos)
            np.savez(npfile_neg, id=batch_chunk['id'], gradcam=smoothGrad_neg)

            smoothGrad_both = self.normalize_batch(smoothGrad_both[..., 0])
            smoothGrad_pos = self.normalize_batch(smoothGrad_pos[..., 0])
            smoothGrad_neg = self.normalize_batch(smoothGrad_neg[..., 0])

            smoothGrad_both_list.append(smoothGrad_both)
            smoothGrad_pos_list.append(smoothGrad_pos)
            smoothGrad_neg_list.append(smoothGrad_neg)

        return smoothGrad_both_list, smoothGrad_pos_list, smoothGrad_neg_list

    def plot_smoothGrad(self, smoothGrad_both_list, smoothGrad_pos_list, \
                        smoothGrad_neg_list, batch_chunk):

        prepath_str = self.vis_dir + '/' + self.dataset_type_str
        steer_output = ['image', 'absolute', 'positive', 'negative']
        n_items = batch_chunk['image'].shape[0]
        out_dim = batch_chunk['label'].shape[1]
        ncol = len(steer_output)

        layer_idx = -1
        if out_dim == 3:
            label_vars = ['x', 'y', 'speed']
        elif out_dim == 2:
            label_vars = ['x', 'y']
        elif out_dim == 1:
            label_vars = ['speed']

        figsize=(7, 2.1*out_dim)
        for item in range(n_items):
            fig = plt.subplots(out_dim, ncol, figsize=figsize)
            id = batch_chunk['id'][item]
            label = batch_chunk['label'][item]
            figpath = prepath_str + '_smoothGrad_' + id + '.svg'
            for label_dim in range(out_dim):
                for num in range(ncol):
                    plt.subplot(out_dim, ncol, num+1+label_dim*ncol)
                    ax = plt.gca()
                    if num == 0:
                        plt.imshow(batch_chunk['image'][item][..., 0], cmap='gray')
                        plt.ylabel(label_vars[label_dim])
                    elif num == 1:
                        plt.imshow(smoothGrad_both_list[label_dim][item, ...], cmap='jet')
                    elif num == 2:
                        plt.imshow(smoothGrad_pos_list[label_dim][item, ...], cmap='jet')
                    elif num == 3:
                        plt.imshow(smoothGrad_neg_list[label_dim][item, ...], cmap='jet')

                    ax.set_title(steer_output[num])
                    ax.axes.xaxis.set_ticks([])
                    ax.axes.yaxis.set_ticks([])
                    ax.set_aspect('equal')

            plt.suptitle('('+ self.dataset_type_str + ') ' + id +
                        ' smoothGrad Map \n (true labels: ' + str(label) + ')')
            plt.tight_layout(rect=[0, 0.03, 1, 0.9])
            plt.savefig(figpath)
            plt.close()

    def get_vanillaGrad_batch(self, batch_chunk):
        out_dim = self.model.pred_label.shape[1]
        vanillaGrad_both_list = []
        vanillaGrad_pos_list = []
        vanillaGrad_neg_list = []

        for dim in range(out_dim):
            grads = tf.gradients(self.model.pred_label[..., dim], self.model.image)
            abs_vanillaGrad = tf.abs(grads[0])
            abs_vanillaGrad = tf.reduce_max(abs_vanillaGrad, axis=3)
            pos_vanillaGrad = tf.nn.relu(grads[0])
            pos_vanillaGrad = tf.reduce_max(pos_vanillaGrad, axis=3)
            neg_vanillaGrad = tf.nn.relu(-grads[0])
            neg_vanillaGrad = tf.reduce_max(neg_vanillaGrad, axis=3)

            [vanillaGrad_both, vanillaGrad_pos, vanillaGrad_neg] = self.session.run(
                [abs_vanillaGrad, pos_vanillaGrad, neg_vanillaGrad],
                feed_dict=self.model.get_feed_dict(batch_chunk)
            )

            vanillaGrad_both = self.normalize_batch(vanillaGrad_both)
            vanillaGrad_pos = self.normalize_batch(vanillaGrad_pos)
            vanillaGrad_neg = self.normalize_batch(vanillaGrad_neg)

            vanillaGrad_both_list.append(vanillaGrad_both)
            vanillaGrad_pos_list.append(vanillaGrad_pos)
            vanillaGrad_neg_list.append(vanillaGrad_neg)

        return vanillaGrad_both_list, vanillaGrad_pos_list, vanillaGrad_neg_list

    def plot_vanillaGrad(self, vanillaGrad_both_list, vanillaGrad_pos_list, \
                        vanillaGrad_neg_list, batch_chunk):

        prepath_str = self.vis_dir + '/' + self.dataset_type_str
        steer_output = ['image', 'absolute', 'positive', 'negative']
        n_items = batch_chunk['image'].shape[0]
        out_dim = batch_chunk['label'].shape[1]
        ncol = len(steer_output)

        layer_idx = -1
        if out_dim == 3:
            label_vars = ['x', 'y', 'speed']
        elif out_dim == 2:
            label_vars = ['x', 'y']
        elif out_dim == 1:
            label_vars = ['speed']

        figsize=(7, 2.1*out_dim)
        for item in range(n_items):
            fig = plt.subplots(out_dim, ncol, figsize=figsize)
            id = batch_chunk['id'][item]
            label = batch_chunk['label'][item]
            figpath = prepath_str + '_vanillaGrad_' + id + '.svg'
            for label_dim in range(out_dim):
                for num in range(ncol):
                    plt.subplot(out_dim, ncol, num+1+label_dim*ncol)
                    ax = plt.gca()
                    if num == 0:
                        plt.imshow(batch_chunk['image'][item][..., 0], cmap='gray')
                        plt.ylabel(label_vars[label_dim])
                    elif num == 1:
                        plt.imshow(vanillaGrad_both_list[label_dim][item, ...], cmap='jet')
                    elif num == 2:
                        plt.imshow(vanillaGrad_pos_list[label_dim][item, ...], cmap='jet')
                    elif num == 3:
                        plt.imshow(vanillaGrad_neg_list[label_dim][item, ...], cmap='jet')

                    ax.set_title(steer_output[num])
                    ax.axes.xaxis.set_ticks([])
                    ax.axes.yaxis.set_ticks([])
                    ax.set_aspect('equal')

            plt.suptitle('('+ self.dataset_type_str + ') ' + id +
                        ' vanillaGrad Map \n (true labels: ' + str(label) + ')')
            plt.tight_layout(rect=[0, 0.03, 1, 0.9])
            plt.savefig(figpath)
            plt.close()


def main():

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    config, model, dataset_train, dataset_val, dataset_test = argparser(is_train=False)
    log.warning("dataset path: %s", config.dataset_path)

    #viewer_val = Visualizer(config, model, dataset_val, 'val')
    #viewer_val.vis_run()
    #config.batch_size = viewer_val.batch_size

    #viewer_train = Visualizer(config, model, dataset_train, 'train')
    #viewer_train.vis_run()
    #config.batch_size = viewer_train.batch_size

    viewer_test = Visualizer(config, model, dataset_test, 'test')
    viewer_test.vis_run()

if __name__ == '__main__':
    main()
