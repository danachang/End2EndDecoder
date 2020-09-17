import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.slim as slim
import numpy as np
from util import log


def print_info(name, shape=None, activation_fn=None):
    if shape is not None:
        log.info('{}{} {}'.format(
            name,  '' if activation_fn is None else ' ('+activation_fn.__name__+')',
            shape))
    else:
        log.info('{}'.format(name))


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * tf.where(x > 0.0, x, alpha * tf.exp(x) - alpha)


def act_str2fn(act_str):
    if act_str == 'tanh':
        return tf.tanh
    elif act_str == 'sigmoid':
        return tf.sigmoid
    elif act_str == 'linear':
        return None
    elif act_str == 'relu':
        return tf.nn.relu
    else:
        raise NotImplementedError


def _l2normalize(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def spectral_normed_weight(weights, num_iters=1, update_collection=None,
                           with_sigma=False):
    """
    Performs Spectral Normalization on a weight tensor.
    """
    w_shape = weights.shape.as_list()
    w_mat = tf.reshape(weights, [-1, w_shape[-1]])  # [-1, output_channel]
    u = tf.get_variable('u', [1, w_shape[-1]],
                        initializer=tf.truncated_normal_initializer(),
                        trainable=False)
    u_ = u
    for _ in range(num_iters):
        v_ = _l2normalize(tf.matmul(u_, w_mat, transpose_b=True))
        u_ = _l2normalize(tf.matmul(v_, w_mat))

    sigma = tf.squeeze(tf.matmul(tf.matmul(v_, w_mat), u_, transpose_b=True))
    w_mat /= sigma
    if update_collection is None:
        with tf.control_dependencies([u.assign(u_)]):
            w_bar = tf.reshape(w_mat, w_shape)
    else:
        w_bar = tf.reshape(w_mat, w_shape)
        if update_collection != 'NO_OPS':
            tf.add_to_collection(update_collection, u.assign(u_))

    if with_sigma:
        return w_bar, sigma
    else:
        return w_bar


def huber_loss(labels, predictions, delta=1.0):
    residual = tf.abs(predictions - labels)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    return tf.where(condition, small_res, large_res)


def flatten(input, name='flatten', info=False):
    output = tf.reshape(input, [input.get_shape()[0], -1])
    if info: print_info(name, output.get_shape())
    return output


def instance_norm(input):
    """
    Instance normalization
    """
    with tf.variable_scope('instance_norm'):
        num_out = input.get_shape()[-1]
        scale = tf.get_variable(
            'scale', [num_out],
            initializer=tf.random_normal_initializer(mean=1.0, stddev=0.02))
        offset = tf.get_variable(
            'offset', [num_out],
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
        mean, var = tf.nn.moments(input, axes=[1, 2], keep_dims=True)
        epsilon = 1e-6
        inv = tf.rsqrt(var + epsilon)
        return scale * (input - mean) * inv + offset


def norm_and_act(input, is_train, norm='batch', activation_fn=None, name="bn_act"):
    """
    Apply normalization and/or activation function
    """
    with tf.variable_scope(name):
        _ = input
        if activation_fn is not None:
            _ = activation_fn(_)
        if norm is not None and norm is not False:
            if norm == 'batch':
                _ = tf.contrib.layers.batch_norm(
                    _, center=True, scale=True,
                    updates_collections=None,
                    is_training=is_train,
                )
                #print(is_train)
            elif norm == 'instance':
                _ = instance_norm(_, is_train)
            elif norm == 'none':
                _ = _
            else:
                raise NotImplementedError
    return _


def conv2d(input, output_shape, is_train, info=False, k=3, s=2, stddev=0.01,
           activation_fn=lrelu, norm='batch', name="conv2d"):
    with tf.variable_scope(name):
        pre_act = slim.conv2d(input, output_shape, [k, k], stride=s, activation_fn=None)
        _ = norm_and_act(pre_act, is_train, norm=norm, activation_fn=activation_fn)
        if info: print_info(name, _.get_shape().as_list(), activation_fn)
    return _, pre_act

def conv2d_even(input, output_shape, is_train, info=False, k=4, s=2, stddev=0.01,
           activation_fn=lrelu, norm='batch', name="conv2d"):
    with tf.variable_scope(name):
        _ = slim.conv2d(input, output_shape, [k, k], stride=s, activation_fn=None)
        _ = norm_and_act(_, is_train, norm=norm, activation_fn=activation_fn)
        if info: print_info(name, _.get_shape().as_list(), activation_fn)
    return _

def conv3d(input, output_shape, is_train, info=False, k=3, s=2, stddev=0.01,
           activation_fn=lrelu, norm='batch', name="conv3d"):
    with tf.variable_scope(name):
        _ = slim.conv3d(input, output_shape, [k, k, k], stride=2, activation_fn=None)
        _ = norm_and_act(_, is_train, norm=norm, activation_fn=activation_fn)
        if info: print_info(name, _.get_shape().as_list(), activation_fn)
    return _

def conv3d_noTemp(input, output_shape, is_train, info=False, k=3, s=2, stddev=0.01,
           activation_fn=lrelu, norm='batch', name="conv3d_noTemp"):
    with tf.variable_scope(name):
        _ = slim.conv3d(input, output_shape, [1, k, k], stride=[1, 2, 2], activation_fn=None)
        _ = norm_and_act(_, is_train, norm=norm, activation_fn=activation_fn)
        if info: print_info(name, _.get_shape().as_list(), activation_fn)
    return _

def maxpool2d(input, info=False, k=3, s=2, p='same', dformat='NHWC',
              name="maxpool2d"):
    with tf.variable_scope(name):
        _ = slim.max_pool2d(input, [k, k], stride=s, padding=p, data_format=dformat)
        if info: print_info(name, _.get_shape().as_list())
    return _


def conv2d_res(input, is_train, info=False, k=3, stddev=0.01, name="conv2d_res"):
    with tf.variable_scope(name):
        if info: print_info(name)
        bs, h, w, ch = input.shape.as_list()
        # BN -> ReLU -> conv -> BN -> ReLU -> conv
        x = tf.contrib.layers.batch_norm(input, center=True, scale=True,
                                         updates_collections=None)
        x = tf.nn.relu(x)
        x = conv2d(x, ch, is_train, info=info,
                   k=k, s=1, stddev=stddev, activation_fn=None,
                   norm='none', name='conv1')

        x = tf.contrib.layers.batch_norm(x, center=True, scale=True,
                                         updates_collections=None)
        x = tf.nn.relu(x)
        x = conv2d(x, ch, is_train, info=info,
                   k=k, s=1, stddev=stddev, activation_fn=None,
                   norm='none', name='conv2')
        out = input + x
    return out


def deconv2d(input, output_shape, is_train, info=False, k=3, s=2, stddev=0.01,
             activation_fn=tf.nn.relu, norm='batch', name='deconv2d'):
    with tf.variable_scope(name):
        _ = layers.conv2d_transpose(
            input,
            num_outputs=output_shape,
            weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
            biases_initializer=tf.zeros_initializer(),
            activation_fn=None,
            kernel_size=[k, k], stride=[s, s], padding='SAME'
        )
        _ = norm_and_act(_, is_train, norm=norm, activation_fn=activation_fn)
        if info: print_info(name, _.get_shape().as_list(), activation_fn)
    return _


def bilinear_deconv2d(input, output_shape, is_train, info=False, k=3, s=2, stddev=0.01,
                      activation_fn=tf.nn.relu, norm='batch', name='deconv2d'):
    with tf.variable_scope(name):
        h = int(input.get_shape()[1]) * s
        w = int(input.get_shape()[2]) * s
        _ = tf.image.resize_bilinear(input, [h, w])
        _ = conv2d(_, output_shape, is_train, k=k, s=1,
                   norm=False, activation_fn=None)
        _ = norm_and_act(_, is_train, norm=norm, activation_fn=activation_fn)
        if info: print_info(name, _.get_shape().as_list(), activation_fn)
    return _


def nn_deconv2d(input, output_shape, is_train, info=False, k=3, s=2, stddev=0.01,
                activation_fn=tf.nn.relu, norm='batch', name='deconv2d'):
    with tf.variable_scope(name):
        h = int(input.get_shape()[1]) * s
        w = int(input.get_shape()[2]) * s
        _ = tf.image.resize_nearest_neighbor(input, [h, w])
        _ = conv2d(_, output_shape, is_train, k=k, s=1,
                   norm=False, activation_fn=None)
        _ = norm_and_act(_, is_train, norm=norm, activation_fn=activation_fn)
        if info: print_info(name, _.get_shape().as_list(), activation_fn)
    return _


def fc(input, output_shape, is_train, info=False, norm='batch',
       activation_fn=lrelu, name="fc"):
    with tf.variable_scope(name):
        _ = slim.fully_connected(input, output_shape, activation_fn=None)
        _ = norm_and_act(_, is_train, norm=norm, activation_fn=activation_fn)
        if info: print_info(name, _.get_shape().as_list(), activation_fn)
    return _


def snconv2d(input, output_dim, is_train, info=False, k=3, s=2, sn_iters=1,
            activation_fn=lrelu, norm='none', update_coll=None, name='snconv2d'):
    """
    Creates a spectral normalized (SN) convolutional layer.
    """
    # input: 4D input tensor (batch size, height, width, channel).
    # sn_iters: The number of SN iterations.
    # update_collection: The update collection used in spectral_normed_weight.
    # Returns:
    # _: The normalized tensor.
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k, k, input.get_shape()[-1], output_dim],
                            initializer=tf.contrib.layers.xavier_initializer())
        w_bar = spectral_normed_weight(w, num_iters=sn_iters,
                                        update_collection=update_coll)
        _ = tf.nn.conv2d(input, w_bar, strides=[1, s, s, 1], padding='SAME')
        biases = tf.get_variable('biases', [output_dim],
                                initializer=tf.zeros_initializer())
        _ = tf.nn.bias_add(_, biases)
        _ = norm_and_act(_, is_train, norm=norm, activation_fn=activation_fn)
        if info: print_info(name, _.get_shape().as_list(), activation_fn)
    return _


def sn_conv1x1(input, output_dim, update_coll=None, name='sn_conv1x1'):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [1, 1, input.get_shape()[-1], output_dim],
                            initializer=tf.contrib.layers.xavier_initializer())
        w_bar = spectral_normed_weight(w, num_iters=1,
                                        update_collection=update_coll)
        _ = tf.nn.conv2d(input, w_bar, strides=[1, 1, 1, 1], padding='SAME')
    return _


def selfAttn(input, update_coll=None, info=False, name='self_attn'):
    with tf.variable_scope(name):
        bs, h, w, c = input.get_shape().as_list()
        location_num = h*w

        # theta
        theta = sn_conv1x1(input, c//8, update_coll=update_coll, name='sn_conv_theta')
        theta = tf.reshape(theta, [bs, location_num, c//8])
        print('theta', theta.get_shape().as_list())

        # phi
        phi = sn_conv1x1(input, c//8, update_coll=update_coll, name='sn_conv_phi')
        phi = tf.layers.max_pooling2d(inputs=phi, pool_size=[2, 2], strides=2)
        phi = tf.reshape(phi, [bs, -1, c//8])
        print('phi', phi.get_shape().as_list())

        # attn
        attn = tf.matmul(theta, phi, transpose_b=True)
        attn = tf.nn.softmax(attn)
        print('attn', attn.get_shape().as_list())

        # g
        g = sn_conv1x1(input, c//2, update_coll=update_coll, name='sn_conv_g')
        g = tf.layers.max_pooling2d(inputs=g, pool_size=[2, 2], strides=2)
        g = tf.reshape(g, [bs, -1, c//2])
        print('g', g.get_shape().as_list())

        sigma = tf.get_variable('sigma_ratio', [], initializer=tf.constant_initializer(0.0))
        _ = tf.matmul(attn, g)
        _ = tf.reshape(_, [bs, h, w, c//2])
        _ = sn_conv1x1(_, c, update_coll=update_coll, name='sn_conv_attng')
        _ = _*sigma + input
        if info: print_info(name, _.get_shape().as_list(), activation_fn=None)

    return _, attn
