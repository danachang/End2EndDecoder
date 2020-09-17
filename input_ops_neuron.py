import numpy as np
import tensorflow as tf

from util import log

def check_data_id(dataset, data_id):
    if not data_id:
        return

    wrong = []
    for id in data_id:
        if id in dataset.data:
            pass
        else:
            wrong.append(id)

    if len(wrong) > 0:
        raise RuntimeError("There are %d invalid ids, including %s" % (
            len(wrong), wrong[:5]
        ))


def create_input_ops(dataset,
                     batch_size,
                     num_threads=16,           # for creating batches
                     is_training=False,
                     data_id=None,
                     scope='inputs',
                     shuffle=True,
                     ):
    '''
    Return a batched tensor for the inputs from the dataset.
    '''
    input_ops = {}

    if data_id is None:
        data_id = dataset.ids
        log.info("input_ops [%s]: Using %d IDs from dataset", scope, len(data_id))
    else:
        log.info("input_ops [%s]: Using specified %d IDs", scope, len(data_id))

    # single operations
    with tf.device("/cpu:0"), tf.name_scope(scope):
        input_ops['id'] = tf.train.string_input_producer(
           tf.convert_to_tensor(data_id),
            capacity=512,
        ).dequeue(name='input_ids_dequeue')

        a, l = dataset.get_data(data_id[0])

        def load_fn(id):
            # activity [l]
            # label [v]
            activity, label = dataset.get_data(id.decode('utf-8'))
            return (id, activity.astype(np.float32), label.astype(np.float32))

        input_ops['id'], input_ops['activity'], input_ops['label'] = tf.py_func(
            load_fn, inp=[input_ops['id']],
            Tout=[tf.string, tf.float32, tf.float32],
            name='func_hp'
        )

        input_ops['id'].set_shape([])
        input_ops['activity'].set_shape(list(a.shape))
        input_ops['label'].set_shape(list(l.shape))

    # batchify
    capacity = 2 * batch_size * num_threads
    min_capacity = min(int(capacity * 0.75), 1024)
    # change min_capacity for testing batch samples
    #capacity = batch_size + 1
    #min_capacity = capacity - batch_size
    # end of testing batch samples

    if shuffle:
        print('shuffle on.....')
        batch_ops = tf.train.shuffle_batch(
            input_ops,
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=capacity, seed=123,
            min_after_dequeue=min_capacity,
        )
    else:
        print('batch.....')
        batch_ops = tf.train.batch(
            input_ops,
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=capacity,
        )

    return input_ops, batch_ops



def create_input_ops_ROI(dataset,
                     batch_size,
                     num_threads=16,           # for creating batches
                     is_training=False,
                     data_id=None,
                     scope='inputs',
                     shuffle=True,
                     ):
    '''
    Return a batched tensor for the inputs from the dataset.
    '''
    input_ops = {}

    if data_id is None:
        data_id = dataset.ids
        log.info("input_ops [%s]: Using %d IDs from dataset", scope, len(data_id))
    else:
        log.info("input_ops [%s]: Using specified %d IDs", scope, len(data_id))

    # single operations
    with tf.device("/cpu:0"), tf.name_scope(scope):
        input_ops['id'] = tf.train.string_input_producer(
           tf.convert_to_tensor(data_id),
            capacity=512,
        ).dequeue(name='input_ids_dequeue')

        a, l = dataset.get_data(data_id[0])

        def load_fn_ROI(id):
            # activity [l]
            # label [v]
            activity, label = dataset.get_data_ROI(id.decode('utf-8'))
            return (id, activity.astype(np.float32), label.astype(np.float32))

        input_ops['id'], input_ops['activity'], input_ops['label'] = tf.py_func(
            load_fn_ROI, inp=[input_ops['id']],
            Tout=[tf.string, tf.float32, tf.float32],
            name='func_hp'
        )

        input_ops['id'].set_shape([])
        input_ops['activity'].set_shape(list(a.shape))
        input_ops['label'].set_shape(list(l.shape))

    # batchify
    capacity = 2 * batch_size * num_threads
    min_capacity = min(int(capacity * 0.75), 1024)
    # change min_capacity for testing batch samples
    #capacity = batch_size + 1
    #min_capacity = capacity - batch_size
    # end of testing batch samples

    if shuffle:
        print('shuffle on.....')
        batch_ops = tf.train.shuffle_batch(
            input_ops,
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=capacity, seed=123,
            min_after_dequeue=min_capacity,
        )
    else:
        print('batch.....')
        batch_ops = tf.train.batch(
            input_ops,
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=capacity,
        )

    return input_ops, batch_ops
