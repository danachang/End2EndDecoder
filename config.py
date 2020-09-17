import argparse
import os

from model import Model
import dataset


def argparser(is_train=True):

    def str2bool(v):
        return v.lower() == 'true'
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--prefix', type=str, default='default')
    parser.add_argument('--train_dir', type=str)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--dataset_path', type=str, default='data/mouse1')
    # Model
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--loss_type', type=str, default='l2',
                        choices=['l1', 'l2'])
    parser.add_argument('--num_d_conv', type=int, default=8)
    parser.add_argument('--num_d_fc', type=int, default=4)
    parser.add_argument('--d_norm_type', type=str, default='none',
                        choices=['batch', 'instance', 'none'])
    parser.add_argument('--output_act_fn', type=str, default='tanh',
                        choices=['tanh', 'linear', 'sigmoid'])

    parser.add_argument('--arch', type=str, default='ConvNet',
                        choices=['MobileNet', 'ResNet50', 'ConvNet'])
    parser.add_argument('--load_pretrained', type=str2bool, default=False)

    # Training config {{{
    # ========
    # log
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--write_summary_step', type=int, default=100)
    parser.add_argument('--ckpt_save_step', type=int, default=1000)
    # learning
    parser.add_argument('--max_training_steps', type=int, default=3000000)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--lr_weight_decay', type=str2bool, default=False)
    # }}}

    # Testing config {{{
    # ========
    # summary file
    parser.add_argument('--summary_file', type=str, default='summary.txt',
                        help='the path to the summary file')
    parser.add_argument('--summary_model_file', type=str, default='summary_model_file.txt')
    parser.add_argument('--summary_indv_file', type=str, default='summary_indv_file.txt')

    # }}}

    config = parser.parse_args()
    dataset_train, dataset_val, dataset_test = \
        dataset.create_default_splits(config.dataset_path)

    image, label = dataset_train.get_data(dataset_train.ids[0])
    config.h = image.shape[0]
    config.w = image.shape[1]
    config.c = image.shape[2]
    config.output_dim = label.shape[0]

    # --- create model ---
    model = Model(config, debug_information=config.debug, is_train=is_train)

    return config, model, dataset_train, dataset_val, dataset_test
