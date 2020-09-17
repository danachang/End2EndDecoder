#!/bin/bash

export CUDA_VISIBLE_DEVICES="0"
printf "gpu"
echo $CUDA_VISIBLE_DEVICES

cd ..

fID='/home/cj/mnt/Tmaze/train_run/lineSpeed/trainRun_lineSpeed_val0/train_dir/raw/mouse1/mouse1_l1_bs_64_lr_0.0001_8_conv_4_fc_none_norm_act_tanh-20200304-224907'
dataset='/home/cj/mnt/Dataset/train_run/lineSpeed/data_trainRun_lineSpeed_val0/mouse1'
model_list=('model-26000')

fID='/home/cj/mnt/Tmaze/train_run/lineSpeed/trainRun_lineSpeed_val1/train_dir/raw/mouse1/mouse1_l1_bs_64_lr_0.0001_8_conv_4_fc_none_norm_act_tanh-20200225-195031'
dataset='/home/cj/mnt/Dataset/train_run/lineSpeed/data_trainRun_lineSpeed_val1/mouse1'
model_list=('model-28750')

for ckpt in ${model_list[@]}
do
  python getEnsemble.py --prefix 'raw' --batch_size 500 --arch 'ConvNet' \
  --num_d_fc 4 --num_d_conv 8 --d_norm_type 'none' \
  --loss_type 'l1' --dataset_path $dataset --train_dir $fID \
  --checkpoint $ckpt
done
