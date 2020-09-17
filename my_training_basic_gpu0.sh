#!/bin/bash

export CUDA_VISIBLE_DEVICES="0"
printf "gpu"
echo $CUDA_VISIBLE_DEVICES

cd ..

animal="mouse1"
data_base_folder='data_trainRun_lineSpeed_val0'
prefix_list=("15frameBefore_clean" "15frameBefore_hollow_thrLocal_good" "raw" "clean" "residual" "hollow_thrLocal_good" "hollow_thrGlobal_good" "hollow_thrLocal_reverse" "hollow_thrGlobal_reverse" "random" "scramble")
prefix_str_list=("_15frameBefore_clean" "_15frameBefore_hollow_thrLocal_good" "" "_clean" "_residual" "_hollow_thrLocal_good" "_hollow_thrGlobal_good" "_hollow_thrLocal_reverse" "_hollow_thrGlobal_reverse" "_random" "_scramble")
lr_list=(1e-4)
loss_list=("l1")
batch_list=("none")
fc_list=(4)
conv_list=(8)
element_list=(2)
ckpt_path="./model_Val2Ckpt/mouse2/model-8000"

while getopts "d:" OPTION
do
  case $OPTION in
    d)
      echo "Set data_base_folder"
      data_base_folder=${OPTARG}
      ;;
    esac
done


for element in ${element_list[@]}
do
  for lrate in ${lr_list[@]}
  do
    for loss in ${loss_list[@]}
    do
      for norm in ${batch_list[@]}
      do
        for fc in ${fc_list[@]}
        do
          for conv in ${conv_list[@]}
          do
            dataset="./$data_base_folder${prefix_str_list[$element]}/$animal"
            echo "prefix: ${prefix_list[$element]}, dataset: $dataset, lrate: $lrate, loss: $loss, norm: $norm, nfc: $fc, nconv: $conv"
            python trainer.py --prefix ${prefix_list[$element]} \
            --dataset_path $dataset --learning_rate $lrate --d_norm_type $norm \
            --loss_type $loss --num_d_conv $conv --num_d_fc $fc \
            --ckpt_save_step 500 --max_training_steps 15001 &
          done
        done
      done
    done
  done
done
