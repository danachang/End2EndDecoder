#!/bin/bash

export CUDA_VISIBLE_DEVICES="0"
printf "gpu"
echo $CUDA_VISIBLE_DEVICES

data_base_folder='data_trainRun_lineSpeed_val1'
train_dir_base_path='./trainRun_lineSpeed_val1'
prefix='raw'
loss_metric='l1'
bs=500

cd ..

while getopts "d:p:t:b:" OPTION
do
  case $OPTION in
    d)
      echo "Set data_base_folder"
      data_base_folder=${OPTARG}
      ;;
    p)
      echo "Set prefix"
      prefix=${OPTARG}
      ;;
    t)
      echo "Set train_dir_base_path"
      train_dir_base_path=${OPTARG}
      ;;
    b)
      echo "Set batch_size"
      bs=${OPTARG}
      ;;
    esac
done

dir_array="$train_dir_base_path/train_dir/$prefix/"

train_dir_base_path="$train_dir_base_path/"
mapfile -t array < <(python select_checkpoint.py --filepath $train_dir_base_path --prefix $prefix)
mouselist=($(echo ${array[0]} | tr -d "[],'"))
ckptlist=($(echo ${array[1]} | tr -d "[],'"))
archlist=($(echo ${array[2]} | tr -d "[],'"))
dconvlist=($(echo ${array[3]} | tr -d "[],'"))
dfclist=($(echo ${array[4]} | tr -d "[],'"))
normlist=($(echo ${array[5]} | tr -d "[],'"))
lrlist=($(echo ${array[6]} | tr -d "[],'"))
trainlist=($(echo ${array[7]} | tr -d "[],'"))
element_list=($(echo ${array[8]} | tr -d "[],'"))

for element in ${element_list[@]}
do
  animal=${mouselist[$element]}
  fID="$dir_array$animal/${trainlist[$element]}"
  arch=${archlist[$element]}
  dconv=${dconvlist[$element]}
  dfc=${dfclist[$element]}
  norm=${normlist[$element]}
  ckpt=${ckptlist[$element]}

  if [ $prefix = 'raw' ]
  then
    dataset="$data_base_folder/$animal"
  else
    prefix_str="_$prefix"
    dataset="$data_base_folder$prefix_str/$animal"
  fi

  python getEnsemble.py --prefix $prefix --batch_size $bs --arch $arch \
  --num_d_fc $dfc --num_d_conv $dconv --d_norm_type $norm \
  --loss_type $loss_metric --dataset_path $dataset --train_dir $fID \
  --checkpoint $ckpt

done
