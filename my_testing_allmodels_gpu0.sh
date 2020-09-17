#!/bin/bash

export CUDA_VISIBLE_DEVICES="0"
printf "gpu"
echo $CUDA_VISIBLE_DEVICES

cd ..

data_base_folder='data_trainRun_lineSpeed_val0'
arch='ConvNet'
prefix='raw'
animal='mouse1'
fc_list=4
conv_list=8
norm_list='none'
bs=500

while getopts "d:a:p:m:f:c:n:b:" OPTION
do
  case $OPTION in
    d)
      echo "Set data_base_folder"
      data_base_folder=${OPTARG}
      ;;
    a)
      echo "Set arch"
      arch=${OPTARG}
      ;;
    p)
      echo "Set prefix"
      prefix=${OPTARG}
      ;;
    m)
      echo "Set animal"
      animal=${OPTARG}
      ;;
    f)
      echo "Set dfc"
      fc_list=${OPTARG}
      ;;
    c)
      echo "Set dconv"
      conv_list=${OPTARG}
      ;;
    n)
      echo "Set norm_type"
      norm_list=${OPTARG}
      ;;
    b)
      echo "Set batch_size"
      bs=${OPTARG}
      ;;
    esac
done

if [ $prefix = 'raw' ]
then
  dataset="./$data_base_folder/$animal"
else
  prefix_str="_$prefix"
  dataset="./$data_base_folder$prefix_str/$animal"
fi


loss_metric='l1'

dir_array='./train_dir/'
fc_char='fc'
conv_char='conv'
norm_char='_norm'

if [ $arch = 'ConvNet' ]
then
  dir_array="$dir_array$prefix/$animal/$arch/$fc_char$fc_list/$conv_char$conv_list/$norm_list$norm_char/*"
else
  dir_array="$dir_array$prefix/$animal/$arch/$fc_char$fc_list/*"
  conv_list=0
  norm_list='none'
  preload='True'
fi

dir_array="./train_dir/$prefix/"
#dir_array="./train_dir/random/"
dir_array="$dir_array$animal/*"

echo $dir_array

for fID in ${dir_array[@]}
do
  fID_array="$fID/*"
  for modelID in ${fID_array[@]}
  do
    if [[ $modelID == *".index" ]] && [[ $modelID == *"model"* ]]
    then
      #echo $modelID
      name=${modelID##*/}
      ckpt=${name%.*}
      python evaler_varSize.py --prefix $prefix --batch_size $bs --arch $arch \
      --num_d_fc $fc_list --num_d_conv $conv_list --d_norm_type $norm_list \
      --loss_type $loss_metric --dataset_path $dataset --train_dir $fID \
      --checkpoint $ckpt
    fi
  done
done
