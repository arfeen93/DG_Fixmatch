#!/bin/bash

domain=("photo" "art" "cartoon" "sketch")

times=1
for i in `seq 1 $times`
do
  max=$((${#domain[@]}-1))
#  for j in `seq 0 $max`
#  do
    for lr_step in 20
    do
      for epoch_num in 50
      do
        for alpha_mix in 0.3

        do

          dir_name="PACS/default/${domain[3]}${i}"
          echo $dir_name
          #CUDA_VISIBLE_DEVICES=1 python ../main/main.py \
          python ../main/main.py \
            --data-root='/home/arfeen/papers_code/dom_gen_aaai_2020/PACS/kfold/' \
            --save-root='/home/arfeen/papers_code/dom_gen_aaai_2020/' \
            --result-dir=$dir_name \
            --train='general' \
            --data='PACS' \
            --model='caffenet' \
            --clustering \
            --clustering-method='Kmeans' \
            --num-clustering=3 \
            --clustering-step=1 \
            --entropy='default' \
            --exp-num=3 \
            --gpu=0 \
            --num-epoch=$epoch_num \
            --scheduler='step' \
            --lr=0.001 \
            --lr-step=$lr_step \
            --lr-decay-gamma=0.1 \
            --nesterov \
            --fc-weight=10.0 \
            --disc-weight=10.0 \
            --entropy-weight=1.0 \
            --grl-weight=1.0 \
            --loss-disc-weight \
            --color-jitter \
            --min-scale=0.8 \
            --instance-stat \
            --alpha-mixup=$alpha_mix
        done
      done
     done
  #done
done


        #--data-root='/home/arfeen/papers_code/dom_gen_aaai_2020/PACS/kfold/' \
      #--save-root='/home/arfeen/papers_code/dom_gen_aaai_2020/' \