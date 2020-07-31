#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/home/jovyan/work/code/yxu-yolov3-xview
CUDA_VISIBLE_DEVICES=0 python train_syn_xview_background_1cls_mean_best_example.py --cfg_dict 'train_cfg/train_1cls_syn_only_example.json'
