#!/bin/bash

CUDA_VISIBLE_DEVICES="0" \
python train.py \
--save_path "../20260318_0" \
--sam3_path "/data1/workspace/ai_shared_workspace/model_zoo_shared/sam3/sam3.pt" \
--train_image_path "/data1/workspace/ai_shared_workspace/train_data/wall_seg_crop/data_train/images/" \
--train_mask_path "/data1/workspace/ai_shared_workspace/train_data/wall_seg_crop/data_train/masks/" \
--test_image_path "/data1/workspace/ai_shared_workspace/train_data/wall_seg_crop/data_test/images/" \
--test_gt_path "/data1/workspace/ai_shared_workspace/train_data/wall_seg_crop/data_test/masks/"
