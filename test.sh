#!/bin/bash

CUDA_VISIBLE_DEVICES="0" \
python test.py \
--checkpoint "../20260318/" \
--save_path "../20260318_results/" \
--test_image_path "/data1/workspace/ai_shared_workspace/train_data/wall_seg_crop/data_test/images/" \
--test_gt_path "/data1/workspace/ai_shared_workspace/train_data/wall_seg_crop/data_test/masks/"
