#!/bin/bash

CUDA_VISIBLE_DEVICES="0" \
python test_onnx.py \
--checkpoint "/home/ha.nguyen/workspace/docker_volumes/SAM3-UNet/SAM3-UNet_epoch-85_loss-0.151_iou-0.803.onnx" \
--save_path "../results_20260318_3_epoch-85/" \
--test_image_path "/data1/workspace/ai_shared_workspace/train_data/wall_seg_crop/data_test/images/" \
--test_gt_path "/data1/workspace/ai_shared_workspace/train_data/wall_seg_crop/data_test/masks/"
