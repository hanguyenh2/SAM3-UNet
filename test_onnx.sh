#!/bin/bash

CUDA_VISIBLE_DEVICES="5" \
python test_onnx.py \
--checkpoint "/home/ha.nguyen/workspace/docker_volumes/SAM3-UNet_epoch-8_loss-0.147_iou-0.862.onnx" \
--save_path "../results_20260319_1344_lr-0.001_weight-5_3_epoch-8/" \
--test_image_path "/data1/workspace/ai_shared_workspace/train_data/wall_seg_crop/data_test/images/" \
--test_gt_path "/data1/workspace/ai_shared_workspace/train_data/wall_seg_crop/data_test/masks/"
