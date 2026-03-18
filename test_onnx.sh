#!/bin/bash

CUDA_VISIBLE_DEVICES="4" \
python test_onnx.py \
--checkpoint "/home/ha.nguyen/workspace/docker_volumes/SAM3-UNet/SAM3-UNet_epoch-60_loss-0.175_iou-0.792.onnx" \
--save_path "../20260318_3_results/" \
--test_image_path "/data1/workspace/ai_shared_workspace/train_data/wall_seg_crop/data_test/images/" \
--test_gt_path "/data1/workspace/ai_shared_workspace/train_data/wall_seg_crop/data_test/masks/"
