import argparse
import os
import time

import cv2
import imageio
import numpy as np
import onnxruntime

from dataset import TestDataset

parser = argparse.ArgumentParser()
parser.add_argument(
    "--checkpoint", type=str, required=True, help="path to the checkpoint of sam2-unet"
)
parser.add_argument(
    "--save_path", type=str, required=True, help="path to save the predicted masks"
)
parser.add_argument(
    "--test_image_path",
    type=str,
    default="../wall_seg_crop/data_test/images/",
    help="path to the image files for testing",
)
parser.add_argument(
    "--test_gt_path",
    type=str,
    default="../wall_seg_crop/data_test/masks/",
    help="path to the mask files for testing",
)
parser.add_argument("--size", default=960, type=int)
parser.add_argument("--use_cpu", action="store_true", default=False, help="inference using CPU")
args = parser.parse_args()

# Determine the device for ONNX Runtime
# Check if CUDA is available in PyTorch, then map to ONNX Runtime providers
providers = ["CPUExecutionProvider"]
try:
    import torch

    if torch.cuda.is_available() and not args.use_cpu:
        # Attempt to use CUDAExecutionProvider, fallback to CPU if CUDA is not fully set up for ONNX Runtime
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        print("CUDA is available. Attempting to use GPU for ONNX Runtime.")
except Exception:
    print("ModuleNotFoundError: No module named 'torch'")

test_loader = TestDataset(args.test_image_path, args.test_gt_path, args.size)

# 1. Create an ONNX Runtime session, specifying the providers
# The 'providers' argument tells ONNX Runtime which hardware backend to use.
model = onnxruntime.InferenceSession(args.checkpoint, providers=providers)

# 2. Get the name of the input layer
input_name = model.get_inputs()[0].name
os.makedirs(args.save_path, exist_ok=True)
test_time = []

print(f"Starting inference with ONNX Runtime using providers: {model.get_providers()}")

for i in range(test_loader.size):
    image, gt, name, padding = test_loader.load_data()

    # Ensure the input 'image' is a NumPy array, which ONNX Runtime expects.
    # It's already on CPU after .cpu().numpy() in your original code, which is fine.
    # ONNX Runtime will handle moving it to GPU if CUDAExecutionProvider is active.
    image = image.cpu().numpy()

    time_start = time.time()

    # 3. Run the model
    # The input 'image' (a numpy array) will be sent to the specified provider (GPU if CUDA)
    res_padded, _, _ = model.run(None, {input_name: image})

    process_time = time.time() - time_start
    test_time.append(process_time)

    gt = np.asarray(gt, np.float32)
    gt_h, gt_w = gt.shape[:2]

    # Post-processing: ONNX output 'res' is a numpy array
    pad_left, pad_top, pad_right, pad_bottom = padding
    res = res_padded[:, :, pad_top : args.size - pad_bottom, pad_left : args.size - pad_right]
    res_sigmoid = 1 / (1 + np.exp(-res))
    res = np.squeeze(res_sigmoid)
    res = cv2.resize(res, (gt_w, gt_h), interpolation=cv2.INTER_LINEAR)
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    res = (res * 255).astype(np.uint8)
    print("Saving " + name)
    print("process_time:", process_time)
    imageio.imsave(os.path.join(args.save_path, name[:-4] + ".png"), res)

print("test_time:", np.mean(test_time))
