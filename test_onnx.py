import abc
import argparse
import os
import time
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from scipy.special import expit

from eval import evaluate_dataset, evaluate_segmentation_performance, print_eval_report


def normalize_image(image, mean=None, std=None):
    """
    Normalize the image to the range [0, 1].
    If mean and std are provided, normalize the image using them.

    Args:
        image (np.ndarray): The input image to normalize.
        mean (list, optional): The mean values for normalization.
        std (list, optional): The standard deviation values for normalization.
    Returns:
        np.ndarray: The normalized image.
    """
    image = image / 255.0
    if mean is not None and std is not None:
        image = (image - mean) / std
    return image


class ModelMixin:
    """
    A mixin class for model inference, providing a unified API for ONNXRuntime and OpenVINO backends.
    Used in [
        `WindoorAttributesTextDetector`,
        `WindoorAttributesTextRecognizer`,
        `DFINEBasedWindoorsDetector`,
        `YoloxBasedWindoorsDetector`,
        ]

    Methods:
        - `load`: Automatically selects and loads the backend based on the model file extension.
        - `run`: Runs inference with a consistent interface, returning either a tuple or dict of outputs.
    """

    def __init__(
        self,
        model_path: str,
        high_acc: bool = False,
        providers: list[str] = None,
        **kwargs,
    ):
        self.model_path = model_path
        self.high_acc = high_acc
        self.providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.model = None
        self.run = None
        self.run_async = None
        self.input_names = None
        self.output_names = None
        self.load()

    def is_ready(self) -> bool:
        """
        Check whether model is ready
        """
        if self.run is None:
            return False
        return True

    def load(self) -> None:
        """
        Loads the model by automatically selecting the backend based on `self.model_path`.

        Assumes `self.model_path` is set (e.g., in the parent class initializer).
        - `.onnx` files use ONNXRuntime.
        - `.xml` files use OpenVINO (with `.bin` assumed in the same directory).

        Raises:
            ValueError: If the model format is unsupported.
            Exception: If loading fails.
        """
        if self.model_path.endswith(".onnx"):
            self._load_onnx_model()
        else:
            raise ValueError(f"Unsupported model format: {self.model_path}")

    def _load_onnx_model(self) -> None:
        """Initialize ONNX model and set up inference."""
        try:
            import onnxruntime

            print(f"Loading ONNX model from {self.model_path}")
            self.model = onnxruntime.InferenceSession(self.model_path, providers=self.providers)
            self.input_names = [input.name for input in self.model.get_inputs()]
            self.output_names = [output.name for output in self.model.get_outputs()]
        except Exception as e:
            raise Exception(f"Failed to load ONNX model: {e}")
        self.run = self._run_onnx_model

    def _run_onnx_model(
        self, inputs: dict, return_dict: bool = False
    ) -> Union[Tuple[np.ndarray], Dict[str, np.ndarray]]:
        """
        Run inference on the ONNX model.

        Args:
            inputs: Dictionary mapping input names to numpy arrays.
            return_dict: If True, return a dict with output names as keys; otherwise, a tuple.

        Returns:
            Tuple of output arrays or a dict mapping output names to arrays.
        """
        outputs = self.model.run(self.output_names, inputs)
        if return_dict:
            return dict(zip(self.output_names, outputs))
        return tuple(outputs)


class BaseSegmenter(metaclass=abc.ABCMeta):
    """Base class for all floor plan segmenter."""

    def __init__(
        self,
        input_size: int,
        mean: List[float],
        std: List[float],
        score_threshold: float,
        **kwargs,
    ):
        self.parser = None
        self.input_size = input_size
        self.mean = mean
        self.std = std
        self.score_threshold = score_threshold

    @abc.abstractmethod
    def segment(self, image: np.ndarray) -> np.ndarray | None:
        pass


class SamUnetBaseSegmenter(ModelMixin, BaseSegmenter):
    """Class for floor plan segmentation."""

    def __init__(self, *args, **kwargs):
        ModelMixin.__init__(self, *args, **kwargs)
        BaseSegmenter.__init__(self, *args, **kwargs)

    def segment(self, image: np.ndarray) -> np.ndarray | None:
        """Segment the input image.

        Args:
            image (np.ndarray): Input image

        Returns:
            np.ndarray: segmented mask
        """
        # 1. Preprocess the image
        img, pad_ltwh = self.preprocess(image[:, :, ::-1], self.input_size)
        # 2. Get image width and height
        height, width = image.shape[:2]
        # 3. Run the model
        try:
            output = self.run(inputs={"images": img})[0]
        except Exception as e:
            # 3.1 Return zeros mask if Exception
            print(f"Error running Segmentation Model: {e}")
            return None

        # 4. Result conversion
        # 4.1. Remove padding
        pad_x, pad_y, pad_w, pad_h = pad_ltwh
        res = output[:, :, pad_y : pad_y + pad_h, pad_x : pad_x + pad_w]
        # 4.2. Converts logits to probabilities (0.0 to 1.0)
        res_sigmoid = expit(res)
        # 4.3. Removes singleton dimensions, now (H_model, W_model)
        res = np.squeeze(res_sigmoid)
        # 4.4. Resize to original image dimensions (res is still 0.0-1.0 probabilities)
        res = cv2.resize(res, (width, height), interpolation=cv2.INTER_LINEAR)
        # 4.5. Apply the threshold
        res = res >= self.score_threshold
        # 4.6. Convert the boolean mask to uint8 (0 for False, 255 for True)
        res = (res * 255).astype(np.uint8)
        return res

    def preprocess(
        self, img: np.ndarray, input_size: int
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Preprocess the input image.
        1. Resize the longest side to input_size, maintaining aspect ratio
        2. Pad the shorter side with zero to make the image square
        3. Normalize and convert image to pytorch tensor

        Args:
            img (np.ndarray): Input image in RGB format
            input_size (int): Input model size

        Returns:
            np.ndarray: processed_image
        """
        # 1. Resize the longest side to input_size, maintaining aspect ratio
        # 1.1. Get original height and width
        original_h, original_w = img.shape[:2]
        # 1.2. Calculate the scaling factor for longest side
        scale_factor = input_size / max(original_h, original_w)
        # 1.3. Calculate new dimensions while maintaining aspect ratio
        new_h = int(round(original_h * scale_factor))
        new_w = int(round(original_w * scale_factor))
        # 1.4. Set interpolation based on scale_factor
        if scale_factor < 1:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_LINEAR
        # 1.5. Resize image
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=interpolation)

        # 2. Pad the shorter side with zero to make the image square
        # 2.1. Calculate padding amounts
        pad_h = input_size - new_h
        pad_w = input_size - new_w
        # 2.2. Determine padding for top and left to center the image
        pad_y = pad_h // 2
        pad_x = pad_w // 2
        pad_ltwh = (pad_x, pad_y, new_w, new_h)
        # 2.3. Create a zero image
        new_image = np.zeros((input_size, input_size, 3), dtype=np.uint8)
        # 2.4. Place resized image in center
        new_image[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized_img

        # 3. Normalize and prepare for model
        new_image = normalize_image(image=new_image, mean=self.mean, std=self.std)
        new_image = new_image.transpose(2, 0, 1)
        new_image = np.expand_dims(new_image, axis=0)
        return new_image.astype(np.float32), pad_ltwh


if __name__ == "__main__":
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
        default="/data1/workspace/ai_shared_workspace/train_data/wall_seg_crop/data_test/images/",
        help="path to the image files for testing",
    )
    parser.add_argument(
        "--test_gt_path",
        type=str,
        default="/data1/workspace/ai_shared_workspace/train_data/wall_seg_crop/data_test/masks/",
        help="path to the mask files for testing",
    )
    parser.add_argument(
        "--use_cpu", action="store_true", default=False, help="inference using CPU"
    )
    args = parser.parse_args()

    # 1. Determine the device for ONNX Runtime
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

    # 2. Load model
    segmentor = SamUnetBaseSegmenter(
        model_path=args.checkpoint,
        input_size=1344,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        score_threshold=0.1,
        providers=providers,
    )

    # Get info for next step
    image_root = args.test_image_path
    gt_root = args.test_gt_path
    gt_list = sorted(os.listdir(gt_root))
    results = []
    test_time = []
    os.makedirs(args.save_path, exist_ok=True)
    log_path = os.path.join(args.save_path, "log.txt")
    # 3. Segment each image
    len_gt_list = len(gt_list)
    for i, file_name in enumerate(gt_list):
        # 2.1. Get image and gt path
        gt_path = os.path.join(gt_root, file_name)
        image_path = os.path.join(image_root, file_name[:-4] + ".png")
        # 2.2. Read image and gt and
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        cv2.imread(image_path)
        image = np.array(Image.open(image_path).convert("RGB"))
        # 2.3. Segment image
        time_start = time.time()
        pred_mask = segmentor.segment(image)
        process_time = time.time() - time_start
        test_time.append(process_time)
        # 2.4. Save result
        cv2.imwrite(os.path.join(args.save_path, file_name[:-4] + ".png"), pred_mask)
        # 2.5. Evaluate result
        result = evaluate_segmentation_performance(pred_mask, gt_mask)
        # 2.6. Save and print eval result
        title = f"[{i + 1}/{len_gt_list}][{process_time:.2f}s] {file_name}"
        print_eval_report(result, title=title, log_path=log_path)
        results.append(result)

    # 3. Evaluate all results
    final_result = evaluate_dataset(results)
    print_eval_report(
        final_result, title=f"Average Process time: {np.mean(test_time):.2f}s", log_path=log_path
    )
