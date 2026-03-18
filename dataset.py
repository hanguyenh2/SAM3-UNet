import os
import random

import cv2
import numpy as np
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class ToTensor(object):

    def __call__(self, data):
        image, label = data["image"], data["label"]
        return {"image": F.to_tensor(image), "label": F.to_tensor(label)}


class Resize(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        image, label = data["image"], data["label"]

        return {
            "image": F.resize(image, self.size),
            "label": F.resize(label, self.size, interpolation=InterpolationMode.NEAREST),
        }


class ResizeLongestSideAndPad(object):
    """
    Resizes the image and label such that the longest side matches `size`,
    maintaining aspect ratio, and then pads the shorter side with zeros
    to make the image square.
    """

    def __init__(self, size: int, p: float = 0.5):
        """
        Args:
            size (int): The target size for the longest side.
                        The output image will be (size, size).
        """
        self.size = size
        self.p = p
        self.pad_range = (1.0, 1.5)
        self.crop_range = (0.5, 1.0)

    def __call__(self, data: dict) -> dict:
        image, label = data["image"], data["label"]

        original_h, original_w = image.shape[-2:]

        # 1. Pad with white
        if random.random() < self.p:
            # 1.1. Randomize scale_factor to pad
            scale_factor_h = random.uniform(*self.pad_range)
            scale_factor_w = random.uniform(*self.pad_range)

            # 1.2. Calculate new dimensions while maintaining aspect ratio
            new_h = int(round(original_h * scale_factor_h))
            new_w = int(round(original_w * scale_factor_w))

            # 1.3. Calculate padding amounts
            pad_h = new_h - original_h
            pad_w = new_w - original_w

            # 1.4. Determine padding for top/bottom and left/right
            pad_top = random.randint(0, pad_h)
            pad_bottom = pad_h - pad_top
            pad_left = random.randint(0, pad_w)
            pad_right = pad_w - pad_left

            # 1.5. Pad image with 1 and Pad label with 0
            padding = [pad_left, pad_right, pad_top, pad_bottom]
            processed_image = F.pad(image, padding, fill=1.0)
            processed_label = F.pad(label, padding, fill=0)

        # 2. Random crop
        else:
            # 2.1. Randomize scale_factor to crop
            scale_factor_h = random.uniform(*self.crop_range)
            scale_factor_w = random.uniform(*self.crop_range)

            # 2.2. Calculate new dimensions while maintaining aspect ratio
            new_h = int(round(original_h * scale_factor_h))
            new_w = int(round(original_w * scale_factor_w))

            # Ensure crop dimensions are not zero
            new_h = max(1, new_h)
            new_w = max(1, new_w)

            # Choose a random top-left corner for the crop
            y1 = random.randint(0, original_h - new_h)
            x1 = random.randint(0, original_w - new_w)

            # Perform the crop using slicing
            processed_image = image[..., y1 : y1 + new_h, x1 : x1 + new_w]
            processed_label = label[..., y1 : y1 + new_h, x1 : x1 + new_w]

        # 3. Perform LongestMaxSize
        # 3.1. Get processed_image dimensions
        processed_h, processed_w = processed_image.shape[-2:]

        # 3.2 Calculate the scaling factor
        scale_factor = self.size / max(processed_h, processed_w)

        # 3.3. Calculate new dimensions while maintaining aspect ratio
        new_h = int(round(processed_h * scale_factor))
        new_w = int(round(processed_w * scale_factor))

        # 3.4. Resize image and label
        # For image: use bilinear or bicubic for quality
        resized_image = F.resize(
            processed_image, [new_h, new_w], interpolation=InterpolationMode.BILINEAR
        )
        # For label (mask): use nearest neighbor to preserve discrete class values
        resized_label = F.resize(
            processed_label, [new_h, new_w], interpolation=InterpolationMode.NEAREST
        )

        # 3.5. Calculate padding amounts
        pad_h = self.size - new_h
        pad_w = self.size - new_w

        # 3.6. Determine padding for top/bottom and left/right to center the image
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        padding = [pad_left, pad_top, pad_right, pad_bottom]  # (left, top, right, bottom)

        # 3.7. Apply padding
        # For image: pad with 0 (black) or mean pixel value if normalized
        padded_image = F.pad(resized_image, padding, fill=0)  # fill=0 for black padding
        # For label: pad with 0 (background class)
        padded_label = F.pad(resized_label, padding, fill=0)  # fill=0 for background class

        return {"image": padded_image, "label": padded_label}


class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        image = F.normalize(image, self.mean, self.std)
        return {"image": image, "label": label}


class RandomRotate(object):
    def __init__(self, p=0.75):
        self.p = p

    def __call__(self, data):
        image, label = data["image"], data["label"]
        # Randomly choose rotation (0, 90, 180, 270 degrees)
        if random.random() < self.p:
            angle = random.choice([90, 180, 270])
            return {
                "image": F.rotate(
                    image, angle, interpolation=InterpolationMode.BILINEAR, expand=False
                ),
                "label": F.rotate(
                    label, angle, interpolation=InterpolationMode.NEAREST, expand=False
                ),
            }
        return {"image": image, "label": label}


class ToGray(object):
    def __init__(self, p=0.5, num_output_channels=3):
        """
        Converts a 3-channel image to grayscale using F.rgb_to_grayscale.
        Args:
            p (float): Probability of applying the transform. Default is 1.0 (always apply).
            num_output_channels (int): Number of output channels (1 or 3).
                                       Typically 3 to keep input shape consistent for models.
        """
        self.p = p
        self.num_output_channels = num_output_channels

    def __call__(self, data):
        image, label = data["image"], data["label"]

        if random.random() < self.p:
            # F.rgb_to_grayscale expects a torch.Tensor (float, CHW)
            # Make sure this transform is applied AFTER ToTensor
            image = F.rgb_to_grayscale(image, num_output_channels=self.num_output_channels)

        return {"image": image, "label": label}


class ColorAugmentations(object):
    def __init__(self, p=0.8):
        """
        Applies a random color augmentation with a given probability.
        Args:
            p (float): Probability of applying any of the color augmentations.
        """
        self.p = p
        # Define ranges for each type of color jitter.
        # These ranges are common defaults or similar to Albumentations defaults.
        self.brightness_range = (0.5, 1.5)  # For RandomBrightnessContrast
        self.contrast_range = (0.5, 1.5)  # For RandomBrightnessContrast
        self.saturation_range = (0.5, 1.5)  # For HueSaturationValue, ColorJitter
        self.hue_range = (-0.5, 0.5)  # For HueSaturationValue, ColorJitter (-0.5 to 0.5)
        self.gamma_range = (0.5, 1.5)  # For RandomGamma

    def __call__(self, data):
        image, label = data["image"], data["label"]

        if random.random() < self.p:
            # Randomly choose one of the four color augmentations
            # 0: BrightnessContrast, 1: ColorJitter (all), 2: HueSaturation, 3: Gamma
            choice = random.randint(0, 3)

            if choice == 0:  # RandomBrightnessContrast (approx)
                brightness_factor = random.uniform(*self.brightness_range)
                contrast_factor = random.uniform(*self.contrast_range)
                image = F.adjust_brightness(image, brightness_factor)
                image = F.adjust_contrast(image, contrast_factor)

            elif choice == 1:  # ColorJitter (all components)
                # Generate factors for all four: brightness, contrast, saturation, hue
                brightness_factor = random.uniform(*self.brightness_range)
                contrast_factor = random.uniform(*self.contrast_range)
                saturation_factor = random.uniform(*self.saturation_range)
                hue_factor = random.uniform(
                    *self.hue_range
                )  # F.adjust_hue expects value from -0.5 to 0.5

                image = F.adjust_brightness(image, brightness_factor)
                image = F.adjust_contrast(image, contrast_factor)
                image = F.adjust_saturation(image, saturation_factor)
                image = F.adjust_hue(image, hue_factor)

            elif choice == 2:  # HueSaturationValue (only hue and saturation)
                saturation_factor = random.uniform(*self.saturation_range)
                hue_factor = random.uniform(*self.hue_range)

                image = F.adjust_saturation(image, saturation_factor)
                image = F.adjust_hue(image, hue_factor)

            elif choice == 3:  # RandomGamma
                gamma_value = random.uniform(*self.gamma_range)
                image = F.adjust_gamma(image, gamma_value)

        return {"image": image, "label": label}


class GaussianBlur(object):
    def __init__(self, p=0.2, blur_limit=(3, 5)):
        """
        Applies Gaussian blur with a given probability and random kernel size.
        Args:
            p (float): Probability of applying the blur. Default is 0.3.
            blur_limit (tuple): A tuple (min_kernel_size, max_kernel_size) for the blur kernel.
                                  Kernel sizes must be odd integers.
        """
        self.p = p
        # Ensure blur_limit contains only odd integers for kernel_size
        self.kernel_sizes = [k for k in range(blur_limit[0], blur_limit[1] + 1) if k % 2 == 1]
        if not self.kernel_sizes:
            raise ValueError("blur_limit must contain at least one odd integer for kernel_size.")

    def __call__(self, data):
        image, label = data["image"], data["label"]

        if random.random() < self.p:
            # Randomly choose a kernel size from the allowed odd integers
            kernel_size = random.choice(self.kernel_sizes)

            # F.gaussian_blur expects a torch.Tensor (float, CHW)
            # Its kernel_size argument can be a single integer or a tuple (height, width).
            # For a square kernel, pass (kernel_size, kernel_size).
            image = F.gaussian_blur(image, kernel_size=[kernel_size, kernel_size])

        return {"image": image, "label": label}


class FullDataset(Dataset):
    def __init__(self, image_root: str, gt_root: str, size: int, mode: str = "train"):
        self.images = [
            image_root + f
            for f in os.listdir(image_root)
            if f.endswith(".jpg") or f.endswith(".png")
        ]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith(".png")]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        if mode == "train":
            self.transform = transforms.Compose(
                [
                    ToTensor(),
                    ResizeLongestSideAndPad(size),
                    RandomRotate(),
                    ToGray(),
                    ColorAugmentations(),
                    GaussianBlur(),
                    Normalize(),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [ToTensor(), ResizeLongestSideAndPad(size), Normalize()]
            )

    def __getitem__(self, idx):
        image = self.rgb_loader(self.images[idx])
        label = self.binary_loader(self.gts[idx])
        data = {"image": image, "label": label}
        data = self.transform(data)
        return data

    def __len__(self):
        return len(self.images)

    def rgb_loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def binary_loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("L")


class ImageToTensor(object):

    def __call__(self, data):
        image = data["image"]
        return {"image": F.to_tensor(image)}


class LongestMaxSizeAndPad(object):
    """
    Resizes the image and label such that the longest side matches `size`,
    maintaining aspect ratio, and then pads the shorter side with zeros
    to make the image square.
    """

    def __init__(self, size: int):
        """
        Args:
            size (int): The target size for the longest side.
                        The output image will be (size, size).
        """
        self.size = size

    def __call__(self, data: dict) -> dict:
        image = data["image"]
        # Make sure image and label are PyTorch Tensors
        # This assumes image is (C, H, W) and label is (H, W) or (1, H, W)
        original_h, original_w = image.shape[-2:]  # Get H, W from (C, H, W)

        # Calculate the scaling factor
        scale_factor = self.size / max(original_h, original_w)

        # Calculate new dimensions while maintaining aspect ratio
        new_h = int(round(original_h * scale_factor))
        new_w = int(round(original_w * scale_factor))

        # Resize image and label
        # For image: use bilinear or bicubic for quality
        resized_image = F.resize(image, [new_h, new_w], interpolation=InterpolationMode.BILINEAR)

        # Calculate padding amounts
        pad_h = self.size - new_h
        pad_w = self.size - new_w

        # Determine padding for top/bottom and left/right to center the image
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        padding = [pad_left, pad_top, pad_right, pad_bottom]  # (left, top, right, bottom)

        # Apply padding
        # For image: pad with 0 (black) or mean pixel value if normalized
        padded_image = F.pad(resized_image, padding, fill=0)  # fill=0 for black padding

        return {"image": padded_image, "padding": padding}


class NormalizeImage(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, padding = sample["image"], sample["padding"]
        image = F.normalize(image, self.mean, self.std)
        return {"image": image, "padding": padding}


class TestDataset:
    def __init__(self, image_root: str, gt_root: str, size: int):
        self.images = [
            image_root + f
            for f in os.listdir(image_root)
            if f.endswith(".jpg") or f.endswith(".png")
        ]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith(".png")]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose(
            [ImageToTensor(), LongestMaxSizeAndPad(size), NormalizeImage()]
        )
        self.size = len(self.images)
        self.index = 0

    def reset_index(self):
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        data = {"image": image}
        data = self.transform(data)
        image = data["image"].unsqueeze(0)
        padding = data["padding"]

        gt = self.binary_loader(self.gts[self.index])
        gt = np.array(gt)

        name = self.images[self.index].split("/")[-1]

        self.index += 1
        return image, gt, name, padding

    def rgb_loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def binary_loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("L")


if __name__ == "__main__":
    train_image_path = "/Users/hhn21/Documents/h2/primus/wall_seg_crop/data_test/images/"
    train_mask_path = "/Users/hhn21/Documents/h2/primus/wall_seg_crop/data_test/masks/"
    result_dir = "result"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    # 1. Load train data
    dataset = FullDataset(train_image_path, train_mask_path, 1344, mode="train")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for i, batch in enumerate(dataloader):
        if i > 100:
            break

        image = batch["image"][0]
        label = batch["label"][0]
        # Denormalize the image
        for channel, m, s in zip(image, mean, std):
            channel.mul_(s).add_(m)
        # Denormalize the label
        for channel, m, s in zip(label, mean, std):
            channel.mul_(s).add_(m)
        # Convert from PyTorch tensor format (C, H, W) to NumPy format (H, W, C)
        np_image = image.numpy()
        np_image = np.transpose(np_image, (1, 2, 0))
        np_label = label.numpy()
        np_label = np.transpose(np_label, (1, 2, 0))

        # Clip values to [0, 1] range and display
        np_image = np.clip(np_image, 0, 1)
        np_label = np.clip(np_label, 0, 1)
        print("============")
        print(np_image.shape)
        print(np_image.max())
        print(np_label.shape)
        print(np_label.max())
        cv2.imwrite(os.path.join(result_dir, f"{i}.jpg"), np_image * 255)
        cv2.imwrite(os.path.join(result_dir, f"label_{i}.jpg"), np_label * 255)
