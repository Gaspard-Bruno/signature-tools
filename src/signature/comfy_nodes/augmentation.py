import albumentations as A
import torch
import numpy as np
import random
from .categories import AUGMENTATION_CAT
from ..img.tensor_image import TensorImage


class RandomCropAugmentation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                    "height": ("INT", {"default": 1024, "min": 32, "step": 32}),
                    "width": ("INT", {"default": 1024, "min": 32, "step": 32}),
                    "min_window": ("INT", {"default": 256, "step": 32}),
                    "max_window": ("INT", {"default": 1024, "step": 32}),
                    "percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
            },
            "optional": {
                "augmentation": ("AUGMENTATION", {"default": None}),
            }
        }

    RETURN_TYPES = ("AUGMENTATION",)
    RETURN_NAMES = ("augmentation",)
    FUNCTION = "process"
    CATEGORY = AUGMENTATION_CAT

    def process(self, height: int, width: int, min_window:int, max_window:int, percent: float, augmentation: list | None = None,):
        if augmentation is None:
            augmentation = []
        augmentation.append(A.RandomSizedCrop(min_max_height=(min_window,max_window), height=height, width=width, p=percent))
        return (augmentation,)

class FlipAugmentation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flip": (["horizontal", "vertical"], {"default": "horizontal"}),
                "percent": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
            },
            "optional": {
                "augmentation": ("AUGMENTATION", {"default": None}),
            }
        }

    RETURN_TYPES = ("AUGMENTATION",)
    RETURN_NAMES = ("augmentation",)
    FUNCTION = "process"
    CATEGORY = AUGMENTATION_CAT

    def process(self, flip:str, percent: float, augmentation: list | None = None):
        if augmentation is None:
            augmentation = []
        if percent is None:
            percent = 0.5
        if flip == "horizontal":
            augmentation.append(A.HorizontalFlip(p=percent))
        else:
            augmentation.append(A.VerticalFlip(p=percent))
        return (augmentation,)

class ComposeAugmentation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "augmentation": ("AUGMENTATION",),
                "height": ("INT", {"default": 1024, "min": 32, "step": 32}),
                "width": ("INT", {"default": 1024, "min": 32, "step": 32}),
                "samples": ("INT", {"default": 1, "min": 1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 10000000000000000}),
            },
            "optional": {
                "image": ("IMAGE", {"default": None}),
                "mask": ("MASK", {"default": None}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    FUNCTION = "process"
    CATEGORY = AUGMENTATION_CAT

    def process(self, augmentation, height: int, width: int, samples: int, image: torch.Tensor | None = None, mask: torch.Tensor | None = None, seed: int = -1):
        image_tensor = TensorImage.from_comfy(image) if image is not None else None
        mask_tensor = TensorImage.from_comfy(mask) if mask is not None else None
        if image_tensor is None and mask_tensor is None:
            raise ValueError("Either image or mask must be provided")

        np_images = image_tensor.get_numpy_image() if image_tensor is not None else None
        np_masks = mask_tensor.get_numpy_image() if mask_tensor is not None else None

        if np_images is not None and len(np_images.shape) == 3:
            np_images = np.expand_dims(np_images, axis=0)

        if np_masks is not None and len(np_masks.shape) == 2:
            np_masks = np.expand_dims(np_masks, axis=0)
            np_masks = np.expand_dims(np_masks, axis=-1)

        if np_images is not None and np_masks is not None:
            if np_images.shape[0] != np_masks.shape[0]:
                raise ValueError("Number of images and masks must be the same")
            if np_images.shape[1] != np_masks.shape[1] or np_images.shape[2] != np_masks.shape[2]:
                raise ValueError("Image and mask dimensions must be the same")


        if seed == -1:
            seed = np.random.randint(0, 10000000000000000)

        total_elements = len(np_images) if np_images is not None else len(np_masks) if np_masks is not None else 0
        total_images = None
        total_masks = None

        for sample in range(samples):
            seed_for_sample = seed + sample
            random.seed(seed_for_sample)
            np.random.default_rng(seed_for_sample)
            transform = A.Compose(augmentation)

            images = np.ones((np_images.shape[0],height, width, np_images.shape[3])) if np_images is not None else None
            masks = np.ones((np_masks.shape[0],height, width, np_masks.shape[3])) if np_masks is not None else None

            for idx in range(total_elements):
                new_image = None
                new_mask = None
                if images is not None and np_images is not None:
                    np_image = np_images[idx]
                    transformed = transform(image=np_image)
                    new_image = transformed["image"]
                    if new_image.shape[0] != images.shape[1] or new_image.shape[2] != images.shape[1]:
                        new_image = A.Resize(height=images.shape[1], width=images.shape[2])(image=new_image)["image"]
                    images[idx] = new_image

                if masks is not None and np_masks is not None:
                    np_mask = np_masks[idx]
                    transformed = transform(mask=np_mask)
                    new_mask = transformed["mask"]
                    if new_mask.shape[0] != masks.shape[1] or new_mask.shape[2] != masks.shape[1]:
                        new_mask = A.Resize(height=masks.shape[1], width=masks.shape[2])(mask=new_mask)["mask"]
                    masks[idx] = new_mask

            if images is not None:
                total_images = np.concatenate((images, total_images), axis=0) if total_images is not None else images
            if masks is not None:
                total_masks = np.concatenate((masks, total_masks), axis=0) if total_masks is not None else masks

        comfy_image = TensorImage.from_numpy(total_images).get_comfy() if total_images is not None else None
        comfy_mask = TensorImage.from_numpy(total_masks).get_comfy() if total_masks is not None else None

        return (comfy_image, comfy_mask, )

NODE_CLASS_MAPPINGS = {
    "ComposeAug": ComposeAugmentation,
    "RandomCropAug": RandomCropAugmentation,
    "FlipAug": FlipAugmentation,
}