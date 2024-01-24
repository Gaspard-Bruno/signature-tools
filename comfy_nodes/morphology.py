import torch
import cv2
import numpy as np
from .categories import MORPHOLOGY_CAT
from kornia.morphology import erosion, dilation
from  ..src.signature.img.tensor_image import TensorImage

class ImageErosion:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {
            "image": ("IMAGE",),
            "kernel_size": ("INT", {"default": 3}),
            "iterations": ("INT", {"default": 1}),
            }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = MORPHOLOGY_CAT
    def process(self, image: torch.Tensor, kernel_size, iterations):
        step = TensorImage.from_comfy(image)
        kernel_size = max(1, kernel_size)
        kernel = torch.ones(kernel_size, kernel_size).to(step.device)
        for _ in range(iterations):
            step = erosion(tensor=step, kernel=kernel)
        output = TensorImage(step).get_comfy()
        return (output,)

class ImageDilation:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {
            "image": ("IMAGE",),
            "kernel_size": ("INT", {"default": 3}),
            "iterations": ("INT", {"default": 1}),
            }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = MORPHOLOGY_CAT
    def process(self, image: torch.Tensor, kernel_size, iterations):
        step = TensorImage.from_comfy(image)
        kernel_size = max(1, kernel_size)
        kernel = torch.ones(kernel_size, kernel_size).to(step.device)
        for _ in range(iterations):
            step = dilation(tensor=step, kernel=kernel)
        output = TensorImage(step).get_comfy()
        return (output,)


class MaskErosion:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {
            "mask": ("MASK",),
            "kernel_size": ("INT", {"default": 3}),
            "iterations": ("INT", {"default": 1}),
            }}
    RETURN_TYPES = ("MASK",)
    FUNCTION = "process"
    CATEGORY = MORPHOLOGY_CAT
    def process(self, mask: torch.Tensor, kernel_size, iterations):
        step = TensorImage.from_comfy(mask)
        kernel_size = max(1, kernel_size)
        kernel = torch.ones(kernel_size, kernel_size).to(step.device)
        for _ in range(iterations):
            step = erosion(tensor=step, kernel=kernel)
        output = TensorImage(step).get_comfy()
        return (output,)


class MaskDilation:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {
            "mask": ("MASK",),
            "kernel_size": ("INT", {"default": 3}),
            "iterations": ("INT", {"default": 1}),
            }}
    RETURN_TYPES = ("MASK",)
    FUNCTION = "process"
    CATEGORY = MORPHOLOGY_CAT
    def process(self, mask: torch.Tensor, kernel_size, iterations):
        step = TensorImage.from_comfy(mask)
        kernel_size = max(1, kernel_size)
        kernel = torch.ones(kernel_size, kernel_size).to(step.device)
        for _ in range(iterations):
            step = dilation(tensor=step, kernel=kernel)
        output = TensorImage(step).get_comfy()
        return (output,)


class CreateTrimap:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {
            "mask": ("MASK",),
            "inner_min_threshold": ("INT", {"default": 200, "min": 0, "max": 255}),
            "inner_max_threshold": ("INT", {"default": 255, "min": 0, "max": 255}),
            "outer_min_threshold": ("INT", {"default": 15, "min": 0, "max": 255}),
            "outer_max_threshold": ("INT", {"default": 240, "min": 0, "max": 255}),
            "kernel_size": ("INT", {"default": 10, "min": 1, "max": 100}),
            }}
    RETURN_TYPES = ("MASK","TRIMAP")
    FUNCTION = "process"
    CATEGORY = MORPHOLOGY_CAT
    def process(self, mask: torch.Tensor, inner_min_threshold, inner_max_threshold, outer_min_threshold, outer_max_threshold, kernel_size):

        step = TensorImage.from_comfy(mask)
        kernel = torch.ones(kernel_size, kernel_size).to(step.device)

        inner_mask = step.clone()
        inner_mask[inner_mask > (inner_max_threshold / 255.0)] = 1.0
        inner_mask[inner_mask <= (inner_min_threshold / 255.0)] = 0.0

        inner_mask = erosion(tensor=inner_mask, kernel=kernel)

        inner_mask[inner_mask != 0.0] = 1.0

        outter_mask = step.clone()
        outter_mask[outter_mask > (outer_max_threshold / 255.0)] = 1.0
        outter_mask[outter_mask <= (outer_min_threshold / 255.0)] = 0.0

        for _ in range(5):
            outter_mask = dilation(tensor=outter_mask, kernel=kernel)

        outter_mask[outter_mask != 0.0] = 1.0


        trimap_im = torch.zeros_like(step)
        trimap_im[outter_mask == 1.0] = 0.5
        trimap_im[inner_mask == 1.0] = 1.0
        batch_size = step.shape[0]

        trimap = torch.zeros(batch_size, 2, step.shape[2], step.shape[3], dtype=step.dtype, device=step.device)
        for i in range(batch_size):
            tar_trimap = trimap_im[i][0]
            trimap[i][1][tar_trimap == 1] = 1
            trimap[i][0][tar_trimap == 0] = 1


        output_0 = TensorImage(trimap_im).get_comfy()
        output_1 = trimap.permute(0, 2, 3, 1)

        print(output_1.shape)
        return (output_0, output_1,)

NODE_CLASS_MAPPINGS = {
    "Image Erosion": ImageErosion,
    "Image Dilation": ImageDilation,
    "Mask Erosion": MaskErosion,
    "Mask Dilation": MaskDilation,
    "Create Trimap": CreateTrimap,
}