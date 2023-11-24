import torch
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


NODE_CLASS_MAPPINGS = {
    "Image Erosion": ImageErosion,
    "Image Dilation": ImageDilation,
    "Mask Erosion": MaskErosion,
    "Mask Dilation": MaskDilation,
}