from .utils import *
from .categories import MORPHOLOGY_CAT
from kornia.morphology import erosion, dilation
from . import helper

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
        step = helper.comfy_img_to_torch(image)
        kernel_size = max(1, kernel_size)
        kernel = torch.ones(kernel_size, kernel_size)
        for _ in range(iterations):
            step = erosion(tensor=step, kernel=kernel)
        output = helper.torch_img_to_comfy(step)
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
        step = helper.comfy_img_to_torch(image)
        kernel_size = max(1, kernel_size)
        kernel = torch.ones(kernel_size, kernel_size)
        for _ in range(iterations):
            step = dilation(tensor=step, kernel=kernel)
        output = helper.torch_img_to_comfy(step)
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
        step = helper.comfy_mask_to_torch(mask)
        kernel_size = max(1, kernel_size)
        kernel = torch.ones(kernel_size, kernel_size)
        for _ in range(iterations):
            step = erosion(tensor=step, kernel=kernel)
        output = helper.torch_mask_to_comfy(step)
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
        step = helper.comfy_mask_to_torch(mask)
        kernel_size = max(1, kernel_size)
        kernel = torch.ones(kernel_size, kernel_size)
        for _ in range(iterations):
            step = dilation(tensor=step, kernel=kernel)
        output = helper.torch_mask_to_comfy(step)
        return (output,)


NODE_CLASS_MAPPINGS = {
    "Image Erosion": ImageErosion,
    "Image Dilation": ImageDilation,
    "Mask Erosion": MaskErosion,
    "Mask Dilation": MaskDilation,
}