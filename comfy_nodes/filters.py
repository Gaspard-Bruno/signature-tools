import torch
from . import helper
from .categories import FILTER_CAT
from kornia.filters import gaussian_blur2d, unsharp_mask
import torch

class ImageGaussianBlur:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {"image": ("IMAGE",),
                             "radius": ("INT", {"default": 13}),
                             "sigma": ("FLOAT", {"default": 10.5}),
                             "interations": ("INT", {"default": 1}),
                             }
                }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = FILTER_CAT

    def process(self, image: torch.Tensor, radius, sigma, interations):
        if radius % 2 == 0:
            radius += 1
        in_kernel_size = (radius, radius)
        in_sigma = (sigma, sigma)
        step = helper.comfy_img_to_torch(image)

        interations = max(1, interations)
        for _ in range(interations):
            step = gaussian_blur2d(step,
                                    kernel_size=in_kernel_size,
                                    sigma=in_sigma,
                                    border_type='reflect',
                                    separable=True)
        output = helper.torch_img_to_comfy(step)
        return (output,)

class ImageUnsharpMask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {"image": ("IMAGE",),
                             "radius": ("INT", {"default": 3}),
                             "sigma": ("FLOAT", {"default": 1.5}),
                             "interations": ("INT", {"default": 1}),
                             }
                }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = FILTER_CAT

    def process(self, image: torch.Tensor, radius, sigma, interations):
        if radius % 2 == 0:
            radius += 1
        in_kernel_size = (radius, radius)
        in_sigma = (sigma, sigma)
        step = helper.comfy_img_to_torch(image)
        interations = max(1, interations)
        for _ in range(interations):
            step = unsharp_mask(step, kernel_size=in_kernel_size, sigma=in_sigma)
        output = helper.torch_img_to_comfy(step)
        return (output,)


class MaskGaussianBlur:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {"image": ("MASK",),
                             "radius": ("INT", {"default": 13}),
                             "sigma": ("FLOAT", {"default": 10.5}),
                             "interations": ("INT", {"default": 1}),
                             }
                }
    RETURN_TYPES = ("MASK",)
    FUNCTION = "process"
    CATEGORY = FILTER_CAT

    def process(self, image: torch.Tensor, radius, sigma, interations):
        if radius % 2 == 0:
            radius += 1
        in_kernel_size = (radius, radius)
        in_sigma = (sigma, sigma)
        step = helper.comfy_mask_to_torch(image)

        interations = max(1, interations)
        for _ in range(interations):
            step = gaussian_blur2d(step,
                                    kernel_size=in_kernel_size,
                                    sigma=in_sigma,
                                    border_type='reflect',
                                    separable=True)
        output = helper.torch_mask_to_comfy(step)
        return (output,)

NODE_CLASS_MAPPINGS = {
    "Image Gaussian Blur": ImageGaussianBlur,
    "Image Unsharp Mask": ImageUnsharpMask,
    "Mask Gaussian Blur": MaskGaussianBlur,
}