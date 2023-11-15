from .utils import *
from .categories import FILTER_CAT
from kornia.filters import gaussian_blur2d, unsharp_mask, laplacian
import torch

class GaussianBlur:
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
        step = image.transpose(3, 1)
        interations = max(1, interations)
        for _ in range(interations):
            step = gaussian_blur2d(step,
                                    kernel_size=in_kernel_size,
                                    sigma=in_sigma,
                                    border_type='reflect',
                                    separable=True)
        output = step.transpose(3, 1)
        return (output,)

class UnsharpMask:
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
        step = image.transpose(3, 1)
        interations = max(1, interations)
        for _ in range(interations):
            step = unsharp_mask(step, kernel_size=in_kernel_size, sigma=in_sigma)
        output = step.transpose(3, 1)
        return (output,)

class Laplacian:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {"image": ("IMAGE",),
                             "kernel_width": ("INT", {"default": 3}),
                             }
                }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = FILTER_CAT

    def process(self, image: torch.Tensor, radius):
        if radius % 2 == 0:
            radius += 1
        in_kernel_size = (radius, radius)
        image = image.transpose(3, 1)
        output = laplacian(image, kernel_size=in_kernel_size)
        output = output.transpose(3, 1)
        return (output,)

NODE_CLASS_MAPPINGS = {
    "Gaussian Blur": GaussianBlur,
    "Unsharp Mask": UnsharpMask,
    "Laplacian": Laplacian,
}