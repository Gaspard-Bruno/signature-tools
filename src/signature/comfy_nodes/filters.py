import torch
from ..img.tensor_image import TensorImage
from .categories import FILTER_CAT
from kornia.filters import gaussian_blur2d, unsharp_mask
import torch

class ImageGaussianBlur:


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
        step = TensorImage.from_comfy(image)
        interations = max(1, interations)
        for _ in range(interations):
            step = gaussian_blur2d(step,
                                    kernel_size=in_kernel_size,
                                    sigma=in_sigma,
                                    border_type='reflect',
                                    separable=True)
        output = TensorImage(step).get_comfy()
        return (output,)

class ImageUnsharpMask:


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
        step = TensorImage.from_comfy(image)
        interations = max(1, interations)
        for _ in range(interations):
            step = unsharp_mask(step, kernel_size=in_kernel_size, sigma=in_sigma)
        output = TensorImage(step).get_comfy()
        return (output,)


class MaskGaussianBlur:


    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {"image": ("MASK",),
                             "radius": ("INT", {"default": 13}),
                             "sigma": ("FLOAT", {"default": 10.5}),
                             "interations": ("INT", {"default": 1}),
                             "only_outline": ("BOOLEAN", {"default": False}),
                             }
                }
    RETURN_TYPES = ("MASK",)
    FUNCTION = "process"
    CATEGORY = FILTER_CAT

    def process(self, image: torch.Tensor, radius:int, sigma:float, interations:int, only_outline:bool):
        if radius % 2 == 0:
            radius += 1
        in_kernel_size = (radius, radius)
        in_sigma = (sigma, sigma)
        step = TensorImage.from_comfy(image)

        interations = max(1, interations)
        for _ in range(interations):
            blurred = gaussian_blur2d(step,
                                    kernel_size=in_kernel_size,
                                    sigma=in_sigma,
                                    border_type='reflect',
                                    separable=True)
            step = torch.where(step == 0, blurred, step) if only_outline else blurred
        output = TensorImage(step).get_comfy()
        return (output,)

NODE_CLASS_MAPPINGS = {
    "Image Gaussian Blur": ImageGaussianBlur,
    "Image Unsharp Mask": ImageUnsharpMask,
    "Mask Gaussian Blur": MaskGaussianBlur,
}