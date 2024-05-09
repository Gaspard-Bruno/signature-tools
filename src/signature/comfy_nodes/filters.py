import torch
from ..img.tensor_image import TensorImage
from .categories import FILTER_CAT
from kornia.filters import gaussian_blur2d, unsharp_mask
from kornia.color import rgb_to_hsv, rgb_to_hls
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

class ImageSoftLight:


    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {"top": ("IMAGE",),
                             "bottom": ("IMAGE",),
                             }
                }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = FILTER_CAT

    def process(self, top: torch.Tensor, bottom:torch.Tensor):
        top_tensor = TensorImage.from_comfy(top)
        bottom_tensor = TensorImage.from_comfy(bottom)
        low_mask = top_tensor < 0.5
        high_mask = ~low_mask

        blend_low = 2 * bottom_tensor * top_tensor + bottom_tensor.pow(2) * (1 - 2 * top_tensor)
        blend_high = 2 * bottom_tensor * (1 - top_tensor) + torch.sqrt(bottom_tensor) * (2 * top_tensor - 1)
        blend = torch.zeros_like(bottom_tensor)
        blend[low_mask] = blend_low[low_mask]
        blend[high_mask] = blend_high[high_mask]

        output = TensorImage(blend).get_comfy()

        return (output,)

class ImageHSV:


    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {"image": ("IMAGE",),
                             }
                }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = FILTER_CAT

    def process(self, image: torch.Tensor):
        image_tensor = TensorImage.from_comfy(image)
        output = TensorImage(rgb_to_hsv(image_tensor)).get_comfy()
        return (output,)

class ImageHLS:
    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {"image": ("IMAGE",),
                             }
                }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = FILTER_CAT

    def process(self, image: torch.Tensor):
        image_tensor = TensorImage.from_comfy(image)
        output = TensorImage(rgb_to_hls(image_tensor)).get_comfy()
        return (output,)

class ImageAverage:
    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {"image": ("IMAGE",),
                             }
                }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = FILTER_CAT

    def process(self, image: torch.Tensor):
        step = TensorImage.from_comfy(image)
        output = step.mean(dim=0, keepdim=True)
        output = TensorImage(output).get_comfy()
        return (output,)

NODE_CLASS_MAPPINGS = {
    "Image Gaussian Blur": ImageGaussianBlur,
    "Image Unsharp Mask": ImageUnsharpMask,
    "Mask Gaussian Blur": MaskGaussianBlur,
    "Image Soft Light": ImageSoftLight,
    "Image HSV": ImageHSV,
    "Image HLS": ImageHLS,
    "Image Average": ImageAverage,
}