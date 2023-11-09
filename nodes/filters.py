from .utils import *
from .categories import FILTER_CAT
from kornia.filters import gaussian_blur2d
import torch

class GaussianBlur:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {"image": ("IMAGE",),
                             "kernel_width": ("INT", {"default": 3}),
                             "sigma_height": ("INT", {"default": 3}),
                             "sigma_x": ("FLOAT", {"default": 1.5}),
                             "sigma_y": ("FLOAT", {"default": 1.5}),
                             }
                }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = FILTER_CAT

    def process(self, image: torch.Tensor, kernel_width, sigma_height, sigma_x, sigma_y):
        image = image.transpose(3, 1)
        output = gaussian_blur2d(image, kernel_size=(kernel_width, kernel_width), sigma=(sigma_x, sigma_y))
        output = output.transpose(3, 1)
        return (output,)


NODE_CLASS_MAPPINGS = {
    "Gaussian Blur": GaussianBlur,
}