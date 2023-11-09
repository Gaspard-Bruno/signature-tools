from .utils import *
from .categories import FILTER_CAT
from kornia.filters import gaussian_blur2d, canny
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

    def process(self, image: torch.Tensor, kernel_width, kernel_height, sigma_x, sigma_y):
        if kernel_width % 2 == 0:
            kernel_width += 1
        if kernel_height % 2 == 0:
            kernel_height += 1
        in_kernel_size = (kernel_width, kernel_height)
        in_sigma = (sigma_x, sigma_y)
        image = image.transpose(3, 1)
        output = gaussian_blur2d(image,
                                 kernel_size=in_kernel_size,
                                 sigma=in_sigma,
                                 border_type='reflect',
                                 separable=True)
        output = output.transpose(3, 1)
        return (output,)

class CannyEdge:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {"image": ("IMAGE",),
                             "low_threshold": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                             "high_threshold": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                             }
                }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = FILTER_CAT

    def process(self, image: torch.Tensor, low_threshold, high_threshold):
        image = image.transpose(3, 1)
        _, output = canny(image, low_threshold=low_threshold, high_threshold=high_threshold)
        #output = tensor_to_image(x_canny.byte())
        output = output[0].transpose(0,2).transpose(0,1)
        return (output,)


NODE_CLASS_MAPPINGS = {
    "Gaussian Blur": GaussianBlur,
    "Canny Edge": CannyEdge,
}