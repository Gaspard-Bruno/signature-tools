import torch
from .categories import COLOR_CAT
from kornia.color import rgb_to_hls, rgb_to_hsv
import torch

class RBGtoHLS:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {"image": ("IMAGE",)}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = COLOR_CAT

    def process(self, image: torch.Tensor):
        image = image.transpose(3, 1)
        output = rgb_to_hls(image)
        output = output.transpose(3, 1)
        return (output,)

class RGBtoHSV:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {"image": ("IMAGE",)}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = COLOR_CAT

    def process(self, image: torch.Tensor):
        image = image.transpose(3, 1)
        output = rgb_to_hsv(image)
        output = output.transpose(3, 1)
        return (output,)

NODE_CLASS_MAPPINGS = {
    "RGB to HLS": RBGtoHLS,
    "RGBA to HSV": RGBtoHSV,
}