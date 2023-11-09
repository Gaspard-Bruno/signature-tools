from .utils import *
from .categories import COLOR_CAT
from kornia.color import rgb_to_grayscale, rgba_to_rgb
import torch

class RBG2Gray:
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
        output = rgb_to_grayscale(image)
        output = output.transpose(3, 1)
        return (output,)

class RGBA2RGB:
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
        output = rgba_to_rgb(image)
        output = output.transpose(3, 1)
        return (output,)

NODE_CLASS_MAPPINGS = {
    "RGB to Grayscale": RBG2Gray,
    "RGBA to RGB": RGBA2RGB,
}