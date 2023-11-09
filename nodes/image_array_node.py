from  ..src.signature.img.image_array import ImageArray
from .utils import *
from .categories import *
from kornia.enhance import adjust_brightness, adjust_hue, adjust_saturation
import torch
class ImageFromWeb:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {"url": ("STRING", {"default": "URL HERE"})}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = IMAGE_CAT
    def process(self, url):
        img_arr = ImageArray.from_web(url)
        image = img_np_to_tensor([img_arr.get_value()])
        return (image,)

class AdjustBrightness:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {"image": ("IMAGE",),
                             "factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                            }
                }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = IMAGE_CAT

    def process(self, image, factor):
        output = adjust_brightness(image, factor, clip_output=True)
        return (output,)


class AdjustSaturation:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {"image": ("IMAGE",),
                             "factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                            }
                }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = IMAGE_CAT

    def process(self, image, factor):
        output = adjust_saturation(image, factor)
        return (output,)

NODE_CLASS_MAPPINGS = {
    "Load from Web": ImageFromWeb,
    "Adjust Brightness": AdjustBrightness,
    "Adjust Saturation": AdjustSaturation,
}