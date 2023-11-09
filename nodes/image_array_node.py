from  ..src.signature.img.image_array import ImageArray
from .utils import *
from .categories import *
from kornia.enhance import adjust_brightness
import torch
class ImageFromWeb:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {"url": ("STRING", {"default": "URL HERE"})}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "image_from_web"
    CATEGORY = IMAGE_CAT
    def image_from_web(self, url):
        img_arr = ImageArray.from_web(url)
        image = img_np_to_tensor([img_arr.get_value()])
        return (image,)

class AdjustBrightness:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {"image": ("IMAGE"), "factor": ("FLOAT", {"default": 0.5})}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "img_adjust_brightness"
    CATEGORY = IMAGE_CAT

    def img_adjust_brightness(self, image, factor):
        output = adjust_brightness(image, np.clip(factor, 0.0, 1.0), clip_output=True)
        output = torch.stack([output])
        return (output,)


NODE_CLASS_MAPPINGS = {
    "Load from Web": ImageFromWeb,
    "Adjust Brightness": AdjustBrightness,
}