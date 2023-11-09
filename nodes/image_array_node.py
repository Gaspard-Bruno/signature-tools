from  ..src.signature.img.image_array import ImageArray
from .utils import *

class ImageFromWeb:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {"url": ("STRING", {"default": "URL HERE"})}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "image_from_web"
    CATEGORY = "🔵 Signature.ai Tools/🖼️ Image"
    def image_from_web(self, url):
        img_arr = ImageArray.from_web(url)
        image = img_np_to_tensor([img_arr.get_value()])
        return (image,)


NODE_CLASS_MAPPINGS = {
    "Load from Web": ImageFromWeb,
}