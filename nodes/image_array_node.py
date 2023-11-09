from  ..src.signature.img.image_array import ImageArray
from .utils import img_np_to_tensor

class ImageFromWeb:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {"url": ("STRING", {"default": "URL HERE"})}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "image_from_web"
    CATEGORY = "Signature/Image"
    def image_from_web(self, url):
        np_array = ImageArray.from_web(url).get_float_value
        return img_np_to_tensor(np_array)


NODE_CLASS_MAPPINGS = {
    "Load from Web": ImageFromWeb,
}