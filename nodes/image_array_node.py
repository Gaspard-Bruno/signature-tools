from  ..src.signature.img.image_array import ImageArray

class ImageFromWeb:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {"url": ("STRING", {"default": "URL HERE"})}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "Load from Web"
    CATEGORY = "Signature/Image"
    def image_from_web(self, url):
        return ImageArray.from_web(url).get_float_value()


NODE_CLASS_MAPPINGS = {
    "Load from Web": ImageFromWeb,
}