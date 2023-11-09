from signature.img.image_array import ImageArray

class IMAGE_FROM_WEB_Preprocessor:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"url": ("STRING", {"default": "URL HERE"})}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "image_from_web"
    CATEGORY = "image/load_from_url"
    def image_from_web(self, url):
        return ImageArray.from_web(url).get_float_value()


NODE_CLASS_MAPPINGS = {
    "IMAGE_FROM_WEB_Preprocessor": IMAGE_FROM_WEB_Preprocessor,
}