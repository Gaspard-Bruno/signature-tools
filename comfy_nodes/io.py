from  ..src.signature.img.image_array import ImageArray
from .helper import *
from .categories import IO_CAT

def image_array_to_tensor(image_array):
    img_numpy = image_array.get_value()
    image = np_to_tensor([img_numpy])
    image_shape = image.shape
    mask_shape = image_shape[:-1]
    mask = torch.ones(mask_shape)
    if image.shape[-1] == 4:
        mask = image[:, :, :, -1]
    return (image, mask, )


class ImageFromWeb():

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {"url": ("STRING", {"default": "URL HERE"})}}
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "process"
    CATEGORY = IO_CAT

    def process(self, url: str):
        img_arr = ImageArray.from_web(url)
        return image_array_to_tensor(img_arr)

class ImageFromBase64():

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {"base64": ("STRING", {"default": "BASE64 HERE"})}}
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "process"
    CATEGORY = IO_CAT

    def process(self, base64: str):
        img_arr = ImageArray.from_base64(base64)
        return image_array_to_tensor(img_arr)

class Base64FromImage():

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {"image": ("IMAGE",)}}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    CATEGORY = IO_CAT
    OUTPUT_NODE = True

    def process(self, image):
        images_array = tensor_to_np(image)
        output = []
        for image_arr in images_array:
            base64_encoded = ImageArray(image_arr).get_base64()
            output.append(base64_encoded)
        return (output,)


NODE_CLASS_MAPPINGS = {
    "Image from Web": ImageFromWeb,
    "Image from Base64": ImageFromBase64,
    "Base64 from Image": Base64FromImage,
}