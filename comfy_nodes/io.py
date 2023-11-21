from  ..src.signature.img.image_array import ImageArray
from .helper import *
from .categories import IO_CAT
import base64 as b64

def image_array_to_tensor(image_array):
    img_numpy = image_array.get_value()
    image = img_np_to_tensor([img_numpy])
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
        padding = '=' * (4 - len(base64) % 4)
        print(len(padding))
        decoded_bytes = b64.b64decode(base64 + padding)
        img_arr = ImageArray.from_bytes(decoded_bytes)
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
        images_array = img_tensor_to_np(image)
        output = []
        for image_arr in images_array:
            image = ImageArray(image_arr)
            print(image.shape)
            b_data = image.get_bytes()
            base64_encoded = b64.b64encode(b_data).decode('utf-8')
            output.append(base64_encoded)
        print(output)
        return (output,)


NODE_CLASS_MAPPINGS = {
    "Image from Web": ImageFromWeb,
    "Image from Base64": ImageFromBase64,
    "Base64 from Image": Base64FromImage,
}