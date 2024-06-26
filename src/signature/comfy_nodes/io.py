from ..img.tensor_image import TensorImage
import torch
from .categories import IO_CAT
import os
import json

BASE_COMFY_DIR = os.getcwd().split('custom_nodes')[0]

def image_array_to_tensor(x: TensorImage):
    image = x.get_comfy()
    mask = torch.ones((x.shape[0],
                       1,
                       x.shape[2],
                       x.shape[3]),
                      dtype=torch.float32)
    if x.shape[1] == 4:
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
        img_arr = TensorImage.from_web(url)
        return image_array_to_tensor(img_arr)

class ImageFromBase64():

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {"base64": ("STRING", {"default": "BASE64 HERE"})}}
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "process"
    CATEGORY = IO_CAT

    def process(self, base64: str):
        img_arr = TensorImage.from_base64(base64)
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
        images = TensorImage.from_comfy(image)
        output = images.get_base64()
        return (output,)

class LoadFile():

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {
            "required": {
                "value": ('STRING', {'default': ''}),
            },
    }

    RETURN_TYPES = ("FILE",)
    FUNCTION = "process"
    CATEGORY = IO_CAT

    def process(self, value: str):
        data = value.split('&&') if '&&' in value else [value]
        input_folder = os.path.join(BASE_COMFY_DIR, "input")
        for i in range(len(data)):
            json_str = data[i]
            data[i] = json.loads(json_str)
            item = data[i]
            if isinstance(item, dict):
                name = item.get('name', None)
                if name is None:
                    continue
                item['name'] = os.path.join(input_folder, name)
                data[i] = item

        return (data,)

class LoadFolder():

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {
            "required": {
                "value": ('STRING', {'default': ''}),
            },
    }

    RETURN_TYPES = ("FILE",)
    FUNCTION = "process"
    CATEGORY = IO_CAT

    def process(self, value: str):
        data = value.split('&&') if '&&' in value else [value]
        input_folder = os.path.join(BASE_COMFY_DIR, "input")
        for i in range(len(data)):
            json_str = data[i]
            data[i] = json.loads(json_str)
            item = data[i]
            if isinstance(item, dict):
                name = item.get('name', None)
                if name is None:
                    continue
                item['name'] = os.path.join(input_folder, name)
                data[i] = item
        return (data,)

NODE_CLASS_MAPPINGS = {
    "Image from Web": ImageFromWeb,
    "Image from Base64": ImageFromBase64,
    "Base64 from Image": Base64FromImage,
    "Load File": LoadFile,
    "Load Folder": LoadFolder,
}