from ..img.tensor_image import TensorImage
from .categories import PLATFROM_IO_CAT
import torch
import os
from datetime import datetime


BASE_COMFY_DIR = os.getcwd().split('custom_nodes')[0]
class AnyType(str):
  def __ne__(self, __value: object) -> bool:
    return False
any = AnyType("*")

class PlatformInputImage():

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {
            "required": {
                "title": ("STRING", {"default": "Input Image"}),
                "short_description": ("STRING", {"default": ""}),
                "subtype": (['image', 'mask'],),
                "required": ("BOOLEAN", {"default": True}),
                "value": ("STRING", {"default": ""}),
                },
                "optional": {"fallback": (any,),}
            }
    RETURN_TYPES = (any,)
    FUNCTION = "apply"
    CATEGORY = PLATFROM_IO_CAT

    def apply(self, value, title:str, short_description:str, subtype: str, required:str, fallback = None):

        if value != "":
            if value.startswith("data:"):
                output = TensorImage.from_base64(value)
            elif value.startswith("http"):
                output = TensorImage.from_web(value)
            else:
                raise ValueError(f"Unsupported input type: {type(value)}")
            if subtype == "mask":
                output = output.get_grayscale()
            else:
                output = output.get_rgb_or_rgba()
            return (output.get_comfy(),)

        if isinstance(fallback, torch.Tensor):
            return (fallback,)

        raise ValueError(f"Unsupported fallback type: {type(fallback)}")

class PlatformInputText():

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {
            "required": {
                "title": ("STRING", {"default": "Input Text"}),
                "short_description": ("STRING", {"default": ""}),
                "subtype": (['string','positive_prompt', 'negative_prompt'],),
                "required": ("BOOLEAN", {"default": True}),
                "value": ("STRING", {"multiline": True, "default": ""}),
                },
            }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "apply"
    CATEGORY = PLATFROM_IO_CAT

    def apply(self, value:str, title:str, short_description:str, subtype: str, required:str):

        if isinstance(value, str):
            return (value,)
        else:
            raise ValueError(f"Unsupported input type: {type(value)}")

class PlatformInputSelector():

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {
            "required": {
                "selected": ("INT", {"default": 0, "min": 0}),
                "input_0": ("STRING", {"default": ""}),
                },
            }
    RETURN_TYPES = (any,)
    FUNCTION = "apply"
    CATEGORY = PLATFROM_IO_CAT

    def apply(self, **kwargs):
        items = [value for key, value in kwargs.items() if key.startswith("input_")]
        item = items[kwargs["selected"]]
        return (item,)

class PlatformInputNumber():
    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {
            "required": {
                "title": ("STRING", {"default": "Input Number"}),
                "short_description": ("STRING", {"default": ""}),
                "subtype": (['float','int'],),
                "required": ("BOOLEAN", {"default": True}),
                "value": ("FLOAT", {"default": 0}),
                },
            }
    RETURN_TYPES = (any,)
    FUNCTION = "apply"
    CATEGORY = PLATFROM_IO_CAT

    def apply(self, value:float, title:str, short_description:str, subtype: str, required:str):
        return (value,)


class PlatformOutput():

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {
            "required": {
                "title": ("STRING", {"default": "Output Image"}),
                "short_description": ("STRING", {"default": ""}),
                "subtype": (['image', 'mask', 'int', 'float', 'string'],),
                "value": (any,),
                },
            }
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "apply"
    CATEGORY = PLATFROM_IO_CAT

    def apply(self, value, title:str, short_description:str, subtype:str):
        supported_types = ["image", "mask", "int", "float", "string"]
        if subtype not in supported_types:
            raise ValueError(f"Unsupported output type: {subtype}")
        results = []
        if subtype == "image" or subtype == "mask":

            output_dir = os.path.join(BASE_COMFY_DIR, 'output')
            tensor_images = TensorImage.from_comfy(value)
            for img in tensor_images:

                current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                random_str = str(torch.randint(0, 100000, (1,)).item())
                file_name = f"signature_{current_time_str}_{random_str}.png"
                save_path = os.path.join(output_dir, file_name)
                output_img = TensorImage(img)
                thumbnail_img = output_img.get_resized(768)
                is_image_saved = output_img.save(save_path)
                is_thumbnail_saved = thumbnail_img.save(save_path.replace(".png", "_thumbnail.jpeg"))
                if is_image_saved and is_thumbnail_saved:
                    output = {
                        "title": title,
                        "short_description": short_description,
                        "type": "image",
                        "value": file_name,
                        "thumbnail": file_name.replace(".png", "_thumbnail.jpeg")
                    }
                    results.append(output)
        else:
            output = {
                "title": title,
                "short_description": short_description,
                "type": "text" if subtype == "string" else "number",
                "value": value
            }
            results.append(output)


        return  { "ui": {"signature_output": results} }


NODE_CLASS_MAPPINGS = {
    "signature_input_image": PlatformInputImage,
    "signature_input_text": PlatformInputText,
    "signature_input_number": PlatformInputNumber,
    # "signature_input_selector": PlatformInputSelector,

    "signature_output": PlatformOutput,
}