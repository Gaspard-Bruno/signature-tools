from ..img.tensor_image import TensorImage
from .categories import PLATFROM_IO_CAT
import torch

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

class PlatformInputNumber():
    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {
            "required": {
                "title": ("STRING", {"default": "Input Number"}),
                "short_description": ("STRING", {"default": ""}),
                "subtype": (['float','int'],),
                "required": ("BOOLEAN", {"default": True}),
                "value": (any,),
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
        results = []
        if subtype == "image" or subtype == "mask":
            tensor_images = TensorImage.from_comfy(value)
            for img in tensor_images:
                b64_output = TensorImage(img).get_base64()
                output = {
                    "title": title,
                    "short_description": short_description,
                    "type": "image",
                    "value": str(b64_output)
                }

                results.append(output)
            return  { "ui": {"signature_output": results} }

        elif subtype == "int" or subtype == "float" or subtype == "string":
            output = {
                "title": title,
                "short_description": short_description,
                "type": "text" if subtype == "string" else "number",
                "value": str(value)
            }
            results.append(output)
            return  { "ui": {"signature_output": results} }

        raise ValueError(f"Unsupported output type: {subtype}")


NODE_CLASS_MAPPINGS = {
    "signature_input_image": PlatformInputImage,
    "signature_input_text": PlatformInputText,
    "signature_input_number": PlatformInputNumber,

    "signature_output": PlatformOutput,
}