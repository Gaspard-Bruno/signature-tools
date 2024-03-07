from ..img.tensor_image import TensorImage
from .categories import PLATFROM_IO_CAT
import torch


class PlatformInput():

    def __init__(self):
        self.id = None
        self.is_required = True

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {
            "required": {
                "id": ("STRING", {"default": "ID HERE"}),
                "value": ("STRING", {"default": ""}),
                "value_type": (['url', 'base64'], {"default": "url"}),
                "is_required": ("BOOLEAN", {"default": True})
                },
            "optional": {
                "fallback": ("IMAGE", {"default": None}),
                }
            }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = PLATFROM_IO_CAT

    def apply(self, id: str, is_required: bool, value: str = "", value_type: str = "url", fallback: torch.Tensor|None = None):
        self.id = id
        self.is_required = is_required
        if value == "" and fallback is None:
            raise ValueError("No input image provided")
        if value == "" and fallback is not None:
            tensor_image = TensorImage.from_comfy(fallback)
        else:
            if value_type == 'url':
                tensor_image = TensorImage.from_web(value)
            elif value_type == 'base64':
                tensor_image = TensorImage.from_base64(value)
            else:
                raise ValueError("Invalid value type")

        output_image = TensorImage(tensor_image).get_comfy()
        return (output_image,)

class PlatformOutput():

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {
            "required": {
                "id": ("STRING", {"default": "ID HERE"}),
                "image": ("IMAGE",)
                },
            }
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "apply"
    CATEGORY = PLATFROM_IO_CAT

    def apply(self, id:str, image: torch.Tensor):
        results = []
        tensor_images = TensorImage.from_comfy(image)

        for img in tensor_images:
            b64_output = TensorImage(img).get_base64()
            results.append({"id": id, "type": "image", "value": b64_output})

        return { "ui": {"signature_output": results} }



NODE_CLASS_MAPPINGS = {
    "🔵 Platform Input": PlatformInput,
    "🔵 Platform Output": PlatformOutput,
}