from  ..src.signature.img.image_array import ImageArray
from . import helper
from .categories import MISC_CAT
import torch


class OnesLike():

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {"image": ("IMAGE",)}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = MISC_CAT

    def process(self, image: torch.Tensor):
        input_image = helper.comfy_img_to_torch(image)
        step = torch.ones_like(input_image)
        output_image = helper.torch_img_to_comfy(step)
        return (output_image,)

class ZerosLike():

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {"image": ("IMAGE",)}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = MISC_CAT

    def process(self, image: torch.Tensor):
        input_image = helper.comfy_img_to_torch(image)
        step = torch.zeros_like(input_image)
        output_image = helper.torch_img_to_comfy(step)
        return (output_image,)

NODE_CLASS_MAPPINGS = {
    "Ones Like": OnesLike,
    "Zeros Like": ZerosLike,
}