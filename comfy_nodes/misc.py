from  ..src.signature.img.tensor_image import TensorImage
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
        input_image = TensorImage.from_comfy(image)
        step = torch.ones_like(input_image)
        output_image = TensorImage(step).get_comfy()
        return (output_image,)

class ZerosLike():

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {"image": ("IMAGE",)}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = MISC_CAT

    def process(self, image: torch.Tensor):
        input_image = TensorImage.from_comfy(image)
        step = torch.zeros_like(input_image)
        output_image = TensorImage(step).get_comfy()
        return (output_image,)

NODE_CLASS_MAPPINGS = {
    "Ones Like": OnesLike,
    "Zeros Like": ZerosLike,
}