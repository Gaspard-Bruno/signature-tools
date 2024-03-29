from ..img.tensor_image import TensorImage
from .categories import MISC_CAT
import torch



class Bitwise():

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {"mask_1": ("MASK",), "mask_2": ("MASK",), "mode": (['and', 'or', 'xor', 'left_shift', 'right_shift'],),},}
    RETURN_TYPES = ("MASK",)
    FUNCTION = "process"
    CATEGORY = MISC_CAT

    def process(self, mask_1: torch.Tensor, mask_2: torch.Tensor, mode: str):
        input_mask_1 = TensorImage.from_comfy(mask_1)
        input_mask_2 = TensorImage.from_comfy(mask_2)
        eight_bit_mask_1 = torch.tensor(input_mask_1 * 255, dtype=torch.uint8)
        eight_bit_mask_2 = torch.tensor(input_mask_2 * 255, dtype=torch.uint8)

        if mode == "and":
            result = torch.bitwise_and(eight_bit_mask_1, eight_bit_mask_2)
        elif mode == "or":
            result = torch.bitwise_or(eight_bit_mask_1, eight_bit_mask_2)
        elif mode == "xor":
            result = torch.bitwise_xor(eight_bit_mask_1, eight_bit_mask_2)
        elif mode == "left_shift":
            result = torch.bitwise_left_shift(eight_bit_mask_1, eight_bit_mask_2)
        elif mode == "right_shift":
            result = torch.bitwise_right_shift(eight_bit_mask_1, eight_bit_mask_2)
        else:
            raise ValueError("Invalid mode")

        float_result = result.float() / 255
        output_mask = TensorImage(float_result).get_comfy()
        return (output_mask,)


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

class MaskBinaryFilter():
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {
            "mask": ("MASK",),
            }}
    RETURN_TYPES = ("MASK",)
    FUNCTION = "process"
    CATEGORY = MISC_CAT
    def process(self, mask: torch.Tensor):
        step = TensorImage.from_comfy(mask)
        step[step > 0.01] = 1.0
        step[step <= 0.01] = 0.0
        output = TensorImage(step).get_comfy()
        return (output,)

NODE_CLASS_MAPPINGS = { 
    "Bitwise": Bitwise,
    "Ones Like": OnesLike,
    "Zeros Like": ZerosLike,
    "Mask Binary Filter": MaskBinaryFilter,
}