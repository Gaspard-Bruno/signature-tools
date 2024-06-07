from ..img.tensor_image import TensorImage
from kornia.color import rgba_to_rgb, rgb_to_hls, rgb_to_hsv, hsv_to_rgb, hls_to_rgb
from .categories import MISC_CAT
import torch

class AnyType(str):
  def __ne__(self, __value: object) -> bool:
    return False
any = AnyType("*")


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


class Ones():

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {"width": ("INT", {"default": 1024}),
                             "height": ("INT", {"default": 1024}),
                             "channels": ("INT", {"default": 1, "min": 1, "max": 4}),
                             "batch": ("INT", {"default": 1})}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = MISC_CAT

    def process(self, width: int, height: int, channels: int, batch: int):
        step = torch.ones((batch, channels, height, width))
        output_image = TensorImage(step).get_comfy()
        return (output_image,)


class Zeros():

        @classmethod
        def INPUT_TYPES(s): # type: ignore
            return {"required": {"width": ("INT", {"default": 1024}),
                                "height": ("INT", {"default": 1024}),
                                "channels": ("INT", {"default": 1}),
                                "batch": ("INT", {"default": 1})}}
        RETURN_TYPES = ("IMAGE",)
        FUNCTION = "process"
        CATEGORY = MISC_CAT

        def process(self, width: int, height: int, channels: int, batch: int):
            step = torch.zeros((batch, channels, height, width))
            output_image = TensorImage(step).get_comfy()
            return (output_image,)

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

class AnyToString():
    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {
            "input": (any,),
            }}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    CATEGORY = MISC_CAT
    def process(self, input):
        return (str(input),)

class RGB2HSL():
    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {"image": ("IMAGE",)}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = MISC_CAT

    def process(self, image: torch.Tensor):
        input_image = TensorImage.from_comfy(image)
        if input_image.shape[1] == 4:
            input_image = rgba_to_rgb(input_image)
        value = rgb_to_hls(input_image)
        output_image = TensorImage(value).get_comfy()
        return (output_image,)

class RGB2HSV():
    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {"image": ("IMAGE",)}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = MISC_CAT

    def process(self, image: torch.Tensor):
        input_image = TensorImage.from_comfy(image)
        if input_image.shape[1] == 4:
            input_image = rgba_to_rgb(input_image)
        value = rgb_to_hsv(input_image)
        output_image = TensorImage(value).get_comfy()
        return (output_image,)

class HSL2RGB():
    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {"image": ("IMAGE",)}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = MISC_CAT

    def process(self, image: torch.Tensor):
        input_image = TensorImage.from_comfy(image)
        value = hls_to_rgb(input_image)
        output_image = TensorImage(value).get_comfy()
        return (output_image,)

class HSV2RGB():
    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {"image": ("IMAGE",)}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = MISC_CAT

    def process(self, image: torch.Tensor):
        input_image = TensorImage.from_comfy(image)
        value = hsv_to_rgb(input_image)
        output_image = TensorImage(value).get_comfy()
        return (output_image,)

class MaskDistance():
    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {"mask_0": ("MASK",), "mask_1": ("MASK",)}}
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "process"
    CATEGORY = MISC_CAT

    def process(self, mask_0: torch.Tensor, mask_1: torch.Tensor):
        tensor1 = TensorImage.from_comfy(mask_0)
        tensor2 = TensorImage.from_comfy(mask_1)
        dist = torch.Tensor((tensor1 - tensor2).pow(2).sum(3).sqrt().mean())
        return (dist,)


NODE_CLASS_MAPPINGS = { 
    "Any to String": AnyToString,
    "Bitwise": Bitwise,
    "Ones": Ones,
    "Zeros": Zeros,
    "Ones Like": OnesLike,
    "Zeros Like": ZerosLike,
    "Mask Binary Filter": MaskBinaryFilter,
    "Rgb2Hsl": RGB2HSL,
    "Rgb2Hsv": RGB2HSV,
    "Hsl2Rgb": HSL2RGB,
    "Hsv2Rgb": HSV2RGB,
    "MaskDistance": MaskDistance
}