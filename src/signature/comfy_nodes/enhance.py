import torch
from .categories import ENHANCE_CAT
from kornia.enhance import adjust_brightness, adjust_saturation, equalize, equalize_clahe
import torch

class AdjustBrightness:


    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {"image": ("IMAGE",),
                             "factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                            }
                }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = ENHANCE_CAT

    def process(self, image: torch.Tensor, factor: float):
        output = adjust_brightness(image, factor, clip_output=True)
        return (output,)


class AdjustSaturation:


    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {"image": ("IMAGE",),
                             "factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                            }
                }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = ENHANCE_CAT

    def process(self, image: torch.Tensor, factor: float):
        image = image.transpose(3, 1)
        output = adjust_saturation(image, factor)
        output = output.transpose(3, 1)
        return (output,)

class Equalize:


    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {"image": ("IMAGE",),
                            }
                }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = ENHANCE_CAT

    def process(self, image: torch.Tensor):
        image = image.transpose(3, 1)
        output = equalize(image)
        output = output.transpose(3, 1)
        return (output,)


class EqualizeClahe:


    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {"image": ("IMAGE",)}
                }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = ENHANCE_CAT

    def process(self, image: torch.Tensor):
        image = image.transpose(3, 1)
        output = equalize_clahe(image)
        output = output.transpose(3, 1)
        return (output,)


NODE_CLASS_MAPPINGS = {
    "Adjust Brightness": AdjustBrightness,
    "Adjust Saturation": AdjustSaturation,
    "Equalize": Equalize,
    "Equalize Clahe": EqualizeClahe,
}