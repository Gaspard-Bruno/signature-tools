import torch
from .categories import TRANSFORM_CAT
from kornia.geometry.transform import rescale, resize

class RescaleImage:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {
            "image": ("IMAGE",),
            "factor": ("FLOAT", {"default": 2.0, "min": 0.001, "max": 100.0, "step": 0.01}),
            "interpolation": (['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', 'area'],),
            "antialias": ("BOOLEAN", {"default": True},),
            }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = TRANSFORM_CAT
    def process(self, image: torch.Tensor, factor, interpolation, antialias):
        image = image.transpose(3, 1)
        output = rescale(image, factor=factor, interpolation=interpolation, antialias=antialias)
        output = output.transpose(3, 1)
        return (output,)

class ResizeImage:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {
            "image": ("IMAGE",),
            "width": ("INT", {"default": 512}),
            "height": ("INT", {"default": 512}),
            "interpolation": (['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', 'area'],),
            "antialias": ("BOOLEAN", {"default": True},),
            }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = TRANSFORM_CAT
    def process(self, image: torch.Tensor, width, height, interpolation, antialias):
        image = image.transpose(3, 1)
        output = resize(image, size=(width, height), interpolation=interpolation, antialias=antialias)
        output = output.transpose(3, 1)
        return (output,)
class RescaleMask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {
            "mask": ("MASK",),
            "factor": ("FLOAT", {"default": 2.0, "min": 0.001, "max": 100.0, "step": 0.01}),
            "interpolation": (['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', 'area'],),
            "antialias": ("BOOLEAN", {"default": True},),
            }}
    RETURN_TYPES = ("MASK",)
    FUNCTION = "process"
    CATEGORY = TRANSFORM_CAT
    def process(self, mask: torch.Tensor, factor, interpolation, antialias):
        #mask = mask.transpose(3, 1)
        mask = torch.stack([mask])
        output = rescale(mask, factor=factor, interpolation=interpolation, antialias=antialias)
        output = output[0]
        return (output,)

class ResizeMask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {
            "mask": ("MASK",),
            "width": ("INT", {"default": 512}),
            "height": ("INT", {"default": 512}),
            "interpolation": (['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', 'area'],),
            "antialias": ("BOOLEAN", {"default": True},),
            }}
    RETURN_TYPES = ("MASK",)
    FUNCTION = "process"
    CATEGORY = TRANSFORM_CAT
    def process(self, mask: torch.Tensor, width, height, interpolation, antialias):
        mask = torch.stack([mask])
        output = resize(mask, size=(width, height), interpolation=interpolation, antialias=antialias)
        output = output[0]
        return (output,)

NODE_CLASS_MAPPINGS = {
    "Rescale Image": RescaleImage,
    "Resize Image": ResizeImage,
    "Rescale Mask": RescaleMask,
    "Resize Mask": ResizeMask,
}