from .utils import *
from .categories import TRANSFORM_CAT
from kornia.geometry.transform import scale

class ScaleByFactor:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {
            "image": ("IMAGE",),
            "factor": ("FLOAT", {"default": 2.0, "min": 0.001, "max": 100.0, "step": 0.01}),
             }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = TRANSFORM_CAT
    def process(self, image, factor):
        image = image.transpose(3, 1)
        output = scale(image, scale_factor=factor)
        output = output.transpose(3, 1)
        return (image,)

NODE_CLASS_MAPPINGS = {
    "Load from Web": ScaleByFactor,
}