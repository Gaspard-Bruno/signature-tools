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
            "mode": (["bilinear", "nearest"],)
            }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = TRANSFORM_CAT
    def process(self, image: torch.Tensor, factor, mode):
        image = image.transpose(3, 1)
        tensor_factor = torch.tensor([[factor, factor]], device=image.device)
        output = scale(image, scale_factor=tensor_factor, mode=mode, align_corners=True)
        output = output.transpose(3, 1)
        return (output,)

NODE_CLASS_MAPPINGS = {
    "Scale by Factor": ScaleByFactor,
}