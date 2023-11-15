from .utils import *
from .categories import MODELS_CAT
from ..src.signature.models.lama import Lama


class MagicEraser:
    def __init__(self):
        self.model = Lama()

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {
            "image": ("IMAGE",),
            "mask": ("MASK",),
            }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = MODELS_CAT
    def process(self, image: torch.Tensor, mask: torch.Tensor):
        input_image = image.clone().transpose(3, 1)[0]
        input_mask = mask.clone()
        output: torch.Tensor = self.model.forward(input_image, input_mask)
        output = output.transpose(3, 1)
        return (output,)

NODE_CLASS_MAPPINGS = {
    "Magic Eraser(LaMa)": MagicEraser,
}