from .utils import *
from .categories import MODELS_CAT
from ..src.signature.models.lama import Lama
from ..src.signature.models.isnet import IsNet
from . import helper

class MagicEraser:
    def __init__(self):
        self.model = Lama()

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {
            "image": ("IMAGE",),
            "mask": ("MASK",),
            "mode": (['CROP', 'FULL', 'FIXED'],),
            }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = MODELS_CAT

    def process(self, image: torch.Tensor, mask: torch.Tensor, mode: str):
        input_image = helper.comfy_img_to_torch(image)
        input_mask = helper.comfy_mask_to_torch(mask)

        highres = self.model.forward(input_image, input_mask, mode)
        highres = helper.torch_img_to_comfy(highres)

        return (highres,)

class SalientObjectDetection:
    def __init__(self):
        self.model = IsNet()

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {
            "image": ("IMAGE",),
            }}
    RETURN_TYPES = ("MASK",)
    FUNCTION = "process"
    CATEGORY = MODELS_CAT

    def process(self, image: torch.Tensor):
        input_image = helper.comfy_img_to_torch(image)
        outputs = self.model.forward(input_image)
        outputs = helper.torch_mask_to_comfy(outputs)
        return (outputs,)

NODE_CLASS_MAPPINGS = {
    "Magic Eraser(LaMa)": MagicEraser,
    "Salient Object Detection(IsNet)": SalientObjectDetection,
}