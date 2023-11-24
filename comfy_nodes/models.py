import torch
from .categories import MODELS_CAT
from  ..src.signature.img.tensor_image import TensorImage
from ..src.signature.models.lama import Lama
from ..src.signature.models.isnet import IsNet

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
        input_image = TensorImage.from_comfy(image)
        input_mask = TensorImage.from_comfy(mask)

        highres = self.model.forward(input_image, input_mask, mode)
        highres = TensorImage(highres).get_comfy()

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
        input_image = TensorImage.from_comfy(image)
        outputs = self.model.forward(input_image)
        print(outputs.shape)
        outputs = TensorImage(outputs).get_comfy()
        print(outputs.shape)
        return (outputs,)

NODE_CLASS_MAPPINGS = {
    "Magic Eraser(LaMa)": MagicEraser,
    "Salient Object Detection(IsNet)": SalientObjectDetection,
}