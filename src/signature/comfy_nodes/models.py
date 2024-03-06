import torch
from .categories import MODELS_CAT
from ..img.tensor_image import TensorImage
from ..models.lama import Lama
from comfy.model_patcher import ModelPatcher # type: ignore
from ..models.salient_object_detection import SalientObjectDetection
from kornia.utils import get_cuda_or_mps_device_if_available
# from ..src.signature.models.fba_matting import FbaMatting
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

class BackgroundRemoval:
    def __init__(self):
        self.model_name = "isnet"
        self.model: SalientObjectDetection | None = None

    @classmethod
    def INPUT_TYPES(s): # type: ignore
        return {"required": {
            "model_name": (['rmbg14', 'isnet_general'],),
            "image": ("IMAGE",),
            }}
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("rgba", "mask")
    FUNCTION = "process"
    CATEGORY = MODELS_CAT


    def process(self, image: torch.Tensor, model_name: str):
        if model_name != self.model_name or self.model is None:
            self.model = SalientObjectDetection(model_name=model_name)
            self.model_name = model_name

        input_image = TensorImage.from_comfy(image)
        output_masks = self.model.forward(input_image)

        output_cutouts = torch.cat((input_image, output_masks), dim=1)
        output_masks = TensorImage(output_masks).get_comfy()
        output_cutouts = TensorImage(output_cutouts).get_comfy()
        return (output_cutouts, output_masks,)

NODE_CLASS_MAPPINGS = {
    "Magic Eraser": MagicEraser,
    "Background Removal": BackgroundRemoval,
}