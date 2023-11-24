import torch
from kornia.utils import get_cuda_or_mps_device_if_available
import kornia.geometry.transform as K
from .helper import (
    load_jit_model,
)
MODEL_URL = "https://huggingface.co/marcojoao/ISNet/resolve/main/isnet.pt"
MODEL_SHA = "0689398c252bd1275392ea5300204ced42a94181fa15cc7a0cbd42752e5a9648"

class IsNet():
    def __init__(self, device: str|None = None):
        self.device = get_cuda_or_mps_device_if_available()
        self.model = load_jit_model(MODEL_URL, self.device, MODEL_SHA).eval()
        self.infer_size = (1024, 1024)

    def forward(self, image: torch.Tensor):
        input_image = image.to(self.device)
        _,_,H, W = image.shape
        resized_image = K.resize(input_image, self.infer_size, interpolation='bilinear')

        result = self.model(resized_image)[0]
        pred = K.resize(result[0], (H, W), interpolation='bilinear')

        ma = torch.max(pred)
        mi = torch.min(pred)
        pred = torch.Tensor((pred-mi)/(ma-mi))
        return pred
