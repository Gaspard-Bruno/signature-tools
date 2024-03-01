import torch
from torchvision.transforms.functional import normalize
from kornia.utils import get_cuda_or_mps_device_if_available
import kornia.geometry.transform as K
from kornia.color import rgba_to_rgb
from .helper import (
    load_jit_model,
)
"https://huggingface.co/gaspardbruno/ISNet/resolve/main/"
MODEL_ISNET_URL = "https://huggingface.co/gaspardbruno/ISNet/resolve/main/isnet.pt"
MODEL_ISNET_SHA = "0689398c252bd1275392ea5300204ced42a94181fa15cc7a0cbd42752e5a9648"

MODEL_RMBG14_URL = "https://huggingface.co/gaspardbruno/RMBG-1.4/resolve/main/RMBG14.pt"
MODEL_RMBG14_SHA = "17193bd5ad929dc5e265c4f6671493221c73396decd29e0d12fb008bd9692b9f"

class SalientObjectDetection():
    def __init__(self, model_name: str, device: str|None = None):
        self.device = device or get_cuda_or_mps_device_if_available()
        if model_name == "isnet_general":
            self.model = load_jit_model(MODEL_ISNET_URL, self.device, MODEL_ISNET_SHA).eval()
        elif model_name == "rmbg14":
            self.model = load_jit_model(MODEL_RMBG14_URL, self.device, MODEL_RMBG14_SHA).eval()
        else :
            raise ValueError("Model name should be either 'isnet' or 'rmbg14'")
        self.infer_size = (1024, 1024)

    def forward(self, image: torch.Tensor):
        input_image = image.to(self.device)
        if image.shape[1] == 4:
            input_image = rgba_to_rgb(input_image)

        _,_,H, W = image.shape
        resized_image = K.resize(input_image, self.infer_size, interpolation='bilinear', align_corners=False)
        resized_image = normalize(resized_image,[0.5,0.5,0.5],[1.0,1.0,1.0])
        result = self.model(resized_image)[0]
        pred = K.resize(result[0], (H, W), interpolation='bilinear', align_corners=False)

        ma = torch.max(pred)
        mi = torch.min(pred)
        result = (pred-mi)/(ma-mi)
        return pred
