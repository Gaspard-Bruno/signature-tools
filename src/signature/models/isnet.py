import torch

import kornia.geometry.transform as K
from .helper import (
    load_jit_model,
)
MODEL_URL = "https://huggingface.co/marcojoao/ISNet/resolve/main/isnet.pt"
MODEL_SHA = "0689398c252bd1275392ea5300204ced42a94181fa15cc7a0cbd42752e5a9648"

class IsNet():
    def __init__(self, device: str|None = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = load_jit_model(MODEL_URL, self.device, MODEL_SHA).eval()
        self.infer_size = (1024, 1024)

    def forward(self, image: torch.Tensor):
        original_shape = image.shape[2:]
        resized_image = K.resize(image, self.infer_size, interpolation='bilinear')

        result = self.model(resized_image.to(self.device))[0]
        pred = K.resize(result[0], original_shape, interpolation='bilinear')

        ma = torch.max(pred)
        mi = torch.min(pred)
        pred = torch.Tensor((pred-mi)/(ma-mi))
        return pred
