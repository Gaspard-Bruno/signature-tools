import torch
from kornia.utils import get_mps_device_if_available
from .helper import (
    load_jit_model,
)
REALISTIC_MODEL_URL = "https://huggingface.co/marcojoao/LineArt/resolve/main/lineart_realistic.pt"
REALISTIC_MODEL_SHA = "998eb87dda8905124d1e7609689cca57f9f0f616b2c66a70d5753f19ea5cdf3a"

COARSE_MODEL_URL = "https://huggingface.co/marcojoao/LineArt/resolve/main/lineart_coarse.pt"
COARSE_MODEL_SHA = "2d3e05100c9e81220b22ca4d41eb1b536de003973e5dcdfa749306a6dd9e8f0d"

class LineArt():
    def __init__(self, device: str|None = None):
        self.device = get_mps_device_if_available()
        self.realistic_model = load_jit_model(REALISTIC_MODEL_URL, self.device, REALISTIC_MODEL_SHA).eval()
        self.coarse_model = load_jit_model(COARSE_MODEL_URL, self.device, COARSE_MODEL_SHA).eval()

    def forward(self, image: torch.Tensor, mode: str = 'realistic'):
        if mode == 'realistic':
            return self.realistic_model(image.to(self.device))
        elif mode == 'coarse':
            return self.coarse_model(image.to(self.device))
        else:
            raise NotImplementedError(f"Mode {mode} is not implemented.")
