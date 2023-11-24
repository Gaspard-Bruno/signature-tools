import torch
from kornia.utils import get_mps_device_if_available
from .helper import (
    load_jit_model,
)
MODEL_URL = "/resources/repos/annotators/lineart_anime.pt"
MODEL_SHA = ""

class LineArtAnime():
    def __init__(self, device: str|None = None):
        self.device = get_mps_device_if_available()
        self.model = load_jit_model(MODEL_URL, self.device, MODEL_SHA).eval()

    def forward(self, image: torch.Tensor):
        input_image = image.to(self.device)
        result = self.model(input_image)
        result = torch.clip(result, 0.0, 1.0)
        return result