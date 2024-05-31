import torch
from kornia.utils import get_cuda_or_mps_device_if_available
from .helper import (
    load_jit_model,
)
MODEL_URL = "https://huggingface.co/gaspardbruno/seemore/resolve/main/seemore_t_x4.pt"
MODEL_SHA = "9d576fad76a46580582afa22595ee11e2fa7ce362ad9b12e38ce8d7486d0a6f0"

class SeeMore():
    def __init__(self, device: str|None = None):
        self.device = get_cuda_or_mps_device_if_available()
        self.model = load_jit_model(MODEL_URL, self.device, MODEL_SHA).eval()

    def forward(self, image: torch.Tensor):
        return self.model(image.to(self.device))