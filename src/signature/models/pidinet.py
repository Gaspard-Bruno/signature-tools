import torch
from kornia.utils import get_cuda_or_mps_device_if_available
from .helper import (
    load_jit_model,
)
MODEL_URL = "https://huggingface.co/gaspardbruno/Pidinet/resolve/main/pidinet.pt"
MODEL_SHA = "800ca4bc576210fefe25c9348cce2a329c34844087e7e12a8658bc8c978d62c2"

class PidiNet():
    def __init__(self, device: str|None = None):
        self.device = get_cuda_or_mps_device_if_available()
        print(f"Device: {self.device}")
        self.model = load_jit_model(MODEL_URL, self.device, MODEL_SHA).eval()

    def safe_step(self, x, step=2):
        y = torch.tensor(x, dtype=torch.float32) * float(step + 1)
        y = y.to(torch.int32).to(torch.float32) / float(step)
        return y

    def forward(self, image: torch.Tensor, is_safe: bool = False, apply_filter=False):

        edge = self.model(image.to(self.device))
        if apply_filter:
            edge = edge > 0.5
        if is_safe:
            edge = self.safe_step(edge)
        return edge
