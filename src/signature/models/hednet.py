import torch
import kornia.geometry.transform as K
from kornia.utils import get_cuda_or_mps_device_if_available
from .helper import (
    load_jit_model,
)
MODEL_URL = "/resources/repos/annotators/hednet.pt"
MODEL_SHA = ""

class HedNet():
    def __init__(self, device: str|None = None):
        self.device = get_cuda_or_mps_device_if_available()
        self.model = load_jit_model(MODEL_URL, self.device, MODEL_SHA).eval()

    def safe_step(self, x, step=2):
        y = torch.tensor(x, dtype=torch.float32) * float(step + 1)
        y = y.to(torch.int32).to(torch.float32) / float(step)
        return y

    def forward(self, image: torch.Tensor, is_safe: bool = False):
        _, _, W, H = image.shape
        edges = self.model(image.to(self.device) * 255.0)

        edges = [K.resize(e, (W, H), interpolation="bilinear")[0, 0] for e in edges]
        edges = torch.stack(edges, dim=0)
        mean_edges = torch.mean(edges, dim=0)
        edge = 1 / (1 + torch.exp(-mean_edges))
        edge = torch.clip(edge, 0.0, 1.0).unsqueeze(0).unsqueeze(0)
        if is_safe:
            edge = self.safe_step(edge)
        return edge
